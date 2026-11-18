//! Optional native GPU backend (Vulkan / Metal / DX12 / GL through `wgpu`).
//!
//! Offloads the large-M int8 GEMMs (encoder, batched decoding) to a tiled WGSL compute
//! shader. Activations are quantised on the CPU and uploaded per call, so only matrices with
//! enough rows to amortise the transfer are dispatched here (see [`MIN_ROWS`]).

use std::sync::{Mutex, OnceLock};

use crate::kernels::PackedB;

/// Below this many rows the transfer overhead outweighs the GPU win and the CPU path is used.
pub const MIN_ROWS: usize = 48;

/// Output tile per workgroup (rows and columns); each of the 8x8 threads computes a 4x4 block.
const BLOCK: u32 = 32;
/// K words (4 int8 each) staged in workgroup memory per iteration.
const KW: u32 = 8;

fn shader_source(use_dot4: bool) -> String {
    let (req, dot) = if use_dot4 {
        ("requires packed_4x8_integer_dot_product;\n", "fn dot4(x: u32, y: u32) -> i32 { return dot4I8Packed(x, y); }")
    } else {
        (
            "",
            "fn sx(x: u32, i: u32) -> i32 { return extractBits(bitcast<i32>(x), i * 8u, 8u); }\n\
             fn dot4(x: u32, y: u32) -> i32 { return sx(x,0u)*sx(y,0u) + sx(x,1u)*sx(y,1u) + sx(x,2u)*sx(y,2u) + sx(x,3u)*sx(y,3u); }",
        )
    };
    format!(
        r#"{req}
struct P {{ m: u32, n: u32, kw: u32, has_bias: u32, scale: f32, pad0: u32, pad1: u32, pad2: u32 }}
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
{dot}
var<workgroup> ta: array<u32, {sz}>;
var<workgroup> tb: array<u32, {sz}>;
@compute @workgroup_size(8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
  let tid = lid.y * 8u + lid.x;
  let row0 = wid.y * {bl}u;
  let col0 = wid.x * {bl}u;
  var acc: array<i32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) {{ acc[i] = 0; }}
  let steps = (p.kw + {kw}u - 1u) / {kw}u;
  for (var s = 0u; s < steps; s = s + 1u) {{
    let k0 = s * {kw}u;
    for (var j = 0u; j < 4u; j = j + 1u) {{
      let idx = tid * 4u + j;
      let r = idx / {kw}u;
      let w = idx % {kw}u;
      let kk = k0 + w;
      var av = 0u;
      if (row0 + r < p.m && kk < p.kw) {{ av = a[(row0 + r) * p.kw + kk]; }}
      ta[idx] = av;
      var bv = 0u;
      if (col0 + r < p.n && kk < p.kw) {{ bv = b[(col0 + r) * p.kw + kk]; }}
      tb[idx] = bv;
    }}
    workgroupBarrier();
    for (var w = 0u; w < {kw}u; w = w + 1u) {{
      var ra: array<u32, 4>;
      var rb: array<u32, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) {{
        ra[i] = ta[(lid.y + 8u * i) * {kw}u + w];
        rb[i] = tb[(lid.x + 8u * i) * {kw}u + w];
      }}
      for (var i = 0u; i < 4u; i = i + 1u) {{
        for (var j = 0u; j < 4u; j = j + 1u) {{
          acc[i * 4u + j] = acc[i * 4u + j] + dot4(ra[i], rb[j]);
        }}
      }}
    }}
    workgroupBarrier();
  }}
  for (var i = 0u; i < 4u; i = i + 1u) {{
    let row = row0 + lid.y + 8u * i;
    for (var j = 0u; j < 4u; j = j + 1u) {{
      let col = col0 + lid.x + 8u * j;
      if (row < p.m && col < p.n) {{
        var v = f32(acc[i * 4u + j]) * p.scale;
        if (p.has_bias == 1u) {{ v = v + bias[col]; }}
        c[row * p.n + col] = v;
      }}
    }}
  }}
}}
"#,
        req = req,
        dot = dot,
        bl = BLOCK,
        kw = KW,
        sz = BLOCK * KW
    )
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
    pub backend: String,
    pub hw_dot4: bool,
    scratch: Mutex<Scratch>,
}

/// Grow-only buffers reused across calls (no per-GEMM allocation).
struct Scratch {
    a: Option<(wgpu::Buffer, u64)>,
    bias: Option<(wgpu::Buffer, u64)>,
    out: Option<(wgpu::Buffer, u64)>,
    read: Option<(wgpu::Buffer, u64)>,
    params: wgpu::Buffer,
}

fn ensure(dev: &wgpu::Device, slot: &mut Option<(wgpu::Buffer, u64)>, need: u64, usage: wgpu::BufferUsages) {
    if slot.as_ref().map_or(true, |(_, cap)| *cap < need) {
        let cap = need.next_power_of_two().max(256);
        let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: cap, usage, mapped_at_creation: false });
        *slot = Some((buf, cap));
    }
}

pub struct GpuWeights {
    buf: wgpu::Buffer,
    n: u32,
    kw: u32,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// Initialise the global GPU context. Returns a description of the adapter, or an error string.
pub fn enable() -> Result<String, String> {
    let g = GPU.get_or_init(Gpu::new);
    match g {
        Some(g) => Ok(format!("{} ({}, int8 dot4 {})", g.adapter_name, g.backend, if g.hw_dot4 { "native" } else { "emulated" })),
        None => Err("no usable GPU adapter found".into()),
    }
}

pub fn get() -> Option<&'static Gpu> {
    GPU.get().and_then(|g| g.as_ref())
}

impl Gpu {
    fn new() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let hw_dot4 = instance.wgsl_language_features().contains(wgpu::WgslLanguageFeatures::Packed4x8IntegerDotProduct);
        let make = |dot4: bool| -> Option<wgpu::ComputePipeline> {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rustberger-gemm"),
                source: wgpu::ShaderSource::Wgsl(shader_source(dot4).into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rustberger-gemm"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            match pollster::block_on(scope.pop()) {
                Some(_) => None,
                None => Some(pipeline),
            }
        };
        let (pipeline, hw_dot4) = match hw_dot4.then(|| make(true)).flatten() {
            Some(p) => (p, true),
            None => (make(false)?, false),
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let scratch = Mutex::new(Scratch { a: None, bias: None, out: None, read: None, params });
        Some(Gpu { device, queue, pipeline, adapter_name: info.name, backend: format!("{:?}", info.backend), hw_dot4, scratch })
    }

    pub fn upload(&self, b: &PackedB) -> GpuWeights {
        use wgpu::util::DeviceExt;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const u8, b.data.len()) };
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weights"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        GpuWeights { buf, n: b.n as u32, kw: (b.kp / 4) as u32 }
    }

    /// `a` is `m x kp` int8 (row stride = `kp`). Returns `m x n` f32 = acc * scale + bias.
    pub fn gemm(&self, a: &[i8], m: usize, w: &GpuWeights, scale: f32, bias: Option<&[f32]>) -> Vec<f32> {
        use wgpu::BufferUsages as U;
        let n = w.n as usize;
        let a_len = m * w.kw as usize * 4;
        let a_bytes: &[u8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a_len) };
        let out_size = (m * n * 4) as u64;
        let mut sc = self.scratch.lock().unwrap();
        ensure(&self.device, &mut sc.a, a_len as u64, U::STORAGE | U::COPY_DST);
        ensure(&self.device, &mut sc.bias, (n.max(1) * 4) as u64, U::STORAGE | U::COPY_DST);
        ensure(&self.device, &mut sc.out, out_size, U::STORAGE | U::COPY_SRC);
        ensure(&self.device, &mut sc.read, out_size, U::MAP_READ | U::COPY_DST);
        self.queue.write_buffer(&sc.a.as_ref().unwrap().0, 0, a_bytes);
        if let Some(b) = bias {
            let bb: &[u8] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) };
            self.queue.write_buffer(&sc.bias.as_ref().unwrap().0, 0, bb);
        }
        let mut params = [0u8; 32];
        params[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params[4..8].copy_from_slice(&w.n.to_le_bytes());
        params[8..12].copy_from_slice(&w.kw.to_le_bytes());
        params[12..16].copy_from_slice(&(bias.is_some() as u32).to_le_bytes());
        params[16..20].copy_from_slice(&scale.to_le_bytes());
        self.queue.write_buffer(&sc.params, 0, &params);

        let (a_buf, bias_buf, out_buf, read_buf) = (&sc.a.as_ref().unwrap().0, &sc.bias.as_ref().unwrap().0, &sc.out.as_ref().unwrap().0, &sc.read.as_ref().unwrap().0);
        let bgl = self.pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sc.params.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((w.n + BLOCK - 1) / BLOCK, (m as u32 + BLOCK - 1) / BLOCK, 1);
        }
        enc.copy_buffer_to_buffer(out_buf, 0, read_buf, 0, out_size);
        self.queue.submit(Some(enc.finish()));
        let slice = read_buf.slice(..out_size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("map failed");
        let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        read_buf.unmap();
        out
    }
}
