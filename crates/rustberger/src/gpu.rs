//! Optional native GPU backend (Vulkan / Metal / DX12 / GL through `wgpu`).
//!
//! Offloads the large-M int8 GEMMs (encoder, batched decoding) to a tiled WGSL compute
//! shader. Activations are quantised on the CPU and uploaded per call, so only matrices with
//! enough rows to amortise the transfer are dispatched here (see [`MIN_ROWS`]).

use std::sync::OnceLock;

use crate::kernels::PackedB;

/// Below this many rows the transfer overhead outweighs the GPU win and the CPU path is used.
pub const MIN_ROWS: usize = 48;

const TILE: u32 = 16;

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
@compute @workgroup_size({t}, {t})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
  let row = wid.y * {t}u + lid.y;
  let col = wid.x * {t}u + lid.x;
  var acc: i32 = 0;
  let tiles = (p.kw + {t}u - 1u) / {t}u;
  for (var t = 0u; t < tiles; t = t + 1u) {{
    let kw0 = t * {t}u;
    let kk = kw0 + lid.x;
    var av = 0u;
    if (row < p.m && kk < p.kw) {{ av = a[row * p.kw + kk]; }}
    ta[lid.y * {t}u + lid.x] = av;
    let brow = wid.x * {t}u + lid.y;
    var bv = 0u;
    if (brow < p.n && kk < p.kw) {{ bv = b[brow * p.kw + kk]; }}
    tb[lid.y * {t}u + lid.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < {t}u; k = k + 1u) {{
      acc = acc + dot4(ta[lid.y * {t}u + k], tb[lid.x * {t}u + k]);
    }}
    workgroupBarrier();
  }}
  if (row < p.m && col < p.n) {{
    var v = f32(acc) * p.scale;
    if (p.has_bias == 1u) {{ v = v + bias[col]; }}
    c[row * p.n + col] = v;
  }}
}}
"#,
        req = req,
        dot = dot,
        t = TILE,
        sz = TILE * TILE
    )
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub adapter_name: String,
    pub backend: String,
    pub hw_dot4: bool,
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
        Some(Gpu { device, queue, pipeline, adapter_name: info.name, backend: format!("{:?}", info.backend), hw_dot4 })
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
        use wgpu::util::DeviceExt;
        let n = w.n as usize;
        let a_bytes: &[u8] = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, m * w.kw as usize * 4) };
        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: a_bytes, usage: wgpu::BufferUsages::STORAGE });
        let zero = [0f32; 1];
        let bias_bytes: &[u8] = match bias {
            Some(b) => unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 4) },
            None => unsafe { std::slice::from_raw_parts(zero.as_ptr() as *const u8, 4) },
        };
        let bias_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bias_bytes, usage: wgpu::BufferUsages::STORAGE });
        let out_size = (m * n * 4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: out_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut params = [0u8; 32];
        params[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params[4..8].copy_from_slice(&w.n.to_le_bytes());
        params[8..12].copy_from_slice(&w.kw.to_le_bytes());
        params[12..16].copy_from_slice(&(bias.is_some() as u32).to_le_bytes());
        params[16..20].copy_from_slice(&scale.to_le_bytes());
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &params, usage: wgpu::BufferUsages::UNIFORM });

        let bgl = self.pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: p_buf.as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((w.n + TILE - 1) / TILE, (m as u32 + TILE - 1) / TILE, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
        self.queue.submit(Some(enc.finish()));
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("map failed");
        let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        read_buf.unmap();
        out
    }
}
