use std::sync::mpsc;
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::{util::DeviceExt, Device, Queue};
//TODO implement triangles
type RotationMatrix = [[f32; 4]; 4];
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Point2D { x: f32, y: f32 }
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Point3D { x: f32, y: f32, z: f32, }
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Rotation3D { pitch: f32, yaw: f32, roll: f32, }
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default, PartialEq)]
struct GpuInput { x: f32, y: f32, z: f32, w: f32}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Triangle3D {
    a: Point3D,
    b: Point3D,
    c: Point3D,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Triangle2D {
    a: Point2D,
    b: Point2D,
    c: Point2D
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProjectUniform {
    rotation: [[f32; 4]; 4],
    camera_position: [f32; 4],
    clip: [f32; 4],
    scale: [f32; 4],
    metadata: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuProjectOutput {
    x: f32,
    y: f32,
    valid: u32,
    _pad: u32,
}

const GPU_WORKGROUP_SIZE: u32 = 64;
const GPU_PROJECT_SHADER: &str = include_str!("project.wgsl");
#[derive(Debug, Default)]
pub(crate) struct Camera { rotation: Rotation3D, position: Point3D, }
#[derive(Debug, Clone)]
pub(crate) struct ProjectionOptions {
    fov: f32,
    fov_epsilon: f32,
    near_clip: f32,
    far_clip: f32,
    aspect: f32,
    screen_width: u32,
    screen_height: u32,
}
pub(crate) struct GpuOptions {
    device: Device,
    queue: Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) enum ProjectionDevice {
    Cpu,
    Gpu(GpuOptions)
}
#[derive(Debug, PartialEq)]
pub(crate) enum Error {
    UnacceptableFOV,
    PointTooClose,
    PointTooFar
}

impl GpuOptions {
    pub(crate) fn new(device: Device, queue: Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projector_gpu_projection_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_PROJECT_SHADER.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("projector_gpu_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("projector_gpu_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("projector_gpu_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("project"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            bind_group_layout,
            pipeline,
        }
    }
}

impl Point2D { fn from(x: f32, y: f32) -> Self { Self { x, y } } }
impl Point3D { fn from(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } } }
impl From<Point3D> for GpuInput {
    fn from(v: Point3D) -> Self {
        GpuInput { x: v.x, y: v.y, z: v.z, w: 1.0 }
    }
}
impl From<GpuInput> for Point3D {
    fn from(v: GpuInput) -> Self {
        Point3D { x: v.x, y: v.y, z: v.z }
    }
}
impl From<GpuProjectOutput> for Point2D {
    fn from(value: GpuProjectOutput) -> Self {
        Point2D { x: value.x, y: value.y }
    }
}
impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            fov: 2.0, fov_epsilon: 0.0, near_clip: 0.1, far_clip: 100.0,
            aspect: 1.0, screen_width: 100, screen_height: 100
        }
    }
}
impl Triangle3D {
    pub(crate) fn new(a: Point3D, b: Point3D, c: Point3D) -> Self {
        Self { a, b, c }
    }
    pub(crate) fn vertices(&self) -> [Point3D; 3] {
        [self.a, self.b, self.c]
    }
    pub(crate) fn project(
        &self,
        mat: RotationMatrix,
        camera: &Camera,
        proj: &ProjectionOptions,
    ) -> Result<Triangle2D, Error> {
        project_triangle(mat, camera, self, proj)
    }
    pub(crate) fn pack(triangles: &[Triangle3D]) -> Vec<Point3D> {
        let mut points: Vec<Point3D> = Vec::with_capacity(triangles.len() * 3);
        for tri in triangles {
            points.push(tri.a);
            points.push(tri.b);
            points.push(tri.c);
        }
        points
    }
    pub(crate) fn unpack(points: &[Point3D]) -> Vec<Triangle3D> {
        points.chunks_exact(3).map(|a| Triangle3D { a: a[0], b: a[1], c: a[2] }).collect()
    }
}

impl Triangle2D {
    pub(crate) fn new(a: Point2D, b: Point2D, c: Point2D) -> Self {
        Self { a, b, c }
    }
    pub(crate) fn vertices(&self) -> [Point2D; 3] {
        [self.a, self.b, self.c]
    }
    pub(crate) fn pack(triangles: &[Triangle2D]) -> Vec<Point2D> {
        let mut points: Vec<Point2D> = Vec::with_capacity(triangles.len() * 3);
        for tri in triangles {
            points.push(tri.a);
            points.push(tri.b);
            points.push(tri.c);
        }
        points
    }
    pub(crate) fn unpack(points: &[Point2D]) -> Vec<Triangle2D> {
        points.chunks_exact(3).map(|a| Triangle2D { a: a[0], b: a[1], c: a[2] }).collect()
    }
}
impl ProjectionDevice {
    pub(crate) fn project(&self, camera: &Camera, positions: &[Point3D], proj: &ProjectionOptions) -> Vec<Option<Point2D>> {
        match self {
            ProjectionDevice::Cpu => Self::project_batch_cpu(camera, positions, proj),
            ProjectionDevice::Gpu(opts) => Self::project_batch_gpu(opts, camera, positions, proj),
        }
    }
    fn project_batch_cpu(camera: &Camera, positions: &[Point3D], proj: &ProjectionOptions) -> Vec<Option<Point2D>> {
        positions.par_iter()
            .map(|v| project(calculate_matrix(&camera.rotation), camera, v, proj).ok())
            .collect()
    }

    fn project_batch_gpu(
        opts: &GpuOptions,
        camera: &Camera,
        positions: &[Point3D],
        proj: &ProjectionOptions,
    ) -> Vec<Option<Point2D>> {
        if positions.is_empty() {
            return Vec::new();
        }
        if positions.len() > u32::MAX as usize {
            return Self::project_batch_cpu(camera, positions, proj);
        }
        if proj.fov <= proj.fov_epsilon
            || proj.fov >= std::f32::consts::PI - proj.fov_epsilon
        {
            return vec![None; positions.len()];
        }

        let rotation = calculate_matrix(&camera.rotation);
        let focal = 1.0 / (proj.fov * 0.5).tan();
        let scale_x = (focal / proj.aspect) * (proj.screen_width as f32 / 2.0);
        let scale_y = focal * (proj.screen_height as f32 / 2.0);

        let uniform = GpuProjectUniform {
            rotation,
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            clip: [proj.near_clip, proj.far_clip, 0.0, 0.0],
            scale: [scale_x, scale_y, 0.0, 0.0],
            metadata: [positions.len() as u32, 0, 0, 0],
        };

        let input: Vec<GpuInput> = positions
            .iter()
            .map(|p| GpuInput {
                x: p.x,
                y: p.y,
                z: p.z,
                w: 1.0,
            })
            .collect();

        match dispatch_gpu_projection(opts, &input, &uniform) {
            Ok(result) => result,
            Err(_) => Self::project_batch_cpu(camera, positions, proj),
        }
    }
}


fn calculate_matrix(camera: &Rotation3D) -> RotationMatrix {
    let (sp, cp) = camera.pitch.sin_cos();
    let (st, ct) = camera.yaw.sin_cos();
    let (ss, cs) = camera.roll.sin_cos();
    [[ct*cs, cs*st*sp-ss*cp, cs*st*cp+ss*sp, 0.0],
     [ct*ss, ss*st*sp+cs*cp, ss*st*cp-cs*sp, 0.0],
     [-st, ct*sp, ct*cp, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

pub(crate) fn project(mat: RotationMatrix, camera: &Camera, position: &Point3D, proj: &ProjectionOptions) -> Result<Point2D, Error> {
    // let mat = calculate_matrix(&camera.rotation);
    let dx = position.x - camera.position.x;
    let dy = position.y - camera.position.y;
    let dz = position.z - camera.position.z;
    let cx = dx * mat[0][0] + dy * mat[0][1] + dz * mat[0][2];
    let cy = dx * mat[1][0] + dy * mat[1][1] + dz * mat[1][2];
    let cz = dx * mat[2][0] + dy * mat[2][1] + dz * mat[2][2];
    if cz < proj.near_clip { return Err(Error::PointTooClose); };
    if cz > proj.far_clip { return Err(Error::PointTooFar); };
    if proj.fov <= proj.fov_epsilon || proj.fov >= std::f32::consts::PI - proj.fov_epsilon { return Err(Error::UnacceptableFOV); }
    let focal = 1.0 / (proj.fov * 0.5).tan();
    let px = (focal / proj.aspect) * (cx / cz);
    let py = focal * (cy / cz);
    Ok( Point2D { x: px * proj.screen_width as f32 / 2.0, y: py * proj.screen_height as f32 / 2.0 } )
}

pub(crate) fn project_triangle(mat: RotationMatrix, camera: &Camera, triangle: &Triangle3D, proj: &ProjectionOptions) -> Result<Triangle2D, Error> {
    Ok(Triangle2D {
        a: project(mat, camera, &triangle.a, proj)?,
        b: project(mat, camera, &triangle.b, proj)?,
        c: project(mat, camera, &triangle.c, proj)?
    })
}

fn dispatch_gpu_projection(
    opts: &GpuOptions,
    payload: &[GpuInput],
    uniforms: &GpuProjectUniform,
) -> Result<Vec<Option<Point2D>>, wgpu::BufferAsyncError> {
    let device = &opts.device;
    let queue = &opts.queue;

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("projector_points_input"),
        contents: bytemuck::cast_slice(payload),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("projector_uniform_buffer"),
        contents: bytemuck::bytes_of(uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_size = (payload.len() * size_of::<GpuProjectOutput>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("projector_output_buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("projector_readback_buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("projector_gpu_bind_group"),
        layout: &opts.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("projector_gpu_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("projector_gpu_compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&opts.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups =
            ((payload.len() as u32) + GPU_WORKGROUP_SIZE - 1) / GPU_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).expect("map_async send failed");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("gpu projection poll failed");
    rx.recv().expect("map_async channel closed")?;

    let mapped = slice.get_mapped_range();
    let gpu_results: &[GpuProjectOutput] = bytemuck::cast_slice(&mapped);
    let mut projected = Vec::with_capacity(gpu_results.len());
    for entry in gpu_results {
        if entry.valid == 0 {
            projected.push(None);
        } else {
            projected.push(Some(Point2D { x: entry.x, y: entry.y }));
        }
    }
    drop(mapped);
    readback_buffer.unmap();

    Ok(projected)
}

/// Copy a `GpuInput` slice on the GPU and synchronously read the results back.
fn gpu_copy_roundtrip(device: &Device, queue: &Queue, payload: &[GpuInput]) -> Result<Vec<GpuInput>, wgpu::BufferAsyncError> {
    if payload.is_empty() {
        return Ok(Vec::new());
    }

    let payload_bytes = bytemuck::cast_slice(payload);
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("projector_gpu_input"),
        contents: payload_bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("projector_gpu_output"),
        size: payload_bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("projector_gpu_copy_encoder"),
    });
    encoder.copy_buffer_to_buffer(&input, 0, &output, 0, payload_bytes.len() as u64);
    queue.submit(Some(encoder.finish()));

    let output_slice = output.slice(..);
    let (tx, rx) = mpsc::channel();
    output_slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).expect("map_async callback send failure");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll for readback");
    rx.recv().expect("map_async channel receive failure")?;

    let mapped = output_slice.get_mapped_range();
    let mut result = vec![GpuInput::default(); payload.len()];
    result.copy_from_slice(bytemuck::cast_slice(&mapped));
    drop(mapped);
    output.unmap();
    Ok(result)
}



#[cfg(test)]
mod tests {
    use crate::projector::Camera;
    use super::*;
    use std::mem::{align_of, size_of};
    use approx::assert_abs_diff_eq;

    fn request_test_device() -> Option<(Device, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = match pollster::block_on(
            instance.request_adapter(
                //TODO
                // note to future self: if something is lagging, try this.
                // power_preference: wgpu::PowerPreference::HighPerformance
                &wgpu::RequestAdapterOptions::default()
            ),
        ) {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!("Skipping GPU-dependent test: {err}");
                return None;
            }
        };
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
            Ok(result) => Some(result),
            Err(err) => {
                eprintln!("Skipping GPU-dependent test: {err}");
                None
            }
        }
    }

    fn generate_test_points() -> Vec<Point3D> {
        (0..255)
            .map(|i| {
                Point3D::from(
                    i as f32 * 0.1,
                    (i as f32 * 0.2).sin(),
                    1.0 + (i % 5) as f32,
                )
            })
            .collect()
    }

    fn generate_test_points_as_gpu_inputs() -> Vec<GpuInput> {
        (0..255)
            .map(|i| {
                Point3D::from(
                    i as f32 * 0.1,
                    (i as f32 * 0.2).sin(),
                    1.0 + (i % 5) as f32,
                ).into()
            })
            .collect()
    }


    #[test]
    pub fn camera_forward_is_screen_center () {
        assert_eq!(
            project(calculate_matrix(&Camera::default().rotation), &Camera::default(), &Point3D::from(0.0, 0.0, 1.0), &ProjectionOptions::default()),
            Ok(Point2D {x: 0.0, y: 0.0})
        );
    }

    #[test]
    pub fn behind_camera_returns_clip_error () {
        assert_eq!(project(calculate_matrix(&Camera::default().rotation), &Camera::default(), &Point3D::from(0.0, 0.0, -1.0), &ProjectionOptions::default()),
                   Err(Error::PointTooClose));
    }

    #[test]
    pub fn too_far_away_returns_clip_error () {
        assert_eq!(project(calculate_matrix(&Camera::default().rotation), &Camera::default(), &Point3D::from(0.0, 0.0, 10000.0), &ProjectionOptions::default()),
                   Err(Error::PointTooFar));
    }

    #[test]
    pub fn invalid_fov_returns_fov_error () {
        let mut def = ProjectionOptions::default();
        def.fov = std::f32::consts::PI;
        def.far_clip = 100.0;
        assert_eq!(project(calculate_matrix(&Camera::default().rotation), &Camera::default(), &Point3D::from(0.0, 0.0, 0.5), &def),
        Err(Error::UnacceptableFOV));
    }

    #[test]
    pub fn yaw_controls_left_right () {
        let cam = Camera {
            rotation: Rotation3D {
                yaw: 3f32.to_radians(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(project(calculate_matrix(&cam.rotation), &cam, &Point3D::from(-3f32.to_radians().tan(), 0.0, 1.0), &ProjectionOptions::default()), Ok(Point2D::from(0.0, 0.0)))
    }

    #[test]
    pub fn pitch_controls_up_down () {
        let cam = Camera {
            rotation: Rotation3D {
                pitch: 3f32.to_radians(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(project(calculate_matrix(&cam.rotation), &cam, &Point3D::from(0.0, 3f32.to_radians().tan(), 1.0), &ProjectionOptions::default()), Ok(Point2D::from(0.0, 0.0)))
    }

    #[test]
    pub fn roll_controls_orientation () {
        let cam = Camera {
            rotation: Rotation3D {
                roll: -90f32.to_radians(),
                ..Rotation3D::default()
            },
            ..Camera::default()
        };

        let p = Point3D::from(0.0, 1.0, 1.0);

        let projected = project(
            calculate_matrix(&cam.rotation),
            &cam,
            &p,
            &ProjectionOptions::default()
        );

        let control = project(
            calculate_matrix(&Camera::default().rotation),
            &Camera::default(),
            &Point3D::from(1.0, 0.0, 1.0),
            &ProjectionOptions::default()
        );
        assert!(projected.is_ok());
        assert!(control.is_ok());
        let unwrapped_proj = projected.unwrap();
        let unwrapped_control = control.unwrap();
        assert_abs_diff_eq!(&unwrapped_proj.x, &unwrapped_control.x, epsilon = 2e-6);
        assert_abs_diff_eq!(&unwrapped_proj.y, &unwrapped_control.y, epsilon = 2e-6);
    }


    #[test]
    pub fn gpu_input_is_right_size () {
        assert_eq!(size_of::<GpuInput>(), 16)
    }

    #[test]
    pub fn gpu_input_is_correctly_aligned () {
        assert_eq!(align_of::<GpuInput>(), 4)
    }

    #[test]
    fn gpu_roundtrip_copies_payload() {
        let Some((device, queue)) = request_test_device() else {
            return;
        };
        // let payload = [
        //     GpuInput { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
        //     GpuInput { x: 5.5, y: -6.5, z: 7.25, w: -8.25, },
        // ];
        let payload = generate_test_points();
        let result: Vec<Point3D> = gpu_copy_roundtrip(&device, &queue, &generate_test_points_as_gpu_inputs())
            .expect("roundtrip")
            .into_iter()
            .map(|p| p.into())
            .collect();
        assert_eq!(result, payload.to_vec());
    }

    #[test]
    fn gpu_projection_matches_cpu() {
        let Some((device, queue)) = request_test_device() else {
            return;
        };
        let gpu_device = ProjectionDevice::Gpu(GpuOptions::new(device.clone(), queue.clone()));
        let cpu_device = ProjectionDevice::Cpu;
        let camera = Camera::default();
        let points = generate_test_points();
        let options = ProjectionOptions {
            fov: 1.5,
            ..ProjectionOptions::default()
        };
        let cpu = cpu_device.project(&camera, &points, &options);
        let gpu = gpu_device.project(&camera, &points, &options);

        assert_eq!(cpu.len(), gpu.len(), "Point count mismatch");

        for (c, g) in cpu.iter().zip(gpu.iter()) {
            match (c, g) {
                (Some(cpu_pt), Some(gpu_pt)) => {
                    assert_abs_diff_eq!(cpu_pt.x, gpu_pt.x, epsilon = 1e-3);
                    assert_abs_diff_eq!(cpu_pt.y, gpu_pt.y, epsilon = 1e-3);
                }
                (None, None) => {}
                _ => {
                    panic!("Mismatch in projection state: CPU is {:?}, GPU is {:?}", c, g);
                }
            }
        }
    }

    #[test]
    pub fn triangle_conversion_is_correct () {
        let points = generate_test_points();
        assert_eq!(points, Triangle3D::pack(&*Triangle3D::unpack(points.as_slice())));
    }
}
