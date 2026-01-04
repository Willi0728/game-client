struct InputPoint {
    data: vec4<f32>,
};

struct OutputPoint {
    xy: vec2<f32>,
    valid: u32,
    _pad: u32,
};

struct ProjectUniform {
    rotation: mat4x4<f32>,
    camera_position: vec4<f32>,
    clip: vec4<f32>,
    scale: vec4<f32>,
    metadata: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read> input_points: array<InputPoint>;

@group(0) @binding(1)
var<storage, read_write> output_points: array<OutputPoint>;

@group(0) @binding(2)
var<uniform> uniforms: ProjectUniform;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= uniforms.metadata.x) {
        return;
    }

    let local = input_points[index].data.xyz - uniforms.camera_position.xyz;
    let rotated = uniforms.rotation * vec4<f32>(local, 1.0);
    let depth = rotated.z;
    if (depth < uniforms.clip.x || depth > uniforms.clip.y) {
        output_points[index].xy = vec2<f32>(0.0, 0.0);
        output_points[index].valid = 0u;
        return;
    }

    let px = uniforms.scale.x * (rotated.x / depth);
    let py = uniforms.scale.y * (rotated.y / depth);
    output_points[index].xy = vec2<f32>(px, py);
    output_points[index].valid = 1u;
}