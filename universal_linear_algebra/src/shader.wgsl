@group(0)
@binding(0)
var<storage, read_write> v_indices1: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> v_indices2: array<f32>;

fn add_vec(n_base1: f32, n_base2: f32) -> f32 {
    return n_base1 + n_base2;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices1[global_id.x] = add_vec(v_indices1[global_id.x], v_indices2[global_id.x]);
}
