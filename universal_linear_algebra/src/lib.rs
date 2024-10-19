pub mod gpu;

use gpu::GPU;
use pyo3::prelude::*;

#[pymodule]
fn universal_linear_algebra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_1d, m)?)?;
    Ok(())
}

#[pyfunction]
fn add_1d(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let gpu = pollster::block_on(GPU::new());
    pollster::block_on(gpu.add_vecs(&a, &b))
}