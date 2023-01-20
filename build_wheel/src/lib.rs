use ::num_dual as num_dual_rs;
use pyo3::prelude::*;

/// Implementation of dual numbers.
#[pymodule]
pub fn num_dual(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    num_dual_rs::python::num_dual(_py, m)
}
