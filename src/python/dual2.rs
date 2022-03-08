use super::dual::PyDual64;
use crate::*;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "Dual2_64")]
#[derive(Clone)]
/// Hyper dual number using 64-bit-floats.
pub struct PyDual2_64(Dual2_64);

#[pymethods]
impl PyDual2_64 {
    #[new]
    fn new(eps: f64, v1: f64, v2: f64) -> Self {
        Dual2::new_scalar(eps, v1, v2).into()
    }

    #[getter]
    /// First hyperdual part.
    fn get_first_derivative(&self) -> f64 {
        self.0.v1[0]
    }

    #[getter]
    /// Second hyperdual part.
    fn get_second_derivative(&self) -> f64 {
        self.0.v2[0]
    }
}

impl_dual_num!(PyDual2_64, Dual2_64, f64);

#[pyclass(name = "Dual2Dual64")]
#[derive(Clone)]
/// Hyper dual number using 64-bit-floats.
pub struct PyDual2Dual64(Dual2<Dual64, f64>);

#[pymethods]
impl PyDual2Dual64 {
    #[new]
    pub fn new(v0: PyDual64, v1: PyDual64, v2: PyDual64) -> Self {
        Dual2::new_scalar(v0.into(), v1.into(), v2.into()).into()
    }

    #[getter]
    /// First hyperdual part.
    fn get_first_derivative(&self) -> PyDual64 {
        self.0.v1[0].into()
    }

    #[getter]
    /// Second hyperdual part.
    fn get_second_derivative(&self) -> PyDual64 {
        self.0.v2[(0, 0)].into()
    }
}

impl_dual_num!(PyDual2Dual64, Dual2<Dual64, f64>, PyDual64);
