use super::dual::PyDual64;
use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "Dual3_64")]
#[derive(Clone)]
/// Third order dual number using 64-bit-floats as fields.
pub struct PyDual3_64(Dual3_64);

#[pymethods]
impl PyDual3_64 {
    #[new]
    fn new(eps: f64, v1: f64, v2: f64, v3: f64) -> Self {
        Dual3::new(eps, v1, v2, v3).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> f64 {
        self.0.v1
    }

    #[getter]
    fn get_second_derivative(&self) -> f64 {
        self.0.v2
    }

    #[getter]
    fn get_third_derivative(&self) -> f64 {
        self.0.v3
    }
}

impl_dual_num!(PyDual3_64, Dual3_64, f64);

#[pyclass(name = "Dual3Dual64")]
#[derive(Clone)]
/// Third order dual number using dual numbers as fields.
pub struct PyDual3Dual64(Dual3<Dual64, f64>);

#[pymethods]
impl PyDual3Dual64 {
    #[new]
    pub fn new(v0: PyDual64, v1: PyDual64, v2: PyDual64, v3: PyDual64) -> Self {
        Dual3::new(v0.into(), v1.into(), v2.into(), v3.into()).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> PyDual64 {
        self.0.v1.into()
    }

    #[getter]
    fn get_second_derivative(&self) -> PyDual64 {
        self.0.v2.into()
    }

    #[getter]
    fn get_third_derivative(&self) -> PyDual64 {
        self.0.v3.into()
    }
}

impl_dual_num!(PyDual3Dual64, Dual3<Dual64, f64>, PyDual64);

#[pyfunction]
/// Calculate the third derivative of a scalar, univariate function.
///
/// Parameters
/// ----------
/// f : callable
///     A scalar, univariate function.
/// x : float
///     The value at which the derivative is evaluated.
///
/// Returns
/// -------
/// function value, first derivative, second derivative, and third derivative
pub fn third_derivative(f: &Bound<'_, PyAny>, x: f64) -> PyResult<(f64, f64, f64, f64)> {
    let g = |x| {
        let res = f.call1((PyDual3_64::from(x),))?;
        if let Ok(res) = res.extract::<PyDual3_64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar.".to_string(),
            ))
        }
    };
    crate::third_derivative(g, x)
}
