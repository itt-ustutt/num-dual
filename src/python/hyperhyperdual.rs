use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "HyperHyperDual64")]
#[derive(Clone)]
/// Third order hyper dual number using 64-bit-floats as fields.
pub struct PyHyperHyperDual64(HyperHyperDual64);

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyHyperHyperDual64 {
    #[new]
    fn new(
        re: f64,
        eps1: f64,
        eps2: f64,
        eps3: f64,
        eps1eps2: f64,
        eps1eps3: f64,
        eps2eps3: f64,
        eps1eps2eps3: f64,
    ) -> Self {
        HyperHyperDual::new(
            re,
            eps1,
            eps2,
            eps3,
            eps1eps2,
            eps1eps3,
            eps2eps3,
            eps1eps2eps3,
        )
        .into()
    }

    #[getter]
    /// First hyperdual part.
    fn get_first_derivative(&self) -> (f64, f64, f64) {
        (self.0.eps1, self.0.eps2, self.0.eps3)
    }

    #[getter]
    /// Second hyperdual part.
    fn get_second_derivative(&self) -> (f64, f64, f64) {
        (self.0.eps1eps2, self.0.eps1eps3, self.0.eps2eps3)
    }

    #[getter]
    /// Third hyperdual part.
    fn get_third_derivative(&self) -> f64 {
        self.0.eps1eps2eps3
    }
}

impl_dual_num!(PyHyperHyperDual64, HyperHyperDual64, f64);

#[pyfunction]
/// Calculate the third partial derivative of a scalar, trivariate function.
///
/// Parameters
/// ----------
/// f : callable
///     A scalar, bivariate function.
/// x : float
///     The value of the first variable.
/// y : float
///     The value of the second variable.
/// z : float
///     The value of the third variable.
///
/// Returns
/// -------
/// function value
/// first partial derivative w.r.t. x
/// first parital derivative w.r.t. y
/// first parital derivative w.r.t. z
/// second partial derivative w.r.t. x and y
/// second partial derivative w.r.t. x and z
/// second partial derivative w.r.t. y and z
/// third partial derivative
#[allow(clippy::type_complexity)]
pub fn third_partial_derivative(
    f: &PyAny,
    x: f64,
    y: f64,
    z: f64,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    let g = |x, y, z| {
        let res = f.call1((
            PyHyperHyperDual64::from(x),
            PyHyperHyperDual64::from(y),
            PyHyperHyperDual64::from(z),
        ))?;
        if let Ok(res) = res.extract::<PyHyperHyperDual64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar.".to_string(),
            ))
        }
    };
    try_third_partial_derivative(g, x, y, z)
}
