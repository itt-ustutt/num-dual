use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "HyperHyperDual64")]
#[derive(Clone)]
/// Third order hyper dual number using 64-bit-floats as fields.
pub struct PyHyperHyperDual64(HyperHyperDual64);

#[pymethods]
#[expect(clippy::too_many_arguments)]
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
    fn get_first_derivative(&self) -> (f64, f64, f64) {
        (self.0.eps1, self.0.eps2, self.0.eps3)
    }

    #[getter]
    fn get_second_derivative(&self) -> (f64, f64, f64) {
        (self.0.eps1eps2, self.0.eps1eps3, self.0.eps2eps3)
    }

    #[getter]
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
#[expect(clippy::type_complexity)]
pub fn third_partial_derivative(
    f: &Bound<'_, PyAny>,
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

#[pyfunction]
/// Calculate the third partial derivative of a scalar function
/// with arbitrary many variables.
///
/// Parameters
/// ----------
/// f : callable
///     A scalar, bivariate function.
/// x : [float]
///     The list of variables.
/// i : integer
///     The index of the first partial derivative.
/// j : integer
///     The index of the second partial derivative.
/// k : integer
///     The index of the third partial derivative.
///
/// Returns
/// -------
/// function value
/// first partial derivative w.r.t. variable i
/// first parital derivative w.r.t. variable j
/// first parital derivative w.r.t. variable k
/// second partial derivative w.r.t. variables i and j
/// second partial derivative w.r.t. variables i and k
/// second partial derivative w.r.t. variables j and k
/// third partial derivative
#[expect(clippy::type_complexity)]
pub fn third_partial_derivative_vec(
    f: &Bound<'_, PyAny>,
    x: Vec<f64>,
    i: usize,
    j: usize,
    k: usize,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    let g = |x: &[_]| {
        let x: Vec<_> = x.iter().map(|&x| PyHyperHyperDual64::from(x)).collect();
        let res = f.call1((x,))?;
        if let Ok(res) = res.extract::<PyHyperHyperDual64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar.".to_string(),
            ))
        }
    };
    try_third_partial_derivative_vec(g, &x, i, j, k)
}
