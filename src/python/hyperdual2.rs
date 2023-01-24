use super::dual::PyHyperDual2_64;
use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "HyperDual2_64")]
#[derive(Clone)]
/// Third order hyper dual number using 64-bit-floats as fields.
pub struct PyHyperDual2_64(HyperDual2_64);

#[pymethods]
impl PyHyperDual2_64 {
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
        HyperDual2::new(
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

// impl_dual_num!(PyHyperDual2_64, HyperDual2_64, f64);

// #[pyclass(name = "Dual3Dual64")]
// #[derive(Clone)]
// /// Third order dual number using dual numbers as fields.
// pub struct PyDual3Dual64(Dual3<Dual64, f64>);

// #[pymethods]
// impl PyDual3Dual64 {
//     #[new]
//     pub fn new(v0: PyDual64, v1: PyDual64, v2: PyDual64, v3: PyDual64) -> Self {
//         Dual3::new(v0.into(), v1.into(), v2.into(), v3.into()).into()
//     }

//     #[getter]
//     /// First hyperdual part.
//     fn get_first_derivative(&self) -> PyDual64 {
//         self.0.v1.into()
//     }

//     #[getter]
//     /// Second hyperdual part.
//     fn get_second_derivative(&self) -> PyDual64 {
//         self.0.v2.into()
//     }

//     #[getter]
//     /// Third hyperdual part.
//     fn get_third_derivative(&self) -> PyDual64 {
//         self.0.v3.into()
//     }
// }

// impl_dual_num!(PyDual3Dual64, Dual3<Dual64, f64>, PyDual64);

// #[pyfunction]
// pub fn derive3(x: &PyAny) -> PyResult<PyObject> {
//     Python::with_gil(|py| {
//         if let Ok(x) = x.extract::<f64>() {
//             return Ok(
//                 PyCell::new(py, PyDual3_64::from(Dual3_64::from_re(x).derive()))?.to_object(py),
//             );
//         };
//         if let Ok(x) = x.extract::<PyDual64>() {
//             return Ok(
//                 PyCell::new(py, PyDual3Dual64::from(Dual3::from_re(x.into()).derive()))?
//                     .to_object(py),
//             );
//         };
//         Err(PyErr::new::<PyTypeError, _>("not implemented!".to_string()))
//     })
// }
