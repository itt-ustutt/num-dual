use crate::*;
use pyo3::exceptions::PyTypeError;
use pyo3::number::PyNumberProtocol;
use pyo3::prelude::*;

#[pyclass(name = "Dual64")]
#[derive(Clone)]
/// Dual number using 64-bit-floats as fields.
///
/// A dual number consists of
/// a + b ε
///
/// Examples
///
/// >>> from dualnum import Dual64 as D64
/// >>> x = D64(1.0, 0.0)
/// >>> y = D64.from_re(2.0)
/// >>> x + y
/// 3 + [0]ε
///
/// First derivative of a function.
///
/// >>> from dualnum import Dual64 as D64, derive1
/// >>> import numpy as np
/// >>> x = derive1(4.0)
/// >>> # this is equivalent to the above
/// >>> x = D64(4.0, 1.0)
/// >>> fx = x*x + np.sqrt(x)
/// >>> fx.value
/// 18
/// >>> fx.first_derivative
/// 8.25
pub struct PyDual64(Dual64);

#[pymethods]
impl PyDual64 {
    #[new]
    pub fn new(re: f64, eps: f64) -> Self {
        Self(Dual64::new_scalar(re, eps))
    }

    #[getter]
    /// Dual part.
    pub fn get_first_derivative(&self) -> f64 {
        self.0.eps[0]
    }
}

impl_dual_num!(PyDual64, Dual64, f64);

macro_rules! impl_dual_n {
    ($py_type_name:ident, $n:literal) => {
        #[pyclass(name = "DualVec64")]
        #[derive(Clone, Copy)]
        pub struct $py_type_name(DualVec64<$n>);

        impl $py_type_name {
            pub fn new(re: f64, eps: [f64; $n]) -> Self {
                DualVec64::new(re, StaticVec::new_vec(eps)).into()
            }
        }

        #[pymethods]
        impl $py_type_name {
            #[getter]
            /// Dual part.
            pub fn get_first_derivative(&self) -> [f64; $n] {
                *self.0.eps.raw_array()
            }
        }

        impl_dual_num!($py_type_name, DualVec64<$n>, f64);
    };
}

macro_rules! impl_derive {
    ([$(($py_type_name:ident, $n:literal)),+]) => {
        #[pyfunction]
        #[pyo3(text_signature = "(x)")]
        pub fn derive1(x: &PyAny) -> PyResult<PyObject> {
            Python::with_gil(|py| {
                if let Ok(x) = x.extract::<f64>() {
                    return Ok(PyCell::new(py, PyDual64::from(Dual64::from_re(x).derive()))?.to_object(py));
                };
                $(
                    if let Ok(x) = x.extract::<[f64; $n]>() {
                        let arr = StaticVec::new_vec(x).map(DualVec64::from).derive();
                        let py_vec: Result<Vec<&PyCell<$py_type_name>>, _> = arr.raw_array().iter().map(|&i| PyCell::new(py, $py_type_name::from(i))).collect();
                        return Ok(py_vec?.to_object(py));
                    };
                )+
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            })
        }
        $(impl_dual_n!($py_type_name, $n);)+
    };
}

impl_derive!([
    (PyDual64_2, 2),
    (PyDual64_3, 3),
    (PyDual64_4, 4),
    (PyDual64_5, 5),
    (PyDual64_6, 6),
    (PyDual64_7, 7),
    (PyDual64_8, 8),
    (PyDual64_9, 9),
    (PyDual64_10, 10)
]);
