use super::dual::PyDual64;
use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "Dual2_64")]
#[derive(Clone)]
/// Second order dual number using 64-bit-floats as fields.
///
/// A second order dual number consists of
/// f_0 + f_1 ε_1 + f_2 ε_1^2
/// where f_0 is the function value, f_1 the first derivative
/// and f_2 the second derivative.
///
/// Examples
///
/// >>> from num_dual import Dual2_64 as D264
/// >>> x = D264(1.0, 0.0, 0.0)
/// >>> y = D264.from_re(2.0)
/// >>> x + y
/// 3 + [0]ε1 + [0]ε1²
///
/// First and second derivative of a function.
///
/// >>> from num_dual import Dual2_64 as D264, derive1
/// >>> import numpy as np
/// >>> x = derive2(4.0)
/// >>> # this is equivalent to the above
/// >>> x = D264(4.0, 1.0, 0.0)
/// >>> fx = x*x + np.sqrt(x)
/// >>> fx.value
/// 18.0
/// >>> fx.first_derivative
/// 8.25
/// >>> fx.second_derivative
/// 1.96875
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
/// Second order dual number using dual numbers as fields.
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

macro_rules! impl_dual2_n {
    ($py_type_name:ident, $n:literal) => {
        #[pyclass(name = "Dual2Vec64")]
        #[derive(Clone, Copy)]
        pub struct $py_type_name(Dual2Vec64<$n>);

        #[pymethods]
        impl $py_type_name {
            #[getter]
            /// Gradient.
            pub fn get_first_derivative(&self) -> [f64; $n] {
                self.0.v1.transpose().data.0[0]
            }

            #[getter]
            /// Hessian.
            pub fn get_second_derivative(&self) -> [[f64; $n]; $n] {
                self.0.v2.data.0
            }
        }

        impl_dual_num!($py_type_name, Dual2Vec64<$n>, f64);
    };
}

pub(crate) use impl_dual2_n;
