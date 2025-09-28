use super::dual::PyDual64;
use crate::*;
use nalgebra::{DVector, SVector};
use numpy::{PyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
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
/// >>> from num_dual import second_derivative
/// >>> import numpy as np
/// >>> f, df, d2f = second_derivative(lambda x: x * x + np.sqrt(x), 4.0)
/// >>> f
/// 18.0
/// >>> df
/// 8.25
/// >>> d2f
/// 1.96875
pub struct PyDual2_64(Dual2_64);

#[pymethods]
impl PyDual2_64 {
    #[new]
    fn new(eps: f64, v1: f64, v2: f64) -> Self {
        Dual2::new(eps, v1, v2).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> f64 {
        self.0.v1
    }

    #[getter]
    fn get_second_derivative(&self) -> f64 {
        self.0.v2
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
        Dual2::new(v0.into(), v1.into(), v2.into()).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> PyDual64 {
        self.0.v1.into()
    }

    #[getter]
    fn get_second_derivative(&self) -> PyDual64 {
        self.0.v2.into()
    }
}

impl_dual_num!(PyDual2Dual64, Dual2<Dual64, f64>, PyDual64);

macro_rules! impl_dual2_n {
    ($py_type_name:ident, $n:literal) => {
        #[pyclass(name = "Dual2Vec64")]
        #[derive(Clone, Copy)]
        pub struct $py_type_name(Dual2SVec64<$n>);

        #[pymethods]
        impl $py_type_name {
            #[getter]
            pub fn get_first_derivative(&self) -> Option<[f64; $n]> {
                self.0.v1.0.as_ref().map(|v1| v1.transpose().data.0[0])
            }

            #[getter]
            pub fn get_second_derivative(&self) -> Option<[[f64; $n]; $n]> {
                self.0.v2.0.as_ref().map(|v2| v2.data.0)
            }
        }

        impl_dual_num!($py_type_name, Dual2SVec64<$n>, f64);
    };
}

#[pyclass(name = "Dual2_64Dyn")]
#[derive(Clone)]
pub struct PyDual2_64Dyn(Dual2DVec64);

impl_dual_num!(PyDual2_64Dyn, Dual2DVec64, f64);

#[pyfunction]
/// Calculate the second derivative of a scalar, univariate function.
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
/// function value, first derivative, and second derivative
pub fn second_derivative(f: &Bound<'_, PyAny>, x: f64) -> PyResult<(f64, f64, f64)> {
    let g = |x| {
        let res = f.call1((PyDual2_64::from(x),))?;
        if let Ok(res) = res.extract::<PyDual2_64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar.".to_string(),
            ))
        }
    };
    crate::second_derivative(g, x)
}

macro_rules! impl_hessian {
    ([$(($py_type_name:ident, $n:literal)),+]) => {
        #[pyfunction]
        /// Calculate the Hessian of a scalar, multivariate function.
        ///
        /// Parameters
        /// ----------
        /// f : callable
        ///     A scalar, multivariate function.
        /// x : [float]
        ///     The vector for which the Hessian is evaluated.
        ///
        /// Returns
        /// -------
        /// function value, gradient and Hessian
        pub fn hessian(f: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>, Vec<Vec<f64>>)> {
            $(
                if let Ok(x) = x.extract::<[f64; $n]>() {
                    let g = |x: SVector<Dual2SVec64<$n>, $n>| {
                        let x: Vec<_> = x.into_iter().map(|&x| $py_type_name::from(x)).collect();
                        let res = f.call1((x,))?;
                        if let Ok(res) = res.extract::<$py_type_name>() {
                            Ok(res.0)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(
                                "argument 'f' must return a scalar."
                                    .to_string(),
                            ))
                        }
                    };
                    crate::hessian(g, &SVector::from(x)).map(|(f, g, h)| {
                        let h = h.row_iter().map(|r| r.iter().copied().collect()).collect();
                        (f, g.data.0[0].to_vec(), h)
                    })
                } else
            )+
            if let Ok(x) = x.extract::<Vec<f64>>() {
                let g = |x: DVector<Dual2DVec64>| {
                    let x: Vec<_> = x.into_iter().map(|x| PyDual2_64Dyn::from(x.clone())).collect();
                    let res = f.call1((x,))?;
                    if let Ok(res) = res.extract::<PyDual2_64Dyn>() {
                        Ok(res.0)
                    } else {
                        Err(PyErr::new::<PyTypeError, _>(
                            "argument 'f' must return a scalar."
                                .to_string(),
                        ))
                    }
                };
                crate::hessian(g, &DVector::from(x)).map(|(f, g, h)| {
                    let h = h.row_iter().map(|r| r.iter().copied().collect()).collect();
                    (f, g.data.as_vec().clone(), h)
                })
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                        "argument 'x': must be a list. For univariate functions use 'second_derivative' instead.".to_string(),
                    ))
            }
        }

        $(impl_dual2_n!($py_type_name, $n);)+
    };
}

impl_hessian!([
    (PyDual2_64_1, 1),
    (PyDual2_64_2, 2),
    (PyDual2_64_3, 3),
    (PyDual2_64_4, 4),
    (PyDual2_64_5, 5),
    (PyDual2_64_6, 6),
    (PyDual2_64_7, 7),
    (PyDual2_64_8, 8),
    (PyDual2_64_9, 9),
    (PyDual2_64_10, 10)
]);
