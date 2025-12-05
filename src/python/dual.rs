use crate::*;
use nalgebra::{DVector, SVector};
use numpy::{PyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "Dual64")]
#[derive(Clone, Debug)]
/// Dual number using 64-bit-floats as fields.
///
/// A dual number consists of
/// f_0 + f_1 ε
/// where f_0 is the function value and f_1 its first derivative.
///
/// Examples
///
/// >>> from num_dual import Dual64 as D64
/// >>> x = D64(1.0, 0.0)
/// >>> y = D64.from_re(2.0)
/// >>> x + y
/// 3 + [0]ε
///
/// First derivative of a function.
///
/// >>> from num_dual import first_derivative
/// >>> import numpy as np
/// >>> f, df = first_derivative(lambda x: x * x + np.sqrt(x), 4.0)
/// >>> f
/// 18.0
/// >>> df
/// 8.25
pub struct PyDual64(Dual64);

#[pymethods]
impl PyDual64 {
    #[new]
    pub fn new(re: f64, eps: f64) -> Self {
        Self(Dual64::new(re, eps))
    }

    #[getter]
    pub fn get_first_derivative(&self) -> f64 {
        self.0.eps
    }
}

impl_dual_num!(PyDual64, Dual64, f64);

macro_rules! impl_dual_n {
    ($py_type_name:ident, $n:literal) => {
        #[pyclass(name = "DualSVec64")]
        #[derive(Clone, Copy)]
        pub struct $py_type_name(DualSVec64<$n>);

        #[pymethods]
        impl $py_type_name {
            #[getter]
            pub fn get_first_derivative(&self) -> Option<[f64; $n]> {
                self.0.eps.0.map(|eps| eps.data.0[0])
            }
        }

        impl_dual_num!($py_type_name, DualSVec64<$n>, f64);
    };
}

#[pyclass(name = "Dual64Dyn")]
#[derive(Clone)]
pub struct PyDual64Dyn(DualDVec64);

#[pymethods]
impl PyDual64Dyn {
    #[getter]
    pub fn get_first_derivative(&self) -> Option<Vec<f64>> {
        self.0.eps.0.as_ref().map(|eps| eps.data.as_vec().clone())
    }
}

impl_dual_num!(PyDual64Dyn, DualDVec64, f64);

#[pyfunction]
/// Calculate the first derivative of a scalar, univariate function.
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
/// function value and first derivative
pub fn first_derivative(f: &Bound<'_, PyAny>, x: f64) -> PyResult<(f64, f64)> {
    let g = |x| {
        let res = f.call1((PyDual64::from(x),))?;
        if let Ok(res) = res.extract::<PyDual64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                    .to_string(),
            ))
        }
    };
    crate::first_derivative(g, x)
}

macro_rules! impl_gradient_and_jacobian {
    ([$(($py_type_name:ident, $n:literal)),+]) => {
        #[pyfunction]
        /// Calculate the gradient of a scalar, multivariate function.
        ///
        /// Parameters
        /// ----------
        /// f : callable
        ///     A scalar, multivariate function.
        /// x : [float]
        ///     The vector for which the gradient is evaluated.
        ///
        /// Returns
        /// -------
        /// function value and gradient
        pub fn gradient(f: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>)> {
            $(
                if let Ok(x) = x.extract::<[f64; $n]>() {
                    let g = |x: SVector<DualSVec64<$n>, $n>| {
                        let x: Vec<_> = x.into_iter().map(|&x| $py_type_name::from(x)).collect();
                        let res = f.call1((x,))?;
                        if let Ok(res) = res.extract::<$py_type_name>() {
                            Ok(res.0)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(
                                "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                                    .to_string(),
                            ))
                        }
                    };
                    crate::gradient(g, &SVector::from(x)).map(|(re, eps)| (re, eps.data.0[0].to_vec()))
                } else
            )+
            if let Ok(x) = x.extract::<Vec<f64>>() {
                let g = |x: DVector<DualDVec64>| {
                    let x: Vec<_> = x.into_iter().map(|x| PyDual64Dyn::from(x.clone())).collect();
                    let res = f.call1((x,))?;
                    if let Ok(res) = res.extract::<PyDual64Dyn>() {
                        Ok(res.0)
                    } else {
                        Err(PyErr::new::<PyTypeError, _>(
                            "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                                .to_string(),
                        ))
                    }
                };
                crate::gradient(g, &DVector::from(x)).map(|(re, eps)| (re, eps.data.as_vec().clone()))
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                        "argument 'x': must be a list. For univariate functions use 'first_derivative' instead.".to_string(),
                    ))
            }
        }

        #[pyfunction]
        /// Calculate the Jacobian of a vector, multivariate function.
        ///
        /// Parameters
        /// ----------
        /// f : callable
        ///     A vector, multivariate function.
        /// x : [float]
        ///     The vector for which the Jacobian is evaluated.
        ///
        /// Returns
        /// -------
        /// function values and Jacobian
        pub fn jacobian(f: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
            $(
                if let Ok(x) = x.extract::<[f64; $n]>() {
                    let g = |x: SVector<DualSVec64<$n>, $n>| {
                        let x: Vec<_> = x.into_iter().map(|&x| $py_type_name::from(x)).collect();
                        let res = f.call1((x,))?;
                        if let Ok(res) = res.extract::<Vec<$py_type_name>>() {
                            let res = DVector::from_iterator(res.len(), res.into_iter().map(|r| r.0));
                            Ok(res)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(
                                "argument 'f' must return a list. For scalar functions use 'first_derivative' or 'gradient' instead."
                                    .to_string(),
                            ))
                        }
                    };
                    crate::jacobian(g, &SVector::from(x)).map(|(re, eps)| {
                        let eps: Vec<_> = eps
                            .row_iter()
                            .map(|r| r.iter().copied().collect::<Vec<_>>())
                            .collect();
                        (re.iter().copied().collect(), eps)
                    })
                } else
            )+
            if let Ok(x) = x.extract::<Vec<f64>>() {
                let g = |x: DVector<DualDVec64>| {
                    let x: Vec<_> = x.into_iter().map(|x| PyDual64Dyn::from(x.clone())).collect();
                    let res = f.call1((x,))?;
                    if let Ok(res) = res.extract::<Vec<PyDual64Dyn>>() {
                        let res = DVector::from_iterator(res.len(), res.into_iter().map(|r| r.0));
                        Ok(res)
                    } else {
                        Err(PyErr::new::<PyTypeError, _>(
                            "argument 'f' must return a list. For scalar functions use 'first_derivative' or 'gradient' instead."
                                .to_string(),
                        ))
                    }
                };
                crate::jacobian(g, &DVector::from(x)).map(|(re, eps)| {
                    let eps: Vec<_> = eps
                    .row_iter()
                    .map(|r| r.iter().copied().collect::<Vec<_>>())
                    .collect();
                    (re.iter().copied().collect(), eps)
                })
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                        "argument 'x': must be a list. For univariate functions use 'first_derivative' instead.".to_string(),
                    ))
            }
        }

        $(impl_dual_n!($py_type_name, $n);)+
    };
}

impl_gradient_and_jacobian!([
    (PyDual64_1, 1),
    (PyDual64_2, 2),
    (PyDual64_3, 3),
    (PyDual64_4, 4),
    (PyDual64_5, 5),
    (PyDual64_6, 6),
    (PyDual64_7, 7),
    (PyDual64_8, 8),
    (PyDual64_9, 9),
    (PyDual64_10, 10),
    (PyDual64_11, 11),
    (PyDual64_12, 12),
    (PyDual64_13, 13),
    (PyDual64_14, 14),
    (PyDual64_15, 15),
    (PyDual64_16, 16)
]);
