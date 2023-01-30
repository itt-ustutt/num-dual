use crate::*;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::Python;

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
        Self(Dual64::new_scalar(re, eps))
    }

    #[getter]
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

        #[pymethods]
        impl $py_type_name {
            #[getter]
            pub fn get_first_derivative(&self) -> [f64; $n] {
                self.0.eps.data.0[0]
            }
        }

        impl_dual_num!($py_type_name, DualVec64<$n>, f64);
    };
}

#[pyfunction]
pub fn first_derivative(f: &PyAny, x: f64) -> PyResult<(f64, f64)> {
    let x = PyDual64::from(Dual::new_scalar(x, 1.0));
    let res = f.call1((x,))?;
    if let Ok(res) = res.extract::<PyDual64>() {
        Ok((res.0.re, res.0.eps[0]))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                .to_string(),
        ))
    }
}

macro_rules! impl_gradient_and_jacobian {
    ([$(($py_type_name:ident, $n:literal)),+]) => {
        #[pyfunction]
        pub fn gradient(f: &PyAny, x: &PyAny) -> PyResult<(f64, Vec<f64>)> {
            let res = if let Ok([x]) = x.extract::<[f64; 1]>() {
                let x = vec![PyDual64::from(Dual::new_scalar(x, 1.0))];
                f.call1((x,))?
            } else
            $(
                if let Ok(x) = x.extract::<[f64; $n]>() {
                    let mut x: Vec<_> = x.iter().copied().map(DualVec::from_re).collect();
                    x.iter_mut().enumerate().for_each(|(i, x)| x.eps[i] = 1.0);
                    let x: Vec<_> = x.into_iter().map($py_type_name::from).collect();
                    f.call1((x,))?
                } else
            )+
            if let Ok(_) = x.extract::<Vec<f64>>() {
                return Err(PyErr::new::<PyTypeError, _>(format!("Gradients are only available for up to 10 variables!")))
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                        "argument 'x': must be a list. For monovariate functions use 'first_derivative' instead.".to_string(),
                    ));
            };
            if let Ok(res) = res.extract::<PyDual64>() {
                let eps = res.0.eps.data.0[0].to_vec();
                Ok((res.0.re, eps))
            } else
            $(
                if let Ok(res) = res.extract::<$py_type_name>() {
                    let eps = res.0.eps.data.0[0].to_vec();
                    Ok((res.0.re, eps))
                } else
            )+
            {
                Err(PyErr::new::<PyTypeError, _>(
                    "argument 'f' must return a scalar. For vector functions use 'jacobian' instead."
                        .to_string(),
                ))
            }
        }

        #[pyfunction]
        pub fn jacobian(f: &PyAny, x: &PyAny) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
            let res = if let Ok([x]) = x.extract::<[f64; 1]>() {
                let x = vec![PyDual64::from(Dual::new_scalar(x, 1.0))];
                f.call1((x,))?
            } else
            $(
                if let Ok(x) = x.extract::<[f64; $n]>() {
                    let mut x: Vec<_> = x.iter().copied().map(DualVec::from_re).collect();
                    x.iter_mut().enumerate().for_each(|(i, x)| x.eps[i] = 1.0);
                    let x: Vec<_> = x.into_iter().map($py_type_name::from).collect();
                    f.call1((x,))?
                } else
            )+
            if let Ok(_) = x.extract::<Vec<f64>>() {
                return Err(PyErr::new::<PyTypeError, _>(format!("Jacobains are only available for up to 10 variables!")))
            } else {
                return Err(PyErr::new::<PyTypeError, _>("argument 'x': must be a list.".to_string()));
            };
            if let Ok(res) = res.extract::<Vec<PyDual64>>() {
                let re: Vec<_> = res.iter().map(|r| r.0.re).collect();
                let eps: Vec<_> = res.iter().map(|r| vec![r.0.eps[0]]).collect();
                Ok((re, eps))
            } else
            $(
                if let Ok(res) = res.extract::<Vec<$py_type_name>>() {
                    let re: Vec<_> = res.iter().map(|r| r.0.re).collect();
                    let eps: Vec<_> = res.iter().map(|r| r.0.eps.data.0[0].to_vec()).collect();
                    Ok((re, eps))
                } else
            )+
            {
                Err(PyErr::new::<PyTypeError, _>(
                    "argument 'f' must return a list. For scalar functions use 'first_derivative' or 'gradient' instead."
                        .to_string(),
                ))
            }
        }

        $(impl_dual_n!($py_type_name, $n);)+
    };
}

impl_gradient_and_jacobian!([
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
