use super::dual::PyDual64;
use crate::*;
use nalgebra::SVector;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyclass(name = "HyperDual64")]
#[derive(Clone)]
/// Hyper-dual number using 64-bit-floats as fields.
///
/// A hyper-dual number consists of
/// a + b ε1 + c ε2 + d ε1ε2
///
/// Examples
///
/// >>> from num_dual import HyperDual64 as HD64
/// >>> x = HD64(1.0, 0.0, 0.0, 0.0) # constructor
/// >>> y = HD64.from_re(2.0)        # from real value
/// >>> x + y
/// 3 + [0]ε1 + [0]ε2 + [0]ε1ε2
///
/// Compute partial derivatives of a function.
///
/// >>> from num_dual import second_partial_derivative
/// >>> f, f_x, f_y, f_xy = second_partial_derivative(lambda x,y: x**2 * y - y**3, 3.0, 4.0)
/// >>> f
/// -28.0
/// >>> f_x
/// 24.0
/// >>> f_y
/// -39.0
/// >>> f_xy
/// 6.0
pub struct PyHyperDual64(HyperDual64);

#[pymethods]
impl PyHyperDual64 {
    #[new]
    pub fn new(re: f64, eps1: f64, eps2: f64, eps1eps2: f64) -> Self {
        HyperDual::new_scalar(re, eps1, eps2, eps1eps2).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> (f64, f64) {
        (self.0.eps1[0], self.0.eps2[0])
    }

    #[getter]
    fn get_second_derivative(&self) -> f64 {
        self.0.eps1eps2[(0, 0)]
    }
}

impl_dual_num!(PyHyperDual64, HyperDual64, f64);

#[pyclass(name = "HyperDualDual64")]
#[derive(Clone)]
/// Hyper-dual number using dual numbers as fields.
pub struct PyHyperDualDual64(HyperDual<Dual64, f64>);

#[pymethods]
impl PyHyperDualDual64 {
    #[new]
    pub fn new(re: PyDual64, eps1: PyDual64, eps2: PyDual64, eps1eps2: PyDual64) -> Self {
        HyperDual::new_scalar(re.into(), eps1.into(), eps2.into(), eps1eps2.into()).into()
    }

    #[getter]
    fn get_first_derivative(&self) -> (PyDual64, PyDual64) {
        (self.0.eps1[0].into(), self.0.eps2[0].into())
    }

    #[getter]
    fn get_second_derivative(&self) -> PyDual64 {
        self.0.eps1eps2[(0, 0)].into()
    }
}

impl_dual_num!(PyHyperDualDual64, HyperDual<Dual64, f64>, PyDual64);

macro_rules! impl_hyper_dual_mn {
    ($py_type_name:ident, $m:literal, $n:literal) => {
        #[pyclass(name = "HyperDualVec64")]
        #[derive(Clone, Copy)]
        pub struct $py_type_name(HyperDualVec64<$m, $n>);

        #[pymethods]
        impl $py_type_name {
            #[getter]
            fn get_first_derivative(&self) -> ([f64; $m], [f64; $n]) {
                (self.0.eps1.data.0[0], self.0.eps2.transpose().data.0[0])
            }

            #[getter]
            pub fn get_second_derivative(&self) -> [[f64; $m]; $n] {
                self.0.eps1eps2.data.0
            }
        }

        impl_dual_num!($py_type_name, HyperDualVec64<$m, $n>, f64);
    };
}

#[pyfunction]
/// Calculate the second partial derivatives of a scalar, bivariate function.
///
/// Parameters
/// ----------
/// f : callable
///     A scalar, bivariate function.
/// x : float
///     The value of the first variable.
/// y : float
///     The value of the second variable.
///
/// Returns
/// -------
/// function value, first partial derivative w.r.t. x,
/// second parital derivative w.r.t. y, and second partial derivative
pub fn second_partial_derivative(f: &PyAny, x: f64, y: f64) -> PyResult<(f64, f64, f64, f64)> {
    let g = |x, y| {
        let res = f.call1((PyHyperDual64::from(x), PyHyperDual64::from(y)))?;
        if let Ok(res) = res.extract::<PyHyperDual64>() {
            Ok(res.0)
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "argument 'f' must return a scalar.".to_string(),
            ))
        }
    };
    try_second_partial_derivative(g, x, y)
}

macro_rules! impl_partial_hessian {
    ([$(($py_type_name:ident, $m:literal, $n:literal)),+]) => {
        #[pyfunction]
        /// Calculate the Hessian of a scalar function w.r.t. a subset of its variables.
        ///
        /// Parameters
        /// ----------
        /// f : callable
        ///     A scalar, multivariate function.
        /// x : [float]
        ///     The first vector for which the partial Hessian is evaluated.
        /// y : [float]
        ///     The second vector for which the partial Hessian is evaluated.
        ///
        /// Returns
        /// -------
        /// function value, gradient w.r.t. x, gradient w.r.t. y, and partial Hessian
        #[allow(clippy::type_complexity)]
        pub fn partial_hessian(
            f: &PyAny,
            x: &PyAny,
            y: &PyAny,
        ) -> PyResult<(f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
            $(
                if let (Ok(x), Ok(y)) = (x.extract::<[f64; $m]>(), y.extract::<[f64; $n]>()) {
                    let g = |x: SVector<HyperDualVec64<$m, $n>, $m>, y: SVector<HyperDualVec64<$m, $n>, $n>| {
                        let x: Vec<_> = x.into_iter().map(|&x| $py_type_name::from(x)).collect();
                        let y: Vec<_> = y.into_iter().map(|&y| $py_type_name::from(y)).collect();
                        let res = f.call1((x, y))?;
                        if let Ok(res) = res.extract::<$py_type_name>() {
                            Ok(res.0)
                        } else {
                            Err(PyErr::new::<PyTypeError, _>(
                                "argument 'f' must return a scalar.".to_string(),
                            ))
                        }
                    };
                    try_partial_hessian(g, SVector::from(x), SVector::from(y)).map(|(f, f_x, f_y, f_xy)| {
                        let f_xy = f_xy
                            .row_iter()
                            .map(|r| r.iter().copied().collect())
                            .collect();
                        (f, f_x.data.0[0].to_vec(), f_y.data.0[0].to_vec(), f_xy)
                    })
                } else
            )+
            if x.extract::<Vec<f64>>().is_ok() {
                Err(PyErr::new::<PyTypeError, _>(
                    "partial Hessians are only available for up to 5 variables for x and y!".to_string(),
                ))
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                        "argument 'x' and 'y' must be lists. For bivariate functions use 'second_partial_derivative' instead.".to_string(),
                    ))
            }
        }
        $(impl_hyper_dual_mn!($py_type_name, $m, $n);)+
    };
}

impl_partial_hessian!([
    (PyHyperDual64_1_1, 1, 1),
    (PyHyperDual64_1_2, 1, 2),
    (PyHyperDual64_1_3, 1, 3),
    (PyHyperDual64_1_4, 1, 4),
    (PyHyperDual64_1_5, 1, 5),
    (PyHyperDual64_2_1, 2, 1),
    (PyHyperDual64_2_2, 2, 2),
    (PyHyperDual64_2_3, 2, 3),
    (PyHyperDual64_2_4, 2, 4),
    (PyHyperDual64_2_5, 2, 5),
    (PyHyperDual64_3_1, 3, 1),
    (PyHyperDual64_3_2, 3, 2),
    (PyHyperDual64_3_3, 3, 3),
    (PyHyperDual64_3_4, 3, 4),
    (PyHyperDual64_3_5, 3, 5),
    (PyHyperDual64_4_1, 4, 1),
    (PyHyperDual64_4_2, 4, 2),
    (PyHyperDual64_4_3, 4, 3),
    (PyHyperDual64_4_4, 4, 4),
    (PyHyperDual64_4_5, 4, 5),
    (PyHyperDual64_5_1, 5, 1),
    (PyHyperDual64_5_2, 5, 2),
    (PyHyperDual64_5_3, 5, 3),
    (PyHyperDual64_5_4, 5, 4),
    (PyHyperDual64_5_5, 5, 5)
]);
