#[macro_export]
macro_rules! impl_dual_num {
    ($py_type_name:ty, $data_type:ty, $field_type:ty) => {
        impl From<$data_type> for $py_type_name {
            fn from(d: $data_type) -> Self {
                Self(d)
            }
        }

        impl From<$py_type_name> for $data_type {
            fn from(d: $py_type_name) -> Self {
                d.0
            }
        }

        #[pymethods]
        impl $py_type_name {
            #[staticmethod]
            /// (Hyper) dual number from real part, setting all other parts to zero.
            pub fn from_re(re: $field_type) -> Self {
                <$data_type>::from_re(re.into()).into()
            }

            #[getter]
            /// Real part.
            fn get_value(&self) -> $field_type {
                self.0.re.into()
            }

            #[inline]
            /// Reciprocal value of self.
            pub fn recip(&self) -> Self {
                self.0.recip().into()
            }

            #[inline]
            /// Power using 32-bit integer as exponent.
            pub fn powi(&self, n: i32) -> Self {
                self.0.powi(n).into()
            }

            #[inline]
            /// Power using 64-bin float as exponent.
            pub fn powf(&self, n: f64) -> Self {
                self.0.powf(n).into()
            }

            #[inline]
            /// Power using self (hyper) dual number as exponent.
            pub fn powd(&self, n: Self) -> Self {
                self.0.powd(n.0).into()
            }

            #[inline]
            /// Sqaure root.
            pub fn sqrt(&self) -> Self {
                self.0.sqrt().into()
            }

            #[inline]
            /// Cubic root.
            pub fn cbrt(&self) -> Self {
                self.0.cbrt().into()
            }

            #[inline]
            /// Calculate the exponential of (hyper) dual number.
            pub fn exp(&self) -> Self {
                self.0.exp().into()
            }

            #[inline]
            /// Calculate 2**x of (hyper) dual number x.
            pub fn exp2(&self) -> Self {
                self.0.exp2().into()
            }

            #[inline]
            /// Calculate exp(x) - 1.
            pub fn expm1(&self) -> Self {
                self.0.exp_m1().into()
            }

            #[inline]
            /// Calculate natural logarithm.
            pub fn log(&self) -> Self {
                self.0.ln().into()
            }

            #[inline]
            /// Calculate logarithm with given base.
            pub fn log_base(&self, base: f64) -> Self {
                self.0.log(base).into()
            }

            #[inline]
            /// Calculate logarithm with base 2.
            pub fn log2(&self) -> Self {
                self.0.log2().into()
            }

            #[inline]
            /// Calculate logarithm with base 10.
            pub fn log10(&self) -> Self {
                self.0.log10().into()
            }

            #[inline]
            /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
            pub fn log1p(&self) -> Self {
                self.0.ln_1p().into()
            }

            #[inline]
            /// Hyperbolic sine function.
            pub fn sin(&self) -> Self {
                self.0.sin().into()
            }

            #[inline]
            /// Hyperbolic cosine function.
            pub fn cos(&self) -> Self {
                self.0.cos().into()
            }

            #[inline]
            /// Computes the tangent of a (hyper) dual number (in radians).
            pub fn tan(&self) -> Self {
                self.0.tan().into()
            }

            #[inline]
            /// Simultaneously computes the sine and cosine of the (hyper) dual number, x.
            pub fn sin_cos(&self) -> (Self, Self) {
                let (a, b) = self.0.sin_cos();
                (a.into(), b.into())
            }

            #[inline]
            /// Computes the arcsine of a (hyper) dual number.
            pub fn arcsin(&self) -> Self {
                self.0.asin().into()
            }

            #[inline]
            /// Computes the arccosine of a (hyper) dual number.
            pub fn arccos(&self) -> Self {
                self.0.acos().into()
            }

            #[inline]
            /// Computes the arctangent of a (hyper) dual number.
            pub fn arctan(&self) -> Self {
                self.0.atan().into()
            }

            #[inline]
            /// Computes the hyperbolic sine of a (hyper) dual number.
            pub fn sinh(&self) -> Self {
                self.0.sinh().into()
            }

            #[inline]
            /// Computes the hyperbolic cosine of a (hyper) dual number.
            pub fn cosh(&self) -> Self {
                self.0.cosh().into()
            }

            #[inline]
            /// Computes the hyperbolic tangent of a (hyper) dual number.
            pub fn tanh(&self) -> Self {
                self.0.tanh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic sine of a (hyper) dual number.
            pub fn arcsinh(&self) -> Self {
                self.0.asinh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic cosine of a (hyper) dual number.
            pub fn arccosh(&self) -> Self {
                self.0.acosh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic tangent of a (hyper) dual number.
            pub fn arctanh(&self) -> Self {
                self.0.atanh().into()
            }

            #[inline]
            /// Computes the first spherical bessel function.
            pub fn sph_j0(&self) -> Self {
                self.0.sph_j0().into()
            }
            #[inline]
            /// Computes the second spherical bessel function.
            pub fn sph_j1(&self) -> Self {
                self.0.sph_j1().into()
            }

            #[inline]
            /// Computes the third spherical bessel function.
            pub fn sph_j2(&self) -> Self {
                self.0.sph_j2().into()
            }

            // #[inline]
            // pub fn is_derivative_zero(&self) -> bool {
            //     self.0.is_derivative_zero()
            // }

            // #[inline]
            // /// Fused multiply-add. Computes (self * a) + b with only one rounding error.
            // fn mul_add(&self, a: Self, b: Self) -> Self {
            //     self.0.mul_add(a.0, b.0).into()
            // }

            fn __add__(&self, rhs: &PyAny) -> PyResult<PyObject> {
                Python::with_gil(|py| {
                    if let Ok(r) = rhs.extract::<f64>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() + r))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<Self>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() + r.0))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<f64>>() {
                        return Ok(PyArray::from_owned_object_array(
                            py,
                            r.as_array()
                                .mapv(|ri| Py::new(py, Self(self.0.clone() + ri)).unwrap()),
                        )
                        .into());
                    }
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<PyObject>>() {
                        // check data type of first element
                        if r.readonly()
                            .as_array()
                            .get(0)
                            .unwrap()
                            .as_ref(py)
                            .is_instance_of::<Self>()
                            .unwrap()
                        {
                            return Ok(PyArray::from_owned_object_array(
                                py,
                                r.as_array().mapv(|ri| {
                                    Py::new(py, Self(self.0.clone() + ri.extract::<Self>(py).unwrap().0))
                                        .unwrap()
                                }),
                            )
                            .into());
                        } else {
                            return Err(PyErr::new::<PyTypeError, _>(format!(
                                "Operation with the provided object type is not implemented. Supported data types are 'float', 'int' and '{}'.",
                                stringify!($py_type_name)
                            )));
                        }
                    }

                    Err(PyErr::new::<PyTypeError, _>(format!(
                        "Addition of \nleft:  {}\nright: {:?}\nis not implemented!",
                        stringify!($py_type_name),
                        rhs.get_type()
                    )))
                })
            }

            fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self.0.clone() + o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __sub__(&self, rhs: &PyAny) -> PyResult<PyObject> {
                Python::with_gil(|py| {
                    if let Ok(r) = rhs.extract::<f64>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() - r))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<Self>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() - r.0))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<f64>>() {
                        return Ok(PyArray::from_owned_object_array(
                            py,
                            r.as_array()
                                .mapv(|ri| Py::new(py, Self(self.0.clone() - ri)).unwrap()),
                        )
                        .into());
                    }
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<PyObject>>() {
                        // check data type of first element
                        if r.readonly()
                            .as_array()
                            .get(0)
                            .unwrap()
                            .as_ref(py)
                            .is_instance_of::<Self>()
                            .unwrap()
                        {
                            return Ok(PyArray::from_owned_object_array(
                                py,
                                r.as_array().mapv(|ri| {
                                    Py::new(py, Self(self.0.clone() - ri.extract::<Self>(py).unwrap().0))
                                        .unwrap()
                                }),
                            )
                            .into());
                        } else {
                            return Err(PyErr::new::<PyTypeError, _>(format!(
                                "Operation with the provided object type is not implemented. Supported data types are 'float', 'int' and '{}'.",
                                stringify!($py_type_name)
                            )));
                        }
                    }

                    Err(PyErr::new::<PyTypeError, _>(format!(
                        "Subtraction of \nleft:  {}\nright: {:?}\nis not implemented!",
                        stringify!($py_type_name),
                        rhs.get_type()
                    )))
                })
            }

            fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((-self.0.clone() + o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __mul__(&self, rhs: &PyAny) -> PyResult<PyObject> {
                Python::with_gil(|py| {
                    if let Ok(r) = rhs.extract::<f64>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() * r))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<Self>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() * r.0))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<f64>>() {
                        return Ok(PyArray::from_owned_object_array(
                            py,
                            r.as_array()
                                .mapv(|ri| Py::new(py, Self(self.0.clone() * ri)).unwrap()),
                        )
                        .into());
                    }
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<PyObject>>() {
                        // check data type of first element
                        if r.readonly()
                            .as_array()
                            .get(0)
                            .unwrap()
                            .as_ref(py)
                            .is_instance_of::<Self>()
                            .unwrap()
                        {
                            return Ok(PyArray::from_owned_object_array(
                                py,
                                r.as_array().mapv(|ri| {
                                    Py::new(py, Self(self.0.clone() * ri.extract::<Self>(py).unwrap().0))
                                        .unwrap()
                                }),
                            )
                            .into());
                        } else {
                            return Err(PyErr::new::<PyTypeError, _>(format!(
                                "Operation with the provided object type is not implemented. Supported data types are 'float', 'int' and '{}'.",
                                stringify!($py_type_name)
                            )));
                        }
                    }

                    Err(PyErr::new::<PyTypeError, _>(format!(
                        "Multiplication of \nleft:  {}\nright: {:?}\nis not implemented!",
                        stringify!($py_type_name),
                        rhs.get_type()
                    )))
                })
            }

            fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self.0.clone() * o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __truediv__(&self, rhs: &PyAny) -> PyResult<PyObject> {
                Python::with_gil(|py| {
                    if let Ok(r) = rhs.extract::<f64>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() / r))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<Self>() {
                        return Ok(PyCell::new(py, Self(self.0.clone() / r.0))?.to_object(py));
                    };
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<f64>>() {
                        return Ok(PyArray::from_owned_object_array(
                            py,
                            r.as_array()
                                .mapv(|ri| Py::new(py, Self(self.0.clone() / ri)).unwrap()),
                        )
                        .into());
                    }
                    if let Ok(r) = rhs.extract::<PyReadonlyArrayDyn<PyObject>>() {
                        // check data type of first element
                        if r.readonly()
                            .as_array()
                            .get(0)
                            .unwrap()
                            .as_ref(py)
                            .is_instance_of::<Self>()
                            .unwrap()
                        {
                            return Ok(PyArray::from_owned_object_array(
                                py,
                                r.as_array().mapv(|ri| {
                                    Py::new(py, Self(self.0.clone() / ri.extract::<Self>(py).unwrap().0))
                                        .unwrap()
                                }),
                            )
                            .into());
                        } else {
                            return Err(PyErr::new::<PyTypeError, _>(format!(
                                "Operation with the provided object type is not implemented. Supported data types are 'float', 'int' and '{}'.",
                                stringify!($py_type_name)
                            )));
                        }
                    }

                    Err(PyErr::new::<PyTypeError, _>(format!(
                        "Division of \nleft:  {}\nright: {:?}\nis not implemented!",
                        stringify!($py_type_name),
                        rhs.get_type()
                    )))
                })
            }

            fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self.0.recip() * o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __pow__(&self, rhs: &PyAny, _mod: Option<u32>) -> PyResult<Self> {
                if let Ok(r) = rhs.extract::<i32>() {
                    return Ok(self.0.powi(r).into());
                };
                if let Ok(r) = rhs.extract::<f64>() {
                    return Ok(self.0.powf(r).into());
                };
                if let Ok(r) = rhs.extract::<Self>() {
                    return Ok(self.0.powd(r.0).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __neg__(&self) -> PyResult<Self> {
                Ok((-self.0.clone()).into())
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }
    };
}
