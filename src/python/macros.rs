macro_rules! impl_dual_num {
    ($py_type_name:ty, $data_type:ty, $field_type:ty) => {
        impl From<$data_type> for $py_type_name {
            fn from(d: $data_type) -> Self {
                Self { _data: d }
            }
        }

        impl From<$py_type_name> for $data_type {
            fn from(d: $py_type_name) -> Self {
                d._data
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
                self._data.re.into()
            }

            #[inline]
            /// Reciprocal value of self.
            pub fn recip(&self) -> Self {
                self._data.recip().into()
            }

            #[inline]
            /// Power using 32-bit integer as exponent.
            pub fn powi(&self, n: i32) -> Self {
                self._data.powi(n).into()
            }

            #[inline]
            /// Power using 64-bin float as exponent.
            pub fn powf(&self, n: f64) -> Self {
                self._data.powf(n).into()
            }

            #[inline]
            /// Power using self (hyper) dual number as exponent.
            pub fn powd(&self, n: Self) -> Self {
                self._data.powd(&n._data).into()
            }

            #[inline]
            /// Sqaure root.
            pub fn sqrt(&self) -> Self {
                self._data.sqrt().into()
            }

            #[inline]
            /// Cubic root.
            pub fn cbrt(&self) -> Self {
                self._data.cbrt().into()
            }

            #[inline]
            /// Calculate the exponential of (hyper) dual number.
            pub fn exp(&self) -> Self {
                self._data.exp().into()
            }

            #[inline]
            /// Calculate 2**x of (hyper) dual number x.
            pub fn exp2(&self) -> Self {
                self._data.exp2().into()
            }

            #[inline]
            /// Calculate exp(x) - 1.
            pub fn expm1(&self) -> Self {
                self._data.exp_m1().into()
            }

            #[inline]
            /// Calculate natural logarithm.
            pub fn log(&self) -> Self {
                self._data.ln().into()
            }

            #[inline]
            /// Calculate logarithm with given base.
            pub fn log_base(&self, base: f64) -> Self {
                self._data.log(base).into()
            }

            #[inline]
            /// Calculate logarithm with base 2.
            pub fn log2(&self) -> Self {
                self._data.log2().into()
            }

            #[inline]
            /// Calculate logarithm with base 10.
            pub fn log10(&self) -> Self {
                self._data.log10().into()
            }

            #[inline]
            /// Returns ln(1+n) (natural logarithm) more accurately than if the operations were performed separately.
            pub fn log1p(&self) -> Self {
                self._data.ln_1p().into()
            }

            #[inline]
            /// Hyperbolic sine function.
            pub fn sin(&self) -> Self {
                self._data.sin().into()
            }

            #[inline]
            /// Hyperbolic cosine function.
            pub fn cos(&self) -> Self {
                self._data.cos().into()
            }

            #[inline]
            /// Computes the tangent of a (hyper) dual number (in radians).
            pub fn tan(&self) -> Self {
                self._data.tan().into()
            }

            #[inline]
            /// Simultaneously computes the sine and cosine of the (hyper) dual number, x.
            pub fn sin_cos(&self) -> (Self, Self) {
                let (a, b) = self._data.sin_cos();
                (a.into(), b.into())
            }

            #[inline]
            /// Computes the arcsine of a (hyper) dual number.
            pub fn arcsin(&self) -> Self {
                self._data.asin().into()
            }

            #[inline]
            /// Computes the arccosine of a (hyper) dual number.
            pub fn arccos(&self) -> Self {
                self._data.acos().into()
            }

            #[inline]
            /// Computes the arctangent of a (hyper) dual number.
            pub fn arctan(&self) -> Self {
                self._data.atan().into()
            }

            #[inline]
            /// Computes the hyperbolic sine of a (hyper) dual number.
            pub fn sinh(&self) -> Self {
                self._data.sinh().into()
            }

            #[inline]
            /// Computes the hyperbolic cosine of a (hyper) dual number.
            pub fn cosh(&self) -> Self {
                self._data.cosh().into()
            }

            #[inline]
            /// Computes the hyperbolic tangent of a (hyper) dual number.
            pub fn tanh(&self) -> Self {
                self._data.tanh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic sine of a (hyper) dual number.
            pub fn arcsinh(&self) -> Self {
                self._data.asinh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic cosine of a (hyper) dual number.
            pub fn arccosh(&self) -> Self {
                self._data.acosh().into()
            }

            #[inline]
            /// Computes the inverse hyperbolic tangent of a (hyper) dual number.
            pub fn arctanh(&self) -> Self {
                self._data.atanh().into()
            }

            #[inline]
            /// Computes the first spherical bessel function.
            pub fn sph_j0(&self) -> Self {
                self._data.sph_j0().into()
            }
            #[inline]
            /// Computes the second spherical bessel function.
            pub fn sph_j1(&self) -> Self {
                self._data.sph_j1().into()
            }

            #[inline]
            /// Computes the third spherical bessel function.
            pub fn sph_j2(&self) -> Self {
                self._data.sph_j2().into()
            }

            #[inline]
            #[pyo3(text_signature = "($self, b: Self, c: Self)")]
            /// Fused multiply-add. Computes (self * a) + b with only one rounding error.
            fn mul_add(&self, a: Self, b: Self) -> Self {
                self._data.mul_add(a._data, b._data).into()
            }
        }

        #[pyproto]
        impl PyNumberProtocol for $py_type_name {
            fn __add__(lhs: PyRef<'p, Self>, rhs: &PyAny) -> PyResult<Self> {
                if let Ok(r) = rhs.extract::<f64>() {
                    return Ok((lhs._data + r).into());
                };
                if let Ok(r) = rhs.extract::<Self>() {
                    return Ok((lhs._data + r._data).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self._data + o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __sub__(lhs: PyRef<'p, Self>, rhs: &PyAny) -> PyResult<Self> {
                if let Ok(r) = rhs.extract::<f64>() {
                    return Ok((lhs._data - r).into());
                };
                if let Ok(r) = rhs.extract::<Self>() {
                    return Ok((lhs._data - r._data).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((-self._data + o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __mul__(lhs: PyRef<'p, Self>, rhs: &PyAny) -> PyResult<Self> {
                if let Ok(r) = rhs.extract::<f64>() {
                    return Ok((lhs._data * r).into());
                };
                if let Ok(r) = rhs.extract::<Self>() {
                    return Ok((lhs._data * r._data).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self._data * o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __truediv__(lhs: PyRef<'p, Self>, rhs: &PyAny) -> PyResult<Self> {
                if let Ok(r) = rhs.extract::<f64>() {
                    return Ok((lhs._data / r).into());
                };
                if let Ok(r) = rhs.extract::<Self>() {
                    return Ok((lhs._data / r._data).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
                if let Ok(o) = other.extract::<f64>() {
                    return Ok((self._data.recip() * o).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __pow__(lhs: &PyAny, rhs: &PyAny, _mod: Option<u32>) -> PyResult<Self> {
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<i32>()) {
                    return Ok(l._data.powi(r).into());
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                    return Ok(l._data.powf(r).into());
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<Self>()) {
                    return Ok(l._data.powd(&r._data).into());
                };
                Err(PyErr::new::<PyTypeError, _>(format!("not implemented!")))
            }

            fn __neg__(&self) -> PyResult<Self> {
                Ok((-self._data).into())
            }
        }

        #[pyproto]
        impl pyo3::class::basic::PyObjectProtocol for $py_type_name {
            fn __repr__(&self) -> PyResult<String> {
                Ok(self._data.to_string())
            }
        }
    };
}
