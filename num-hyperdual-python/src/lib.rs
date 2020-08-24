use num_hyperdual::DualNum;
use num_hyperdual::{Dual64 as D64, HyperDual64 as HD64};
use pyo3::exceptions::TypeError;
use pyo3::prelude::*;
use pyo3::PyNumberProtocol;

#[pyclass]
#[derive(Clone)]
pub struct Dual64 {
    _data: D64,
}

impl From<D64> for Dual64 {
    fn from(d: D64) -> Self {
        Self { _data: d }
    }
}

#[pymethods]
impl Dual64 {
    #[new]
    pub fn new(re: f64, eps: f64) -> Self {
        Self {
            _data: D64::new(re, eps),
        }
    }

    #[getter]
    fn get_eps(&self) -> f64 {
        self._data.eps
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HyperDual64 {
    _data: HD64,
}

impl From<HD64> for HyperDual64 {
    fn from(hd: HD64) -> Self {
        Self { _data: hd }
    }
}

#[pymethods]
impl HyperDual64 {
    #[new]
    pub fn new(re: f64, eps1: f64, eps2: f64, eps1eps2: f64) -> Self {
        Self {
            _data: HD64::new(re, eps1, eps2, eps1eps2),
        }
    }

    #[getter]
    fn get_eps1(&self) -> f64 {
        self._data.eps1
    }

    #[getter]
    fn get_eps2(&self) -> f64 {
        self._data.eps2
    }

    #[getter]
    fn get_eps1eps2(&self) -> f64 {
        self._data.eps1eps2
    }
}

macro_rules! impl_dual_num {
    ($type_name:ty, $data_type_name:ty) => {
        #[pymethods]
        impl $type_name {
            #[staticmethod]
            pub fn from_re(re: f64) -> Self {
                Self::from(<$data_type_name>::from(re))
            }

            #[getter]
            fn get_re(&self) -> f64 {
                self._data.re()
            }

            pub fn recip(&self) -> Self {
                Self {
                    _data: self._data.recip(),
                }
            }

            pub fn powi(&self, n: i32) -> Self {
                Self {
                    _data: self._data.powi(n),
                }
            }

            pub fn powf(&self, n: f64) -> Self {
                Self {
                    _data: self._data.powf(n),
                }
            }

            pub fn sqrt(&self) -> Self {
                Self {
                    _data: self._data.sqrt(),
                }
            }

            pub fn cbrt(&self) -> Self {
                Self {
                    _data: self._data.cbrt(),
                }
            }

            pub fn exp(&self) -> Self {
                Self {
                    _data: self._data.exp(),
                }
            }

            pub fn exp2(&self) -> Self {
                Self {
                    _data: self._data.exp2(),
                }
            }

            pub fn exp_m1(&self) -> Self {
                Self {
                    _data: self._data.exp_m1(),
                }
            }

            pub fn ln(&self) -> Self {
                Self {
                    _data: self._data.ln(),
                }
            }

            pub fn log(&self, base: f64) -> Self {
                Self {
                    _data: self._data.log(base),
                }
            }

            pub fn log2(&self) -> Self {
                Self {
                    _data: self._data.log2(),
                }
            }

            pub fn log10(&self) -> Self {
                Self {
                    _data: self._data.log10(),
                }
            }

            pub fn ln_1p(&self) -> Self {
                Self {
                    _data: self._data.ln_1p(),
                }
            }

            pub fn sin(&self) -> Self {
                Self {
                    _data: self._data.sin(),
                }
            }

            pub fn cos(&self) -> Self {
                Self {
                    _data: self._data.cos(),
                }
            }

            pub fn tan(&self) -> Self {
                Self {
                    _data: self._data.tan(),
                }
            }

            pub fn sin_cos(&self) -> (Self, Self) {
                let (a, b) = self._data.sin_cos();
                (Self::from(a), Self::from(b))
            }
        }

        #[pyproto]
        impl PyNumberProtocol for $type_name {
            // fn __add__(lhs: PyRef<'p, Self>, rhs: &PyAny) -> PyResult<Self> {
            //     if let Ok(r) = rhs.extract::<f64>() {
            //         return Ok(Self {
            //             _data: lhs._data + r,
            //         });
            //     };
            //     if let Ok(r) = rhs.extract::<Self>() {
            //         return Ok(Self {
            //             _data: lhs._data + r._data,
            //         });
            //     };
            //     Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            // }

            // fn __radd__(&self, rhs: &PyAny) -> PyResult<Self> {
            //     unimplemented!()
            // }

            fn __add__(lhs: &PyAny, rhs: &PyAny) -> PyResult<Self> {
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                    return Ok(Self { _data: l._data + r });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<f64>(), rhs.extract::<Self>()) {
                    return Ok(Self { _data: l + r._data });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<Self>()) {
                    return Ok(Self {
                        _data: l._data + r._data,
                    });
                };
                Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            }

            fn __sub__(lhs: &PyAny, rhs: &PyAny) -> PyResult<Self> {
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                    return Ok(Self { _data: l._data - r });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<f64>(), rhs.extract::<Self>()) {
                    return Ok(Self { _data: l - r._data });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<Self>()) {
                    return Ok(Self {
                        _data: l._data - r._data,
                    });
                };
                Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            }

            fn __mul__(lhs: &PyAny, rhs: &PyAny) -> PyResult<Self> {
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                    return Ok(Self { _data: l._data * r });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<f64>(), rhs.extract::<Self>()) {
                    return Ok(Self { _data: l * r._data });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<Self>()) {
                    return Ok(Self {
                        _data: l._data * r._data,
                    });
                };
                Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            }

            fn __truediv__(lhs: &PyAny, rhs: &PyAny) -> PyResult<Self> {
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                    return Ok(Self { _data: l._data / r });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<f64>(), rhs.extract::<Self>()) {
                    return Ok(Self { _data: l / r._data });
                };
                if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<Self>()) {
                    return Ok(Self {
                        _data: l._data / r._data,
                    });
                };
                Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            }

            fn __pow__(lhs: &PyAny, rhs: i32, _mod: Option<u32>) -> PyResult<Self> {
                // if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<f64>()) {
                //     return Ok(Self { _data: l._data.powf(r) });
                // };
                // if let (Ok(l), Ok(r)) = (lhs.extract::<Self>(), rhs.extract::<i8>()) {
                //     return Ok(Self { _data: l._data.powi(r) });
                // };
                if let Ok(l) = lhs.extract::<Self>() {
                    return Ok(Self {
                        _data: l._data.powi(rhs),
                    });
                };
                Err(PyErr::new::<TypeError, _>(format!("not implemented!")))
            }
        }

        #[pyproto]
        impl pyo3::class::basic::PyObjectProtocol for $type_name {
            fn __repr__(&self) -> PyResult<String> {
                Ok(self._data.to_string())
            }
        }
    };
}

impl_dual_num!(Dual64, D64);
impl_dual_num!(HyperDual64, HD64);

#[pymodule]
fn hyperdual(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dual64>()?;
    m.add_class::<HyperDual64>()?;
    Ok(())
}
