use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[macro_use]
mod macros;
mod dual;
mod dual2;
mod dual3;
mod hyperdual;

use dual::derive1;
use dual3::derive3;
use hyperdual::derive2;

pub use dual::{
    PyDual64, PyDual64_10, PyDual64_2, PyDual64_3, PyDual64_4, PyDual64_5, PyDual64_6, PyDual64_7,
    PyDual64_8, PyDual64_9,
};
pub use dual2::{PyDual2Dual64, PyDual2_64};
pub use dual3::{PyDual3Dual64, PyDual3_64};
pub use hyperdual::{
    PyDual2_64_2, PyDual2_64_3, PyDual2_64_4, PyDual2_64_5, PyHyperDual64, PyHyperDual64_1_2,
    PyHyperDual64_1_3, PyHyperDual64_1_4, PyHyperDual64_1_5, PyHyperDual64_2_1, PyHyperDual64_2_2,
    PyHyperDual64_2_3, PyHyperDual64_2_4, PyHyperDual64_2_5, PyHyperDual64_3_1, PyHyperDual64_3_2,
    PyHyperDual64_3_3, PyHyperDual64_3_4, PyHyperDual64_3_5, PyHyperDual64_4_1, PyHyperDual64_4_2,
    PyHyperDual64_4_3, PyHyperDual64_4_4, PyHyperDual64_4_5, PyHyperDual64_5_1, PyHyperDual64_5_2,
    PyHyperDual64_5_3, PyHyperDual64_5_4, PyHyperDual64_5_5, PyHyperDualDual64,
};

pub fn num_dual(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyDual64>()?;
    m.add_class::<PyHyperDual64>()?;
    m.add_class::<PyDual2_64>()?;
    m.add_class::<PyDual3_64>()?;
    m.add_class::<PyHyperDualDual64>()?;
    m.add_class::<PyDual2Dual64>()?;
    m.add_class::<PyDual3Dual64>()?;
    m.add_function(wrap_pyfunction!(derive1, m)?).unwrap();
    m.add_function(wrap_pyfunction!(derive2, m)?).unwrap();
    m.add_function(wrap_pyfunction!(derive3, m)?).unwrap();
    Ok(())
}
