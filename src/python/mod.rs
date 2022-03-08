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

pub use dual::PyDual64;
pub use dual2::{PyDual2Dual64, PyDual2_64};
pub use dual3::{PyDual3Dual64, PyDual3_64};
pub use hyperdual::{PyHyperDual64, PyHyperDualDual64};

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
