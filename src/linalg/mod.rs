mod linalg_ndarray;
pub use linalg_ndarray::*;

mod lu;
pub use lu::LU;

use std::fmt;

#[derive(Debug)]
pub struct LinAlgErr();

impl fmt::Display for LinAlgErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

impl std::error::Error for LinAlgErr {}
