#[cfg(feature = "linalg")]
mod linalg_ndarray;
#[cfg(feature = "linalg")]
pub use linalg_ndarray::*;

mod lu;
mod static_mat;

pub use lu::LU;
pub use static_mat::{StaticMat, StaticVec};

use std::fmt;

pub trait Scale<F> {
    fn scale(&mut self, f: F);
}

#[derive(Debug)]
pub struct LinAlgErr();

impl fmt::Display for LinAlgErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

impl std::error::Error for LinAlgErr {}
