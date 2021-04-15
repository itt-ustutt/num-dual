mod linalg_ndarray;
mod lu;
mod static_mat;

pub use linalg_ndarray::*;
pub use lu::LU;
pub use static_mat::{StaticMat, StaticVec};

use num_traits::Signed;
use std::fmt;

pub trait Scale<F> {
    fn scale(&mut self, f: F);
}

pub trait LinAlgNum: Copy + Signed + PartialOrd + From<f64> {}
impl<T> LinAlgNum for T where T: Copy + Signed + PartialOrd + From<f64> {}

#[derive(Debug)]
pub struct LinAlgErr();

impl fmt::Display for LinAlgErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

impl std::error::Error for LinAlgErr {}
