use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A real number for the calculations of zeroth derivatives in generic contexts.
///
/// In most situations f64 or f32 can be used directly!
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Real<T> {
    /// Real part of the dual number
    pub re: T,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum> ndarray::ScalarOperand for Real<T> {}

impl<T> Real<T> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T) -> Self {
        Self { re }
    }
}

impl<T> Real<T> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re)
    }
}

/* chain rule */
impl<T> Real<T> {
    #[inline]
    fn chain_rule(&self, f0: T) -> Self {
        Self::new(f0)
    }
}

/* product rule */
impl<T: DualNum> Mul<&Real<T>> for &Real<T> {
    type Output = Real<T>;
    #[inline]
    fn mul(self, other: &Real<T>) -> Self::Output {
        Real::new(self.re.clone() * other.re.clone())
    }
}

/* quotient rule */
impl<T: DualNum> Div<&Real<T>> for &Real<T> {
    type Output = Real<T>;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: &Real<T>) -> Real<T> {
        let inv = other.re.recip();
        Real::new(self.re.clone() * inv.clone())
    }
}

/* string conversions */
impl<T: DualNum> fmt::Display for Real<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.re, f)
    }
}

impl_zeroth_derivatives!(Real, []);
impl_dual!(Real, []);
impl_nalgebra!(Real, []);
