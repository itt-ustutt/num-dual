use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A real number for the calculations of zeroth derivatives in generic contexts.
///
/// In most situations f64 or f32 can be used directly!
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Real<T: DualNum<F>, F> {
    /// Real part of the dual number
    pub re: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    f: PhantomData<F>,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum<F>, F: DualNumFloat> ndarray::ScalarOperand for Real<T, F> {}

impl<T: DualNum<F>, F> Real<T, F> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T) -> Self {
        Self { re, f: PhantomData }
    }
}

impl<T: DualNum<F> + Zero, F> Real<T, F> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re)
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float> Real<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T) -> Self {
        Self::new(f0)
    }
}

/* product rule */
impl<T: DualNum<F>, F: Float> Mul<&Real<T, F>> for &Real<T, F> {
    type Output = Real<T, F>;
    #[inline]
    fn mul(self, other: &Real<T, F>) -> Self::Output {
        Real::new(self.re.clone() * other.re.clone())
    }
}

/* quotient rule */
impl<T: DualNum<F>, F: Float> Div<&Real<T, F>> for &Real<T, F> {
    type Output = Real<T, F>;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: &Real<T, F>) -> Real<T, F> {
        let inv = other.re.recip();
        Real::new(self.re.clone() * inv.clone())
    }
}

/* string conversions */
impl<T: DualNum<F>, F> fmt::Display for Real<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.re, f)
    }
}

impl_zeroth_derivatives!(Real, []);
impl_dual!(Real, []);
