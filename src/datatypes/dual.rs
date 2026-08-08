use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar dual number for the calculations of first derivatives.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual<T> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: T,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum> ndarray::ScalarOperand for Dual<T> {}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

impl<T> Dual<T> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Self { re, eps }
    }
}

impl<T: Zero> Dual<T> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero())
    }
}

impl<T: One> Dual<T> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps, 10.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.eps = T::one();
        self
    }
}

/* chain rule */
impl<T: DualNum> Dual<T> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, self.eps.clone() * f1)
    }
}

/* product rule */
impl<T: DualNum> Mul<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn mul(self, other: &Dual<T>) -> Self::Output {
        Dual::new(
            self.re.clone() * other.re.clone(),
            self.eps.clone() * other.re.clone() + other.eps.clone() * self.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum> Div<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn div(self, other: &Dual<T>) -> Dual<T> {
        let inv = other.re.recip();
        Dual::new(
            self.re.clone() * inv.clone(),
            (self.eps.clone() * other.re.clone() - other.eps.clone() * self.re.clone())
                * inv.clone()
                * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl_first_derivatives!(Dual, [eps]);
impl_dual!(Dual, [eps]);
#[cfg(feature = "nalgebra")]
impl_nalgebra!(Dual, [eps]);
