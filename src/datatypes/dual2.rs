use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar second order dual number for the calculation of second derivatives.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual2<T> {
    /// Real part of the second order dual number
    pub re: T,
    /// First derivative part of the second order dual number
    pub v1: T,
    /// Second derivative part of the second order dual number
    pub v2: T,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum> ndarray::ScalarOperand for Dual2<T> {}

pub type Dual2_32 = Dual2<f32>;
pub type Dual2_64 = Dual2<f64>;

impl<T> Dual2<T> {
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T) -> Self {
        Self { re, v1, v2 }
    }
}

impl<T: One> Dual2<T> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual2, DualNum};
    /// let x = Dual2::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);             // x²
    /// assert_eq!(x.v1, 10.0);    // 2x
    /// assert_eq!(x.v2, 2.0);     // 2
    /// ```
    ///
    /// Can also be used for higher order derivatives.
    /// ```
    /// # use num_dual::{Dual64, Dual2, DualNum};
    /// let x = Dual2::from_re(Dual64::from_re(5.0).derivative())
    ///     .derivative()
    ///     .powi(2);
    /// assert_eq!(x.re.re, 25.0);      // x²
    /// assert_eq!(x.re.eps, 10.0);     // 2x
    /// assert_eq!(x.v1.re, 10.0);      // 2x
    /// assert_eq!(x.v1.eps, 2.0);      // 2
    /// assert_eq!(x.v2.re, 2.0);       // 2
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.v1 = T::one();
        self
    }
}

impl<T: Zero> Dual2<T> {
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero())
    }
}

/* chain rule */
impl<T: DualNum> Dual2<T> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.v1.clone() * f1.clone(),
            self.v2.clone() * f1 + self.v1.clone() * self.v1.clone() * f2,
        )
    }
}

/* product rule */
impl<T: DualNum> Mul<&Dual2<T>> for &Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn mul(self, other: &Dual2<T>) -> Dual2<T> {
        Dual2::new(
            self.re.clone() * other.re.clone(),
            other.v1.clone() * self.re.clone() + self.v1.clone() * other.re.clone(),
            other.v2.clone() * self.re.clone()
                + self.v1.clone() * other.v1.clone()
                + other.v1.clone() * self.v1.clone()
                + self.v2.clone() * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum> Div<&Dual2<T>> for &Dual2<T> {
    type Output = Dual2<T>;
    #[inline]
    fn div(self, other: &Dual2<T>) -> Dual2<T> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * inv.clone();
        Dual2::new(
            self.re.clone() * inv.clone(),
            (self.v1.clone() * other.re.clone() - other.v1.clone() * self.re.clone())
                * inv2.clone(),
            self.v2.clone() * inv.clone()
                - (other.v2.clone() * self.re.clone()
                    + self.v1.clone() * other.v1.clone()
                    + other.v1.clone() * self.v1.clone())
                    * inv2.clone()
                + other.v1.clone()
                    * other.v1.clone()
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum> fmt::Display for Dual2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε1 + {}ε1²", self.re, self.v1, self.v2)
    }
}

impl_second_derivatives!(Dual2, [v1, v2]);
impl_dual!(Dual2, [v1, v2]);
#[cfg(feature = "nalgebra")]
impl_nalgebra!(Dual2, [v1, v2]);
