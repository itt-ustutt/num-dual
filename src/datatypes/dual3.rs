use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::*;

/// A scalar third order dual number for the calculation of third derivatives.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual3<T> {
    /// Real part of the third order dual number
    pub re: T,
    /// First derivative part of the third order dual number
    pub v1: T,
    /// Second derivative part of the third order dual number
    pub v2: T,
    /// Third derivative part of the third order dual number
    pub v3: T,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum> ndarray::ScalarOperand for Dual3<T> {}

pub type Dual3_32 = Dual3<f32>;
pub type Dual3_64 = Dual3<f64>;

impl<T> Dual3<T> {
    /// Create a new third order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T, v3: T) -> Self {
        Self { re, v1, v2, v3 }
    }
}

impl<T: One + Zero> Dual3<T> {
    /// Create a new third order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero(), T::zero())
    }

    /// Set the first derivative part to 1.
    /// ```
    /// # use num_dual::{Dual3, DualNum};
    /// let x = Dual3::from_re(5.0).derivative().powi(3);
    /// assert_eq!(x.re, 125.0);
    /// assert_eq!(x.v1, 75.0);
    /// assert_eq!(x.v2, 30.0);
    /// assert_eq!(x.v3, 6.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.v1 = T::one();
        self
    }
}

impl<T: DualNum> Dual3<T> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        let three = T::one() + T::one() + T::one();
        Self::new(
            f0,
            f1.clone() * &self.v1,
            f2.clone() * &self.v1 * &self.v1 + f1.clone() * &self.v2,
            f3 * &self.v1 * &self.v1 * &self.v1 + three * f2 * &self.v1 * &self.v2 + f1 * &self.v3,
        )
    }
}

impl<T: DualNum> Mul<&Dual3<T>> for &Dual3<T> {
    type Output = Dual3<T>;
    #[inline]
    fn mul(self, rhs: &Dual3<T>) -> Dual3<T> {
        let two = T::one() + T::one();
        let three = T::one() + &two;
        Dual3::new(
            self.re.clone() * &rhs.re,
            self.v1.clone() * &rhs.re + self.re.clone() * &rhs.v1,
            self.v2.clone() * &rhs.re + two * &self.v1 * &rhs.v1 + self.re.clone() * &rhs.v2,
            self.v3.clone() * &rhs.re
                + three * (self.v2.clone() * &rhs.v1 + self.v1.clone() * &rhs.v2)
                + self.re.clone() * &rhs.v3,
        )
    }
}

impl<T: DualNum> Div<&Dual3<T>> for &Dual3<T> {
    type Output = Dual3<T>;
    #[inline]
    fn div(self, rhs: &Dual3<T>) -> Dual3<T> {
        let rec = T::one() / &rhs.re;
        let f0 = rec.clone();
        let f1 = -f0.clone() * &rec;
        let f2 = -f1.clone() * &rec * T::Primitive::TWO;
        let f3 = -f2.clone() * rec * T::Primitive::THREE;
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display> fmt::Display for Dual3<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}v1 + {}v2 + {}v3",
            self.re, self.v1, self.v2, self.v3
        )
    }
}

impl_third_derivatives!(Dual3, [v1, v2, v3]);
impl_dual!(Dual3, [v1, v2, v3]);
#[cfg(feature = "nalgebra")]
impl_nalgebra!(Dual3, [v1, v2, v3]);
