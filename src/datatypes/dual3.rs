use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar third order dual number for the calculation of third derivatives.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual3<T, F = T> {
    /// Real part of the third order dual number
    pub re: T,
    /// First derivative part of the third order dual number
    pub v1: T,
    /// Second derivative part of the third order dual number
    pub v2: T,
    /// Third derivative part of the third order dual number
    pub v3: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    f: PhantomData<F>,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum<F>, F: DualNumFloat> ndarray::ScalarOperand for Dual3<T, F> {}

pub type Dual3_32 = Dual3<f32>;
pub type Dual3_64 = Dual3<f64>;

impl<T, F> Dual3<T, F> {
    /// Create a new third order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            re,
            v1,
            v2,
            v3,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual3<T, F> {
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

impl<T: DualNum<F>, F: Float> Dual3<T, F> {
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

impl<T: DualNum<F>, F: Float> Mul<&Dual3<T, F>> for &Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn mul(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
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

impl<T: DualNum<F>, F: Float> Div<&Dual3<T, F>> for &Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn div(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
        let rec = T::one() / &rhs.re;
        let f0 = rec.clone();
        let f1 = -f0.clone() * &rec;
        let f2 = f1.clone() * &rec * F::from(-2.0).unwrap();
        let f3 = f2.clone() * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for Dual3<T, F> {
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

/// Comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialEq, F: Float> PartialEq for Dual3<T, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}
impl<T: DualNum<F> + PartialEq, F: Float> PartialEq<T> for Dual3<T, F> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.re.eq(&other)
    }
}

/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialOrd, F: Float> PartialOrd for Dual3<T, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}
impl<T: DualNum<F> + PartialOrd, F: Float> PartialOrd<T> for Dual3<T, F> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other)
    }
}
