use crate::{DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar third order dual number for the calculation of third derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct Dual3<T, F = T> {
    /// Real part of the third order dual number
    pub re: T,
    /// First derivative part of the third order dual number
    pub v1: T,
    /// Second derivative part of the third order dual number
    pub v2: T,
    /// Third derivative part of the third order dual number
    pub v3: T,
    f: PhantomData<F>,
}

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

impl<T: Zero, F> Dual3<T, F> {
    /// Create a new third order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero(), T::zero())
    }
}

impl<T: Clone + Zero + One, F> Dual3<T, F> {
    /// Derive a third order dual number, i.e. set the first derivative part to 1.
    /// ```
    /// # use num_dual::{Dual3, DualNum};
    /// let x = Dual3::from_re(5.0).derive().powi(3);
    /// assert_eq!(x.re, 125.0);
    /// assert_eq!(x.v1, 75.0);
    /// assert_eq!(x.v2, 30.0);
    /// assert_eq!(x.v3, 6.0);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
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
            f1 * self.v1,
            f2 * self.v1 * self.v1 + f1 * self.v2,
            f3 * self.v1 * self.v1 * self.v1 + three * f2 * self.v1 * self.v2 + f1 * self.v3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a Dual3<T, F>> for &'b Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn mul(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
        let two = T::one() + T::one();
        let three = two + T::one();
        Dual3::new(
            self.re * rhs.re,
            self.v1 * rhs.re + self.re * rhs.v1,
            self.v2 * rhs.re + two * self.v1 * rhs.v1 + self.re * rhs.v2,
            self.v3 * rhs.re
                + three * self.v2 * rhs.v1
                + three * self.v1 * rhs.v2
                + self.re * rhs.v3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a Dual3<T, F>> for &'b Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn div(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
        let rec = T::one() / rhs.re;
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
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

impl_third_derivatives!(Dual3, [], [v1, v2, v3]);
impl_dual!(Dual3, [], [v1, v2, v3]);
