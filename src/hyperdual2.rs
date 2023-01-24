use crate::{DualNum, DualNumFloat, IsDerivativeZero};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar third order dual number for the calculation of third derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct HyperDual2<T, F = T> {
    /// Real part of the third order hyper dual number
    pub re: T,
    /// First derivative part of the third order hyper dual number
    pub eps1: T,
    /// First derivative part of the third order hyper dual number
    pub eps2: T,
    /// First derivative part of the third order hyper dual number
    pub eps3: T,
    /// Second derivative part of the third order hyper dual number
    pub eps1eps2: T,
    /// Second derivative part of the third order hyper dual number
    pub eps1eps3: T,
    /// Second derivative part of the third order hyper dual number
    pub eps2eps3: T,
    /// Third derivative part of the third order hyper dual number
    pub eps1eps2eps3: T,
    f: PhantomData<F>,
}

pub type HyperDual2_32 = HyperDual2<f32>;
pub type HyperDual2_64 = HyperDual2<f64>;

impl<T, F> HyperDual2<T, F> {
    /// Create a new third order hyper dual number from its fields.
    #[inline]
    pub fn new(
        re: T,
        eps1: T,
        eps2: T,
        eps3: T,
        eps1eps2: T,
        eps1eps3: T,
        eps2eps3: T,
        eps1eps2eps3: T,
    ) -> Self {
        Self {
            re,
            eps1,
            eps2,
            eps3,
            eps1eps2,
            eps1eps3,
            eps2eps3,
            eps1eps2eps3,
            f: PhantomData,
        }
    }
}

impl<T: Zero, F> HyperDual2<T, F> {
    /// Create a new third order hyper dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(
            re,
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
    }
}

impl<T: Clone + Zero + One, F> HyperDual2<T, F> {
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
    pub fn derive1(mut self) -> Self {
        self.eps1 = T::one();
        self
    }

    #[inline]
    pub fn derive2(mut self) -> Self {
        self.eps2 = T::one();
        self
    }

    #[inline]
    pub fn derive3(mut self) -> Self {
        self.eps3 = T::one();
        self
    }
}

impl<T: DualNum<F>, F: Float> HyperDual2<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        // let three = T::one() + T::one() + T::one();
        Self::new(
            f0,
            f1 * self.eps1,
            f1 * self.eps2,
            f1 * self.eps3,
            f1 * self.eps1eps2 + f2 * self.eps1 * self.eps2,
            f1 * self.eps1eps3 + f2 * self.eps1 * self.eps3,
            f1 * self.eps2eps3 + f2 * self.eps2 * self.eps3,
            f2 * (self.eps1 * self.eps2eps3
                + self.eps2 * self.eps1eps3
                + self.eps3 * self.eps1eps2)
                + f3 * self.eps1 * self.eps2 * self.eps3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HyperDual2<T, F>> for &'b HyperDual2<T, F> {
    type Output = HyperDual2<T, F>;
    #[inline]
    fn mul(self, rhs: &HyperDual2<T, F>) -> HyperDual2<T, F> {
        HyperDual2::new(
            self.re * rhs.re,
            self.eps1 * rhs.re + self.re * rhs.eps1,
            self.eps2 * rhs.re + self.re * rhs.eps2,
            self.eps3 * rhs.re + self.re * rhs.eps3,
            self.eps1eps2 * rhs.re
                + self.eps1 * rhs.eps2
                + self.eps2 * rhs.eps1
                + self.re * rhs.eps1eps2,
            self.eps1eps3 * rhs.re
                + self.eps1 * rhs.eps3
                + self.eps3 * rhs.eps1
                + self.re * rhs.eps1eps3,
            self.eps2eps3 * rhs.re
                + self.eps2 * rhs.eps3
                + self.eps3 * rhs.eps2
                + self.re * rhs.eps2eps3,
            self.eps1eps2eps3 * rhs.re
                + self.eps1 * rhs.eps2eps3
                + self.eps2 * rhs.eps1eps3
                + self.eps3 * rhs.eps1eps2
                + self.eps2eps3 * rhs.eps1
                + self.eps1eps3 * rhs.eps2
                + self.eps1eps2 * rhs.eps3
                + self.re * rhs.eps1eps2eps3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HyperDual2<T, F>> for &'b HyperDual2<T, F> {
    type Output = HyperDual2<T, F>;
    #[inline]
    fn div(self, rhs: &HyperDual2<T, F>) -> HyperDual2<T, F> {
        let rec = T::one() / rhs.re;
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HyperDual2<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε3 + {}ε1ε2 + {}ε1ε3 + {}ε2ε3 + {}ε1ε2ε3",
            self.re,
            self.eps1,
            self.eps2,
            self.eps3,
            self.eps1eps2,
            self.eps1eps3,
            self.eps2eps3,
            self.eps1eps2eps3
        )
    }
}

impl_third_derivatives!(
    HyperDual2,
    [],
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
impl_dual!(
    HyperDual2,
    [],
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
