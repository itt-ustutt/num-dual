use crate::{DualNum, DualNumFloat, IsDerivativeZero};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar hyper hyper dual number for the calculation of third partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct HyperHyperDual<T, F = T> {
    /// Real part of the hyper hyper dual number
    pub re: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps1: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps2: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps3: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps1eps2: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps1eps3: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps2eps3: T,
    /// Third partial derivative part of the hyper hyper dual number
    pub eps1eps2eps3: T,
    f: PhantomData<F>,
}

pub type HyperHyperDual32 = HyperHyperDual<f32>;
pub type HyperHyperDual64 = HyperHyperDual<f64>;

impl<T, F> HyperHyperDual<T, F> {
    /// Create a new hyper hyper dual number from its fields.
    #[inline]
    #[allow(clippy::too_many_arguments)]
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

impl<T: Zero, F> HyperHyperDual<T, F> {
    /// Create a new hyper hyper dual number from the real part.
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

impl<T: Clone + Zero + One, F> HyperHyperDual<T, F> {
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

impl<T: DualNum<F>, F: Float> HyperHyperDual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        Self::new(
            f0,
            f1 * self.eps1,
            f1 * self.eps2,
            f1 * self.eps3,
            f1 * self.eps1eps2 + f2 * self.eps1 * self.eps2,
            f1 * self.eps1eps3 + f2 * self.eps1 * self.eps3,
            f1 * self.eps2eps3 + f2 * self.eps2 * self.eps3,
            f1 * self.eps1eps2eps3
                + f2 * (self.eps1 * self.eps2eps3
                    + self.eps2 * self.eps1eps3
                    + self.eps3 * self.eps1eps2)
                + f3 * self.eps1 * self.eps2 * self.eps3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HyperHyperDual<T, F>> for &'b HyperHyperDual<T, F> {
    type Output = HyperHyperDual<T, F>;
    #[inline]
    fn mul(self, rhs: &HyperHyperDual<T, F>) -> HyperHyperDual<T, F> {
        HyperHyperDual::new(
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

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HyperHyperDual<T, F>> for &'b HyperHyperDual<T, F> {
    type Output = HyperHyperDual<T, F>;
    #[inline]
    fn div(self, rhs: &HyperHyperDual<T, F>) -> HyperHyperDual<T, F> {
        let rec = T::one() / rhs.re;
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HyperHyperDual<T, F> {
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
    HyperHyperDual,
    [],
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
impl_dual!(
    HyperHyperDual,
    [],
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
