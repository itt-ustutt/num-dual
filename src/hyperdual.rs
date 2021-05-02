use crate::dual::{Dual32, Dual64};
use crate::dual_n::{DualN32, DualN64};
use crate::{DualNum, DualNumMethods};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar hyper dual number for the calculation of second partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct HyperDual<T, F = T> {
    /// Real part of the hyper dual number
    pub re: T,
    /// Partial derivative part of the hyper dual number
    pub eps1: T,
    /// Partial derivative part of the hyper dual number
    pub eps2: T,
    /// Second partial derivative part of the hyper dual number
    pub eps1eps2: T,
    f: PhantomData<F>,
}

pub type HyperDual32 = HyperDual<f32>;
pub type HyperDual64 = HyperDual<f64>;
pub type HyperDualDual32 = HyperDual<Dual32, f32>;
pub type HyperDualDual64 = HyperDual<Dual64, f64>;
pub type HyperDualDualN32<const N: usize> = HyperDual<DualN32<N>, f32>;
pub type HyperDualDualN64<const N: usize> = HyperDual<DualN64<N>, f64>;

impl<T, F> HyperDual<T, F> {
    /// Create a new hyperdual number from its fields.
    #[inline]
    pub fn new(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        HyperDual {
            re,
            eps1,
            eps2,
            eps1eps2,
            f: PhantomData,
        }
    }
}

impl<T: Zero, F> HyperDual<T, F> {
    /// Create a new hyperdual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDual::new(re, T::zero(), T::zero(), T::zero())
    }
}

impl<T: One, F> HyperDual<T, F> {
    /// Derive a hyperdual number w.r.t. to the first variable.
    #[inline]
    pub fn derive1(mut self) -> Self {
        self.eps1 = T::one();
        self
    }
    /// Derive a hyperdual number w.r.t. to the second variable.
    /// ```
    /// # use num_hyperdual::{HyperDual, DualNumMethods};
    /// let x = HyperDual::from_re(5.0).derive1();
    /// let y = HyperDual::from_re(3.0).derive2();
    /// let z = x * y.powi(2);
    /// assert_eq!(z.re, 45.0);         // xy²
    /// assert_eq!(z.eps1, 9.0);        // y²
    /// assert_eq!(z.eps2, 30.0);       // 2xy
    /// assert_eq!(z.eps1eps2, 6.0);    // 2y
    /// ```
    #[inline]
    pub fn derive2(mut self) -> Self {
        self.eps2 = T::one();
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float> HyperDual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.eps1 * f1,
            self.eps2 * f1,
            self.eps1eps2 * f1 + self.eps1 * self.eps2 * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HyperDual<T, F>> for &'b HyperDual<T, F> {
    type Output = HyperDual<T, F>;
    #[inline]
    fn mul(self, other: &HyperDual<T, F>) -> HyperDual<T, F> {
        HyperDual::new(
            self.re * other.re,
            other.eps1 * self.re + self.eps1 * other.re,
            other.eps2 * self.re + self.eps2 * other.re,
            other.eps1eps2 * self.re
                + self.eps1 * other.eps2
                + other.eps1 * self.eps2
                + self.eps1eps2 * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HyperDual<T, F>> for &'b HyperDual<T, F> {
    type Output = HyperDual<T, F>;
    #[inline]
    fn div(self, other: &HyperDual<T, F>) -> HyperDual<T, F> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        HyperDual::new(
            self.re * inv,
            (self.eps1 * other.re - other.eps1 * self.re) * inv2,
            (self.eps2 * other.re - other.eps2 * self.re) * inv2,
            self.eps1eps2 * inv
                - (other.eps1eps2 * self.re + self.eps1 * other.eps2 + other.eps1 * self.eps2)
                    * inv2
                + other.eps1 * other.eps2 * (T::one() + T::one()) * self.re * inv2 * inv,
        )
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HyperDual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε1ε2",
            self.re, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

impl_second_derivatives!(HyperDual, [], [eps1, eps2, eps1eps2]);
impl_dual!(HyperDual, [], [eps1, eps2, eps1eps2]);
