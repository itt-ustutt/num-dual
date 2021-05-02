use crate::{Dual32, Dual64, DualN32, DualN64, DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar hyper dual number for the calculation of second derivatives
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HD2<T, F = T> {
    /// Real part of the hyper dual number
    pub re: T,
    /// First derivative part of the hyper dual number
    pub v1: T,
    /// Second derivative part of the hyper dual number
    pub v2: T,
    f: PhantomData<F>,
}

pub type HD2_32 = HD2<f32>;
pub type HD2_64 = HD2<f64>;
pub type HD2Dual32 = HD2<Dual32, f32>;
pub type HD2Dual64 = HD2<Dual64, f64>;
pub type HD2DualN32<const N: usize> = HD2<DualN32<N>, f32>;
pub type HD2DualN64<const N: usize> = HD2<DualN64<N>, f64>;

impl<T, F> HD2<T, F> {
    /// Create a new hyper dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T) -> Self {
        Self {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
}

impl<T: Zero, F> HD2<T, F> {
    /// Create a new hyper dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero())
    }
}

impl<T: Clone + Zero + One, F> HD2<T, F> {
    /// Derive a hyper dual number, i.e. set the first derivative part to 1.
    /// ```
    /// # use num_hyperdual::{HD2, DualNum};
    /// let x = HD2::from_re(5.0).derive().powi(3);
    /// assert_eq!(x.re, 125.0);
    /// assert_eq!(x.v1, 75.0);
    /// assert_eq!(x.v2, 30.0);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        self.v1 = T::one();
        self
    }
}

impl<T: DualNum<F>, F: Float> HD2<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(f0, f1 * self.v1, f2 * self.v1 * self.v1 + f1 * self.v2)
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HD2<T, F>> for &'b HD2<T, F> {
    type Output = HD2<T, F>;
    #[inline]
    fn mul(self, rhs: &HD2<T, F>) -> HD2<T, F> {
        let two = T::one() + T::one();
        HD2::new(
            self.re * rhs.re,
            self.v1 * rhs.re + self.re * rhs.v1,
            self.v2 * rhs.re + two * self.v1 * rhs.v1 + self.re * rhs.v2,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HD2<T, F>> for &'b HD2<T, F> {
    type Output = HD2<T, F>;
    #[inline]
    fn div(self, rhs: &HD2<T, F>) -> HD2<T, F> {
        let rec = T::one() / rhs.re;
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HD2<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}v1 + {}v2", self.re, self.v1, self.v2)
    }
}

impl_second_derivatives!(HD2, [], [v1, v2]);
impl_dual!(HD2, [], [v1, v2]);
