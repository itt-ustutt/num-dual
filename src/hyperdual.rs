use crate::{DualNum, DualNumFloat, IsDerivativeZero, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A hyper dual number for the calculation of second partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct HyperDualVec<T, F, const M: usize, const N: usize> {
    /// Real part of the hyper dual number
    pub re: T,
    /// Partial derivative part of the hyper dual number
    pub eps1: StaticVec<T, M>,
    /// Partial derivative part of the hyper dual number
    pub eps2: StaticVec<T, N>,
    /// Second partial derivative part of the hyper dual number
    pub eps1eps2: StaticMat<T, M, N>,
    f: PhantomData<F>,
}

pub type HyperDualVec32<const M: usize, const N: usize> = HyperDualVec<f32, f32, M, N>;
pub type HyperDualVec64<const M: usize, const N: usize> = HyperDualVec<f64, f64, M, N>;
pub type HyperDual<T, F> = HyperDualVec<T, F, 1, 1>;
pub type HyperDual32 = HyperDual<f32, f32>;
pub type HyperDual64 = HyperDual<f64, f64>;

impl<T, F, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    /// Create a new hyper dual number from its fields.
    #[inline]
    pub fn new(
        re: T,
        eps1: StaticVec<T, M>,
        eps2: StaticVec<T, N>,
        eps1eps2: StaticMat<T, M, N>,
    ) -> Self {
        Self {
            re,
            eps1,
            eps2,
            eps1eps2,
            f: PhantomData,
        }
    }
}

impl<T, F> HyperDual<T, F> {
    /// Create a new scalar hyper dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        Self::new(
            re,
            StaticVec::new_vec([eps1]),
            StaticVec::new_vec([eps2]),
            StaticMat::new([[eps1eps2]]),
        )
    }
}

impl<T: Copy + Zero + AddAssign, F, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    /// Create a new hyper dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDualVec::new(re, StaticVec::zero(), StaticVec::zero(), StaticMat::zero())
    }
}

impl<T: One, F, const N: usize> HyperDualVec<T, F, 1, N> {
    /// Derive a hyper dual number w.r.t. the first variable.
    #[inline]
    pub fn derive1(mut self) -> Self {
        self.eps1[0] = T::one();
        self
    }
}

impl<T: One, F, const M: usize> HyperDualVec<T, F, M, 1> {
    /// Derive a hyper dual number w.r.t. the 2nd variable.
    #[inline]
    pub fn derive2(mut self) -> Self {
        self.eps2[0] = T::one();
        self
    }
}

impl<T: One, F, const M: usize, const N: usize> StaticVec<HyperDualVec<T, F, M, N>, M> {
    /// Derive a vector of hyper dual numbers w.r.t. to the first set of variables.
    #[inline]
    pub fn derive1(mut self) -> Self {
        for i in 0..M {
            self[i].eps1[i] = T::one();
        }
        self
    }
}

impl<T: One, F, const M: usize, const N: usize> StaticVec<HyperDualVec<T, F, M, N>, N> {
    /// Derive a vector of hyper dual numbers w.r.t. to the second set of variables.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_dual::{HyperDualVec64, DualNum, StaticVec};
    /// let x = HyperDualVec64::<1, 2>::from_re(2.0).derive1();
    /// let v = StaticVec::new_vec([2.0, 3.0]).map(HyperDualVec64::<1, 2>::from_re).derive2();
    /// let n = (x.powi(2)*v[0].powi(2) + v[1].powi(2)).sqrt();
    /// assert_eq!(n.re, 5.0);
    /// assert_relative_eq!(n.eps1[0], 1.6);
    /// assert_relative_eq!(n.eps2[0], 1.6);
    /// assert_relative_eq!(n.eps2[1], 0.6);
    /// assert_relative_eq!(n.eps1eps2[(0,0)], 1.088);
    /// assert_relative_eq!(n.eps1eps2[(0,1)], -0.192);
    /// ```
    #[inline]
    pub fn derive2(mut self) -> Self {
        for i in 0..N {
            self[i].eps2[i] = T::one();
        }
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.eps1 * f1,
            self.eps2 * f1,
            self.eps1eps2 * f1 + self.eps1.transpose_matmul(&self.eps2) * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const M: usize, const N: usize>
    Mul<&'a HyperDualVec<T, F, M, N>> for &'b HyperDualVec<T, F, M, N>
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn mul(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        HyperDualVec::new(
            self.re * other.re,
            other.eps1 * self.re + self.eps1 * other.re,
            other.eps2 * self.re + self.eps2 * other.re,
            other.eps1eps2 * self.re
                + self.eps1.transpose_matmul(&other.eps2)
                + other.eps1.transpose_matmul(&self.eps2)
                + self.eps1eps2 * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const M: usize, const N: usize>
    Div<&'a HyperDualVec<T, F, M, N>> for &'b HyperDualVec<T, F, M, N>
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn div(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        HyperDualVec::new(
            self.re * inv,
            (self.eps1 * other.re - other.eps1 * self.re) * inv2,
            (self.eps2 * other.re - other.eps2 * self.re) * inv2,
            self.eps1eps2 * inv
                - (other.eps1eps2 * self.re
                    + self.eps1.transpose_matmul(&other.eps2)
                    + other.eps1.transpose_matmul(&self.eps2))
                    * inv2
                + other.eps1.transpose_matmul(&other.eps2)
                    * ((T::one() + T::one()) * self.re * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: fmt::Display, F: fmt::Display, const M: usize, const N: usize> fmt::Display
    for HyperDualVec<T, F, M, N>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε1ε2",
            self.re, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

impl_second_derivatives!(HyperDualVec, [M, N], [eps1, eps2, eps1eps2]);
impl_dual!(HyperDualVec, [M, N], [eps1, eps2, eps1eps2]);
