use crate::dual::{Dual32, Dual64};
use crate::dual_n::{DualN32, DualN64};
use crate::{DualNum, DualNumFloat, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A vector hyper dual number for the calculation of Hessians.
#[derive(PartialEq, Copy, Clone)]
pub struct HyperDualMN<T, F, const M: usize, const N: usize> {
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

pub type HyperDualMN32<const M: usize, const N: usize> = HyperDualMN<f32, f32, M, N>;
pub type HyperDualMN64<const M: usize, const N: usize> = HyperDualMN<f64, f64, M, N>;
// pub type HyperDualMNDual32<const N: usize> = HyperDualMN<Dual32, f32, N>;
// pub type HyperDualMNDual64<const N: usize> = HyperDualMN<Dual64, f64, N>;
// pub type HyperDualMNDualN32<const M: usize, const N: usize> = HyperDualMN<DualN32<M>, f32, N>;
// pub type HyperDualMNDualN64<const M: usize, const N: usize> = HyperDualMN<DualN64<M>, f64, N>;

impl<T, F, const M: usize, const N: usize> HyperDualMN<T, F, M, N> {
    /// Create a new hyperdual number from its fields.
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

impl<T: Copy + Zero + AddAssign, F, const M: usize, const N: usize> HyperDualMN<T, F, M, N> {
    /// Create a new hyperdual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDualMN::new(re, StaticVec::zero(), StaticVec::zero(), StaticMat::zero())
    }
}

impl<T: One, F, const N: usize> HyperDualMN<T, F, 1, N> {
    /// Derive a dual number w.r.t. the first variable.
    #[inline]
    pub fn derive1(mut self) -> Self {
        self.eps1[0] = T::one();
        self
    }
}

impl<T: One, F, const M: usize> HyperDualMN<T, F, M, 1> {
    /// Derive a dual number w.r.t. the 2nd variable.
    #[inline]
    pub fn derive2(mut self) -> Self {
        self.eps2[0] = T::one();
        self
    }
}

impl<T: One, F, const M: usize, const N: usize> StaticVec<HyperDualMN<T, F, M, N>, M> {
    /// Derive a Vector of hyper dual numbers.
    #[inline]
    pub fn derive1(mut self) -> Self {
        for i in 0..M {
            self[i].eps1[i] = T::one();
        }
        self
    }
}

impl<T: One, F, const M: usize, const N: usize> StaticVec<HyperDualMN<T, F, M, N>, N> {
    /// Derive a Vector of hyper dual numbers.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_hyperdual::{HyperDualMN64, DualNum, StaticVec};
    /// let x = HyperDualMN64::<1, 2>::from_re(2.0).derive1();
    /// let v = StaticVec::new_vec([2.0, 3.0]).map(HyperDualMN64::<1, 2>::from_re).derive2();
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
impl<T: DualNum<F>, F: Float, const M: usize, const N: usize> HyperDualMN<T, F, M, N> {
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
    Mul<&'a HyperDualMN<T, F, M, N>> for &'b HyperDualMN<T, F, M, N>
{
    type Output = HyperDualMN<T, F, M, N>;
    #[inline]
    fn mul(self, other: &HyperDualMN<T, F, M, N>) -> HyperDualMN<T, F, M, N> {
        HyperDualMN::new(
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
    Div<&'a HyperDualMN<T, F, M, N>> for &'b HyperDualMN<T, F, M, N>
{
    type Output = HyperDualMN<T, F, M, N>;
    #[inline]
    fn div(self, other: &HyperDualMN<T, F, M, N>) -> HyperDualMN<T, F, M, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        HyperDualMN::new(
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
    for HyperDualMN<T, F, M, N>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε1ε2",
            self.re, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

impl_second_derivatives!(HyperDualMN, [M, N], [eps1, eps2, eps1eps2]);
impl_dual!(HyperDualMN, [M, N], [eps1, eps2, eps1eps2]);
