use crate::dual::{Dual32, Dual64};
use crate::dual_n::{DualN32, DualN64};
use crate::linalg::{Scale, StaticMat, StaticVec};
use crate::{DualNum, DualNumMethods};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A hyper dual number.
#[derive(PartialEq, Copy, Clone)]
pub struct HyperDualN<T, F, const N: usize> {
    /// Real part of the hyper dual number
    pub re: T,
    /// gradient
    pub gradient: StaticVec<T, N>,
    /// Hessian
    pub hessian: StaticMat<T, N, N>,
    f: PhantomData<F>,
}

pub type HyperDualN32<const N: usize> = HyperDualN<f32, f32, N>;
pub type HyperDualN64<const N: usize> = HyperDualN<f64, f64, N>;
pub type HyperDualNDual32<const N: usize> = HyperDualN<Dual32, f32, N>;
pub type HyperDualNDual64<const N: usize> = HyperDualN<Dual64, f64, N>;
pub type HyperDualNDualN32<const M: usize, const N: usize> = HyperDualN<DualN32<M>, f32, N>;
pub type HyperDualNDualN64<const M: usize, const N: usize> = HyperDualN<DualN64<M>, f64, N>;

impl<T, F, const N: usize> HyperDualN<T, F, N> {
    /// Create a new hyperdual number
    #[inline]
    pub fn new(re: T, gradient: StaticVec<T, N>, hessian: StaticMat<T, N, N>) -> Self {
        Self {
            re,
            gradient,
            hessian,
            f: PhantomData,
        }
    }
}

impl<T: Copy + Zero + AddAssign, F, const N: usize> HyperDualN<T, F, N> {
    /// Create a new hyperdual number from the real part
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDualN::new(re, StaticVec::zero(), StaticMat::zero())
    }
}

impl<T: One, F, const N: usize> HyperDualN<T, F, N> {
    #[inline]
    pub fn derive(mut self, i: usize) -> Self {
        self.gradient[i] = T::one();
        self
    }
}

impl<T: One, F, const N: usize> StaticVec<HyperDualN<T, F, N>, N> {
    #[inline]
    pub fn derive(mut self) -> Self {
        for i in 0..N {
            self[i].gradient[i] = T::one();
        }
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const N: usize> HyperDualN<T, F, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.gradient * f1,
            self.hessian * f1 + self.gradient.transpose_matmul(&self.gradient) * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Mul<&'a HyperDualN<T, F, N>>
    for &'b HyperDualN<T, F, N>
{
    type Output = HyperDualN<T, F, N>;
    #[inline]
    fn mul(self, other: &HyperDualN<T, F, N>) -> HyperDualN<T, F, N> {
        HyperDualN::new(
            self.re * other.re,
            other.gradient * self.re + self.gradient * other.re,
            other.hessian * self.re
                + self.gradient.transpose_matmul(&other.gradient)
                + other.gradient.transpose_matmul(&self.gradient)
                + self.hessian * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Div<&'a HyperDualN<T, F, N>>
    for &'b HyperDualN<T, F, N>
{
    type Output = HyperDualN<T, F, N>;
    #[inline]
    fn div(self, other: &HyperDualN<T, F, N>) -> HyperDualN<T, F, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        HyperDualN::new(
            self.re * inv,
            (self.gradient * other.re - other.gradient * self.re) * inv2,
            self.hessian * inv
                - (other.hessian * self.re
                    + self.gradient.transpose_matmul(&other.gradient)
                    + other.gradient.transpose_matmul(&self.gradient))
                    * inv2
                + other.gradient.transpose_matmul(&other.gradient)
                    * ((T::one() + T::one()) * self.re * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: fmt::Display, F: fmt::Display, const N: usize> fmt::Display for HyperDualN<T, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε1 + {}ε2", self.re, self.gradient, self.hessian)
    }
}

impl_second_derivatives!(HyperDualN, [N]);
impl_dual!(HyperDualN, [N], [gradient, hessian]);
