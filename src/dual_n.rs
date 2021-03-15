use crate::linalg::{Scale, StaticVec};
use crate::{DualNum, DualNumMethods};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number.
#[derive(PartialEq, Copy, Clone)]
pub struct DualN<T, F, const N: usize> {
    /// Real part of the dual number
    pub re: T,
    /// Eps part
    pub eps: StaticVec<T, N>,
    f: PhantomData<F>,
}

pub type DualN32<const N: usize> = DualN<f32, f32, N>;
pub type DualN64<const N: usize> = DualN<f64, f64, N>;

impl<T, F, const N: usize> DualN<T, F, N> {
    /// Create a new dual number
    #[inline]
    pub fn new(re: T, eps: StaticVec<T, N>) -> Self {
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: Copy + Zero, F, const N: usize> DualN<T, F, N> {
    /// Create a new dual number from the real part
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, StaticVec::new_zero())
    }
}

impl<T: One, F, const N: usize> DualN<T, F, N> {
    #[inline]
    pub fn derive(mut self, i: usize) -> Self {
        self.eps[i] = T::one();
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const N: usize> DualN<T, F, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        let mut eps = [T::zero(); N];
        for i in 0..N {
            eps[i] = self.eps[i] * f1;
        }
        Self::new(f0, StaticVec::new_vec(eps))
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Mul<&'a DualN<T, F, N>>
    for &'b DualN<T, F, N>
{
    type Output = DualN<T, F, N>;
    #[inline]
    fn mul(self, other: &DualN<T, F, N>) -> Self::Output {
        let mut eps = [T::zero(); N];
        for i in 0..N {
            eps[i] = self.eps[i] * other.re + other.eps[i] * self.re;
        }
        DualN::new(self.re * other.re, StaticVec::new_vec(eps))
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Div<&'a DualN<T, F, N>>
    for &'b DualN<T, F, N>
{
    type Output = DualN<T, F, N>;
    #[inline]
    fn div(self, other: &DualN<T, F, N>) -> DualN<T, F, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        let mut eps = [T::zero(); N];
        for i in 0..N {
            eps[i] = (self.eps[i] * other.re - other.eps[i] * self.re) * inv2;
        }
        DualN::new(self.re * inv, StaticVec::new_vec(eps))
    }
}

/* string conversions */
impl<T: fmt::Display, F, const N: usize> fmt::Display for DualN<T, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl_first_derivatives!(DualN, [N]);
impl_dual!(DualN, [N], [eps]);
