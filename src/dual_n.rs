use crate::linalg::Scale;
use crate::{DualNum, DualNumMethods, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A vector dual number for the calculations of gradients or Jacobians.
#[derive(PartialEq, Copy, Clone)]
pub struct DualN<T, F, const N: usize> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: StaticVec<T, N>,
    f: PhantomData<F>,
}

pub type DualN32<const N: usize> = DualN<f32, f32, N>;
pub type DualN64<const N: usize> = DualN<f64, f64, N>;

impl<T, F, const N: usize> DualN<T, F, N> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: StaticVec<T, N>) -> Self {
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: Copy + Zero + AddAssign, F, const N: usize> DualN<T, F, N> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, StaticVec::zero())
    }
}

impl<T: One, F, const N: usize> DualN<T, F, N> {
    /// Derive a dual number w.r.t. the i-th variable.
    /// ```
    /// # use num_hyperdual::{DualN64, DualNumMethods};
    /// let x = DualN64::<2>::from_re(5.0).derive(1).powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps[0], 0.0);
    /// assert_eq!(x.eps[1], 10.0);
    /// ```
    #[inline]
    pub fn derive(mut self, i: usize) -> Self {
        self.eps[i] = T::one();
        self
    }
}

impl<T: One, F, const N: usize> StaticVec<DualN<T, F, N>, N> {
    /// Derive a Vector of dual numbers.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_hyperdual::{DualN64, DualNumMethods, StaticVec};
    /// let v = StaticVec::new_vec([4.0, 3.0]).map(DualN64::<2>::from_re).derive();
    /// let n = (v[0].powi(2) + v[1].powi(2)).sqrt();
    /// assert_eq!(n.re, 5.0);
    /// assert_relative_eq!(n.eps[0], 0.8);
    /// assert_relative_eq!(n.eps[1], 0.6);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        for i in 0..N {
            self[i].eps[i] = T::one();
        }
        self
    }
}

impl<T: One + Zero + Copy + AddAssign, F, const M: usize, const N: usize>
    StaticVec<DualN<T, F, N>, M>
{
    /// Extract the Jacobian from a vector of DualN numbers.
    /// ```
    /// # use num_hyperdual::{DualN64, DualNumMethods, StaticVec};
    /// let x = DualN64::<2>::from(5.0).derive(0);
    /// let y = DualN64::<2>::from(3.0).derive(1);
    /// let j = StaticVec::new_vec([x * y.powi(3), x.powi(2) * y]).jacobian();
    /// assert_eq!(j[(0,0)], 27.0);     // y³
    /// assert_eq!(j[(0,1)], 135.0);    // 3xy²
    /// assert_eq!(j[(1,0)], 30.0);     // 2xy
    /// assert_eq!(j[(1,1)], 25.0);     // x²
    /// ```
    #[inline]
    pub fn jacobian(&self) -> StaticMat<T, M, N> {
        let mut res = StaticMat::zero();
        for i in 0..M {
            for j in 0..N {
                res[(i, j)] = self[i].eps[j];
            }
        }
        res
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
