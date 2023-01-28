use crate::{DualNum, DualNumFloat, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A second order dual number for the calculation of Hessians.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct Dual2Vec<T, F, const N: usize> {
    /// Real part of the second order dual number
    pub re: T,
    /// Gradient part of the second order dual number
    pub v1: StaticVec<T, N>,
    /// Hessian part of the second order dual number
    pub v2: StaticMat<T, N, N>,
    f: PhantomData<F>,
}

pub type Dual2Vec32<const N: usize> = Dual2Vec<f32, f32, N>;
pub type Dual2Vec64<const N: usize> = Dual2Vec<f64, f64, N>;
pub type Dual2<T, F> = Dual2Vec<T, F, 1>;
pub type Dual2_32 = Dual2<f32, f32>;
pub type Dual2_64 = Dual2<f64, f64>;

impl<T, F, const N: usize> Dual2Vec<T, F, N> {
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: StaticVec<T, N>, v2: StaticMat<T, N, N>) -> Self {
        Self {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
}

impl<T, F> Dual2<T, F> {
    /// Create a new scalar second order dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, v1: T, v2: T) -> Self {
        Self::new(re, StaticVec::new_vec([v1]), StaticMat::new([[v2]]))
    }
}

impl<T: Copy + Zero + AddAssign, F, const N: usize> Dual2Vec<T, F, N> {
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Dual2Vec::new(re, StaticVec::zero(), StaticMat::zero())
    }
}

impl<T: One, F> Dual2<T, F> {
    /// Derive a scalar second order dual number
    /// ```
    /// # use num_dual::{Dual2, DualNum};
    /// let x = Dual2::from_re(5.0).derive().powi(2);
    /// assert_eq!(x.re, 25.0);            // x²
    /// assert_eq!(x.v1[0], 10.0);         // 2x
    /// assert_eq!(x.v2[(0,0)], 2.0);      // 2
    /// ```
    ///
    /// The argument can also be a dual number
    /// ```
    /// # use num_dual::{Dual64, Dual2, DualNum};
    /// let x = Dual2::from_re(Dual64::from_re(5.0).derive())
    ///     .derive()
    ///     .powi(2);
    /// assert_eq!(x.re.re(), 25.0);      // x²
    /// assert_eq!(x.re.eps[0], 10.0);    // 2x
    /// assert_eq!(x.v1[0].re, 10.0);     // 2x
    /// assert_eq!(x.v1[0].eps[0], 2.0);  // 2
    /// assert_eq!(x.v2[(0,0)].re, 2.0);  // 2
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        self.v1[0] = T::one();
        self
    }
}

impl<T: One, F, const N: usize> StaticVec<Dual2Vec<T, F, N>, N> {
    /// Derive a vector of second order dual numbers.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_dual::{Dual2Vec64, DualNum, StaticVec};
    /// let v = StaticVec::new_vec([4.0, 3.0]).map(Dual2Vec64::<2>::from_re).derive();
    /// let n = (v[0].powi(2) + v[1].powi(2)).sqrt();
    /// assert_eq!(n.re, 5.0);
    /// assert_relative_eq!(n.v1[0], 0.8);
    /// assert_relative_eq!(n.v1[1], 0.6);
    /// assert_relative_eq!(n.v2[(0,0)], 0.072);
    /// assert_relative_eq!(n.v2[(0,1)], -0.096);
    /// assert_relative_eq!(n.v2[(1,0)], -0.096);
    /// assert_relative_eq!(n.v2[(1,1)], 0.128);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        for i in 0..N {
            self[i].v1[i] = T::one();
        }
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const N: usize> Dual2Vec<T, F, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.v1 * f1,
            self.v2 * f1 + self.v1.transpose_matmul(&self.v1) * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Mul<&'a Dual2Vec<T, F, N>>
    for &'b Dual2Vec<T, F, N>
{
    type Output = Dual2Vec<T, F, N>;
    #[inline]
    fn mul(self, other: &Dual2Vec<T, F, N>) -> Dual2Vec<T, F, N> {
        Dual2Vec::new(
            self.re * other.re,
            other.v1 * self.re + self.v1 * other.re,
            other.v2 * self.re
                + self.v1.transpose_matmul(&other.v1)
                + other.v1.transpose_matmul(&self.v1)
                + self.v2 * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Div<&'a Dual2Vec<T, F, N>>
    for &'b Dual2Vec<T, F, N>
{
    type Output = Dual2Vec<T, F, N>;
    #[inline]
    fn div(self, other: &Dual2Vec<T, F, N>) -> Dual2Vec<T, F, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        Dual2Vec::new(
            self.re * inv,
            (self.v1 * other.re - other.v1 * self.re) * inv2,
            self.v2 * inv
                - (other.v2 * self.re
                    + self.v1.transpose_matmul(&other.v1)
                    + other.v1.transpose_matmul(&self.v1))
                    * inv2
                + other.v1.transpose_matmul(&other.v1)
                    * ((T::one() + T::one()) * self.re * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: fmt::Display, F: fmt::Display, const N: usize> fmt::Display for Dual2Vec<T, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε1 + {}ε1²", self.re, self.v1, self.v2)
    }
}

impl_second_derivatives!(Dual2Vec, [N], [v1, v2]);
impl_dual!(Dual2Vec, [N], [v1, v2]);
