use crate::{DualNum, DualNumFloat, IsDerivativeZero, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number for the calculations of gradients or Jacobians.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct DualVec<T, F, const N: usize> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: StaticVec<T, N>,
    f: PhantomData<F>,
}

pub type DualVec32<const N: usize> = DualVec<f32, f32, N>;
pub type DualVec64<const N: usize> = DualVec<f64, f64, N>;
pub type Dual<T, F> = DualVec<T, F, 1>;
pub type Dual32 = Dual<f32, f32>;
pub type Dual64 = Dual<f64, f64>;

impl<T, F, const N: usize> DualVec<T, F, N> {
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

impl<T, F> Dual<T, F> {
    /// Create a new scalar dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, eps: T) -> Self {
        Self::new(re, StaticVec::new_vec([eps]))
    }
}

impl<T: Copy + Zero + AddAssign, F, const N: usize> DualVec<T, F, N> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, StaticVec::zero())
    }
}

impl<T: One, F> Dual<T, F> {
    /// Derive a scalar dual number, i.e. set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derive().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps[0], 10.0);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        self.eps[0] = T::one();
        self
    }
}

impl<T: One, F, const N: usize> StaticVec<DualVec<T, F, N>, N> {
    /// Derive a vector of dual numbers.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_dual::{DualVec64, DualNum, StaticVec};
    /// let v = StaticVec::new_vec([4.0, 3.0]).map(DualVec64::<2>::from_re).derive();
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
    StaticVec<DualVec<T, F, N>, M>
{
    /// Extract the Jacobian from a vector of Dual numbers.
    /// ```
    /// # use num_dual::{DualVec64, DualNum, StaticVec};
    /// let xy = StaticVec::new_vec([5.0, 3.0]).map(DualVec64::<2>::from).derive();
    /// let j = StaticVec::new_vec([xy[0] * xy[1].powi(3), xy[0].powi(2) * xy[1]]).jacobian();
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
impl<T: DualNum<F>, F: Float, const N: usize> DualVec<T, F, N> {
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
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Mul<&'a DualVec<T, F, N>>
    for &'b DualVec<T, F, N>
{
    type Output = DualVec<T, F, N>;
    #[inline]
    fn mul(self, other: &DualVec<T, F, N>) -> Self::Output {
        let mut eps = [T::zero(); N];
        for i in 0..N {
            eps[i] = self.eps[i] * other.re + other.eps[i] * self.re;
        }
        DualVec::new(self.re * other.re, StaticVec::new_vec(eps))
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Div<&'a DualVec<T, F, N>>
    for &'b DualVec<T, F, N>
{
    type Output = DualVec<T, F, N>;
    #[inline]
    fn div(self, other: &DualVec<T, F, N>) -> DualVec<T, F, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        let mut eps = [T::zero(); N];
        for i in 0..N {
            eps[i] = (self.eps[i] * other.re - other.eps[i] * self.re) * inv2;
        }
        DualVec::new(self.re * inv, StaticVec::new_vec(eps))
    }
}

/* string conversions */
impl<T: fmt::Display, F, const N: usize> fmt::Display for DualVec<T, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl_first_derivatives!(DualVec, [N], [eps]);
impl_dual!(DualVec, [N], [eps]);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn is_derivative_zero() {
        let d = Dual::new(
            DualVec64::new(1.0, StaticMat::new([[2.5; 1]; 1])),
            StaticVec::new([[DualVec64::zero(); 1]; 1]),
        );
        assert!(!d.is_derivative_zero());
        let d: DualVec64<1> = Dual::one();
        assert!(d.is_derivative_zero())
    }
}
