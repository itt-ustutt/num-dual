use crate::dual::{Dual32, Dual64};
use crate::dual_n::{DualN32, DualN64};
use crate::{DualNum, DualNumMethods, StaticMat, StaticVec};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A vector hyper dual number for the calculation of Hessians.
#[derive(PartialEq, Copy, Clone)]
pub struct HyperDualN<T, F, const N: usize> {
    /// Real part of the hyper dual number
    pub re: T,
    /// Gradient part of the hyper dual number
    pub gradient: StaticVec<T, N>,
    /// Hessian part of the hyper dual number
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
    /// Create a new hyperdual number from its fields.
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
    /// Create a new hyperdual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDualN::new(re, StaticVec::zero(), StaticMat::zero())
    }
}

impl<T: One, F, const N: usize> HyperDualN<T, F, N> {
    /// Derive a dual number w.r.t. the i-th variable.
    /// ```
    /// # use num_hyperdual::{HyperDualN64, DualNumMethods};
    /// let x = HyperDualN64::<2>::from_re(5.0).derive(0);
    /// let y = HyperDualN64::<2>::from_re(3.0).derive(1);
    /// let z = x * y.powi(2);
    /// assert_eq!(z.re, 45.0);                 // xy²
    /// assert_eq!(z.gradient[0], 9.0);         // y²
    /// assert_eq!(z.gradient[1], 30.0);        // 2xy
    /// assert_eq!(z.hessian[(0,0)], 0.0);      // 0
    /// assert_eq!(z.hessian[(0,1)], 6.0);      // 2y
    /// assert_eq!(z.hessian[(1,0)], 6.0);      // 2y
    /// assert_eq!(z.hessian[(1,1)], 10.0);     // 2x
    /// ```
    #[inline]
    pub fn derive(mut self, i: usize) -> Self {
        self.gradient[i] = T::one();
        self
    }
}

impl<T: One, F, const N: usize> StaticVec<HyperDualN<T, F, N>, N> {
    /// Derive a Vector of hyper dual numbers.
    /// ```
    /// # use approx::assert_relative_eq;
    /// # use num_hyperdual::{HyperDualN64, DualNumMethods, StaticVec};
    /// let v = StaticVec::new_vec([4.0, 3.0]).map(HyperDualN64::<2>::from_re).derive();
    /// let n = (v[0].powi(2) + v[1].powi(2)).sqrt();
    /// assert_eq!(n.re, 5.0);
    /// assert_relative_eq!(n.gradient[0], 0.8);
    /// assert_relative_eq!(n.gradient[1], 0.6);
    /// assert_relative_eq!(n.hessian[(0,0)], 0.072);
    /// assert_relative_eq!(n.hessian[(0,1)], -0.096);
    /// assert_relative_eq!(n.hessian[(1,0)], -0.096);
    /// assert_relative_eq!(n.hessian[(1,1)], 0.128);
    /// ```
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

impl_second_derivatives!(HyperDualN, [N], [gradient, hessian]);
impl_dual!(HyperDualN, [N], [gradient, hessian]);
