use crate::{DualNum, DualNumFloat};
use nalgebra::{SMatrix, SVector};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number for the calculations of gradients or Jacobians.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct DualVec<T: DualNum<F>, F, const N: usize> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: SVector<T, N>,
    f: PhantomData<F>,
}

pub type DualVec32<const N: usize> = DualVec<f32, f32, N>;
pub type DualVec64<const N: usize> = DualVec<f64, f64, N>;
pub type Dual<T, F> = DualVec<T, F, 1>;
pub type Dual32 = Dual<f32, f32>;
pub type Dual64 = Dual<f64, f64>;

impl<T: DualNum<F>, F, const N: usize> DualVec<T, F, N> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: SVector<T, N>) -> Self {
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual<T, F> {
    /// Create a new scalar dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, eps: T) -> Self {
        Self::new(re, SVector::from_element(eps))
    }
}

impl<T: DualNum<F>, F, const N: usize> DualVec<T, F, N> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, SVector::zero())
    }
}

/// Calculate the first derivative of a scalar function.
/// ```
/// # use num_dual::{first_derivative, DualNum};
/// let (f, df) = first_derivative(|x| x.powi(2), 5.0);
/// assert_eq!(f, 25.0);
/// assert_eq!(df, 10.0);
/// ```
pub fn first_derivative<G, T: DualNum<F>, F>(g: G, x: T) -> (T, T)
where
    G: FnOnce(Dual<T, F>) -> Dual<T, F>,
{
    let mut x = Dual::from_re(x);
    x.eps[0] = T::one();
    let Dual { re, eps, f: _ } = g(x);
    (re, eps[0])
}

/// Calculate the gradient of a scalar function
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<DualVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g) = gradient(fun, v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// ```
pub fn gradient<G, T: DualNum<F>, F: DualNumFloat, const N: usize>(
    g: G,
    x: SVector<T, N>,
) -> (T, SVector<T, N>)
where
    G: FnOnce(SVector<DualVec<T, F, N>, N>) -> DualVec<T, F, N>,
{
    let mut x = x.map(DualVec::from_re);
    for i in 0..N {
        x[i].eps[i] = T::one();
    }
    let DualVec { re, eps, f: _ } = g(x);
    (re, eps)
}

/// Calculate the Jacobian of a vector function.
/// ```
/// # use num_dual::{jacobian, DualVec64, DualNum};
/// # use nalgebra::SVector;
/// let xy = SVector::from([5.0, 3.0, 2.0]);
/// let fun = |xy: SVector<DualVec64<3>, 3>| SVector::from([
///                      xy[0] * xy[1].powi(3) * xy[2],
///                      xy[0].powi(2) * xy[1] * xy[2].powi(2)
///                     ]);
/// let (f, jac) = jacobian(fun, xy);
/// assert_eq!(f[0], 270.0);          // xy³z
/// assert_eq!(f[1], 300.0);          // x²yz²
/// assert_eq!(jac[(0,0)], 54.0);     // y³z
/// assert_eq!(jac[(0,1)], 270.0);    // 3xy²z
/// assert_eq!(jac[(0,2)], 135.0);    // xy³
/// assert_eq!(jac[(1,0)], 120.0);    // 2xyz²
/// assert_eq!(jac[(1,1)], 100.0);    // x²z²
/// assert_eq!(jac[(1,2)], 300.0);     // 2x²yz
/// ```
pub fn jacobian<G, T: DualNum<F>, F: DualNumFloat, const M: usize, const N: usize>(
    g: G,
    x: SVector<T, N>,
) -> (SVector<T, M>, SMatrix<T, M, N>)
where
    G: FnOnce(SVector<DualVec<T, F, N>, N>) -> SVector<DualVec<T, F, N>, M>,
{
    let mut x = x.map(DualVec::from_re);
    for i in 0..N {
        x[i].eps[i] = T::one();
    }
    let res = g(x);
    (
        SVector::from_fn(|i, _| res[i].re),
        SMatrix::from_fn(|i, j| res[i].eps[j]),
    )
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const N: usize> DualVec<T, F, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, self.eps * f1)
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const N: usize> Mul<&'a DualVec<T, F, N>>
    for &'b DualVec<T, F, N>
{
    type Output = DualVec<T, F, N>;
    #[inline]
    fn mul(self, other: &DualVec<T, F, N>) -> Self::Output {
        DualVec::new(
            self.re * other.re,
            self.eps * other.re + other.eps * self.re,
        )
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
        DualVec::new(
            self.re * inv,
            (self.eps * other.re - other.eps * self.re) * inv * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F, const N: usize> fmt::Display for DualVec<T, F, N> {
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
            DualVec64::new(1.0, SVector::from([2.5])),
            SVector::from([DualVec64::zero()]),
        );
        assert!(!d.is_derivative_zero());
        let d: DualVec64<1> = Dual::one();
        assert!(d.is_derivative_zero())
    }
}
