use crate::{DualNum, DualNumFloat};
use nalgebra::{RowSVector, SMatrix, SVector};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A second order dual number for the calculation of Hessians.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct Dual2Vec<T: DualNum<F>, F, const N: usize> {
    /// Real part of the second order dual number
    pub re: T,
    /// Gradient part of the second order dual number
    pub v1: RowSVector<T, N>,
    /// Hessian part of the second order dual number
    pub v2: SMatrix<T, N, N>,
    f: PhantomData<F>,
}

pub type Dual2Vec32<const N: usize> = Dual2Vec<f32, f32, N>;
pub type Dual2Vec64<const N: usize> = Dual2Vec<f64, f64, N>;
pub type Dual2<T, F> = Dual2Vec<T, F, 1>;
pub type Dual2_32 = Dual2<f32, f32>;
pub type Dual2_64 = Dual2<f64, f64>;

impl<T: DualNum<F>, F, const N: usize> Dual2Vec<T, F, N> {
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: RowSVector<T, N>, v2: SMatrix<T, N, N>) -> Self {
        Self {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual2<T, F> {
    /// Create a new scalar second order dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, v1: T, v2: T) -> Self {
        Self::new(re, RowSVector::from([v1]), SMatrix::from([[v2]]))
    }

    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual2, DualNum};
    /// let x = Dual2::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);            // x²
    /// assert_eq!(x.v1[0], 10.0);         // 2x
    /// assert_eq!(x.v2[(0,0)], 2.0);      // 2
    /// ```
    ///
    /// Can also be used for higher order derivatives.
    /// ```
    /// # use num_dual::{Dual64, Dual2, DualNum};
    /// let x = Dual2::from_re(Dual64::from_re(5.0).derivative())
    ///     .derivative()
    ///     .powi(2);
    /// assert_eq!(x.re.re, 25.0);        // x²
    /// assert_eq!(x.re.eps[0], 10.0);    // 2x
    /// assert_eq!(x.v1[0].re, 10.0);     // 2x
    /// assert_eq!(x.v1[0].eps[0], 2.0);  // 2
    /// assert_eq!(x.v2[(0,0)].re, 2.0);  // 2
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.v1[0] = T::one();
        self
    }
}

impl<T: DualNum<F>, F, const N: usize> Dual2Vec<T, F, N> {
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Dual2Vec::new(re, RowSVector::zero(), SMatrix::zero())
    }
}

/// Calculate the second derivative of a univariate function.
/// ```
/// # use num_dual::{second_derivative, DualNum};
/// let (f, df, d2f) = second_derivative(|x| x.powi(2), 5.0);
/// assert_eq!(f, 25.0);       // x²
/// assert_eq!(df, 10.0);      // 2x
/// assert_eq!(d2f, 2.0);      // 2
/// ```
///
/// The argument can also be a dual number.
/// ```
/// # use num_dual::{second_derivative, Dual2, Dual64, DualNum};
/// let x = Dual64::new_scalar(5.0, 1.0);
/// let (f, df, d2f) = second_derivative(|x| x.powi(2), x);
/// assert_eq!(f.re(), 25.0);      // x²
/// assert_eq!(f.eps[0], 10.0);    // 2x
/// assert_eq!(df.re, 10.0);       // 2x
/// assert_eq!(df.eps[0], 2.0);    // 2
/// assert_eq!(d2f.re, 2.0);       // 2
/// assert_eq!(d2f.eps[0], 0.0);   // 0
/// ```
pub fn second_derivative<G, T: DualNum<F>, F>(g: G, x: T) -> (T, T, T)
where
    G: FnOnce(Dual2<T, F>) -> Dual2<T, F>,
{
    try_second_derivative(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [second_derivative] for fallible functions.
pub fn try_second_derivative<G, T: DualNum<F>, F, E>(g: G, x: T) -> Result<(T, T, T), E>
where
    G: FnOnce(Dual2<T, F>) -> Result<Dual2<T, F>, E>,
{
    let mut x = Dual2::from_re(x);
    x.v1[0] = T::one();
    let Dual2 { re, v1, v2, f: _ } = g(x)?;
    Ok((re, v1[0], v2[0]))
}

/// Calculate the Hessian of a scalar function.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{hessian, DualNum, Dual2Vec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<Dual2Vec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g, h) = hessian(fun, v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// assert_relative_eq!(h[(0,0)], 0.072);
/// assert_relative_eq!(h[(0,1)], -0.096);
/// assert_relative_eq!(h[(1,0)], -0.096);
/// assert_relative_eq!(h[(1,1)], 0.128);
/// ```
pub fn hessian<G, T: DualNum<F>, F: DualNumFloat, const N: usize>(
    g: G,
    x: SVector<T, N>,
) -> (T, SVector<T, N>, SMatrix<T, N, N>)
where
    G: FnOnce(SVector<Dual2Vec<T, F, N>, N>) -> Dual2Vec<T, F, N>,
{
    try_hessian(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [hessian] for fallible functions.
pub fn try_hessian<G, T: DualNum<F>, F: DualNumFloat, E, const N: usize>(
    g: G,
    x: SVector<T, N>,
) -> Result<(T, SVector<T, N>, SMatrix<T, N, N>), E>
where
    G: FnOnce(SVector<Dual2Vec<T, F, N>, N>) -> Result<Dual2Vec<T, F, N>, E>,
{
    let mut x = x.map(Dual2Vec::from_re);
    for i in 0..N {
        x[i].v1[i] = T::one();
    }
    let Dual2Vec { re, v1, v2, f: _ } = g(x)?;
    Ok((re, v1.transpose(), v2))
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const N: usize> Dual2Vec<T, F, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.v1 * f1,
            self.v2 * f1 + self.v1.tr_mul(&self.v1) * f2,
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
                + self.v1.tr_mul(&other.v1)
                + other.v1.tr_mul(&self.v1)
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
                - (other.v2 * self.re + self.v1.tr_mul(&other.v1) + other.v1.tr_mul(&self.v1))
                    * inv2
                + other.v1.tr_mul(&other.v1) * ((T::one() + T::one()) * self.re * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display, const N: usize> fmt::Display for Dual2Vec<T, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε1 + {}ε1²", self.re, self.v1, self.v2)
    }
}

impl_second_derivatives!(Dual2Vec, [N], [v1, v2]);
impl_dual!(Dual2Vec, [N], [v1, v2]);
