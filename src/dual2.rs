use crate::{Derivative, DualNum, DualNumFloat};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, OMatrix, OVector, SMatrix, SVector, U1};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A second order dual number for the calculation of Hessians.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Dual2Vec<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    /// Real part of the second order dual number
    pub re: T,
    /// Gradient part of the second order dual number
    pub v1: Derivative<T, F, U1, D>,
    /// Hessian part of the second order dual number
    pub v2: Derivative<T, F, D, D>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Copy, const N: usize> Copy for Dual2Vec<T, F, Const<N>> {}

pub type Dual2Vec32<D> = Dual2Vec<f32, f32, D>;
pub type Dual2Vec64<D> = Dual2Vec<f64, f64, D>;
pub type Dual2SVec32<const N: usize> = Dual2Vec<f32, f32, Const<N>>;
pub type Dual2SVec64<const N: usize> = Dual2Vec<f64, f64, Const<N>>;
pub type Dual2DVec32 = Dual2Vec<f32, f32, Dyn>;
pub type Dual2DVec64 = Dual2Vec<f64, f64, Dyn>;
pub type Dual2<T, F> = Dual2Vec<T, F, U1>;
pub type Dual2_32 = Dual2<f32, f32>;
pub type Dual2_64 = Dual2<f64, f64>;

impl<T: DualNum<F>, F, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: Derivative<T, F, U1, D>, v2: Derivative<T, F, D, D>) -> Self {
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
        Self::new(
            re,
            Derivative::some(SMatrix::from_element(v1)),
            Derivative::some(SMatrix::from_element(v2)),
        )
    }

    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual2, DualNum};
    /// let x = Dual2::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);             // x²
    /// assert_eq!(x.v1.unwrap(), 10.0);    // 2x
    /// assert_eq!(x.v2.unwrap(), 2.0);     // 2
    /// ```
    ///
    /// Can also be used for higher order derivatives.
    /// ```
    /// # use num_dual::{Dual64, Dual2, DualNum};
    /// let x = Dual2::from_re(Dual64::from_re(5.0).derivative())
    ///     .derivative()
    ///     .powi(2);
    /// assert_eq!(x.re.re, 25.0);                    // x²
    /// assert_eq!(x.re.eps.unwrap(), 10.0);          // 2x
    /// assert_eq!(x.v1.unwrap().re, 10.0);           // 2x
    /// assert_eq!(x.v1.unwrap().eps.unwrap(), 2.0);  // 2
    /// assert_eq!(x.v2.unwrap().re, 2.0);            // 2
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.v1 = Derivative::some(SVector::from_element(T::one()));
        self
    }
}

impl<T: DualNum<F>, F, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, Derivative::none(), Derivative::none())
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
/// let (f, df, d2f) = second_derivative(|x| x.powi(3), x);
/// assert_eq!(f.re(), 125.0);           // x³
/// assert_eq!(f.eps.unwrap(), 75.0);    // 3x²
/// assert_eq!(df.re, 75.0);             // 3x²
/// assert_eq!(df.eps.unwrap(), 30.0);   // 6x
/// assert_eq!(d2f.re, 30.0);            // 6x
/// assert_eq!(d2f.eps.unwrap(), 6.0);   // 6
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
    let x = Dual2::from_re(x).derivative();
    g(x).map(|r| (r.re, r.v1.unwrap(), r.v2.unwrap()))
}

/// Calculate the Hessian of a scalar function.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{hessian, DualNum, Dual2SVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<Dual2SVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g, h) = hessian(fun, v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// assert_relative_eq!(h[(0,0)], 0.072);
/// assert_relative_eq!(h[(0,1)], -0.096);
/// assert_relative_eq!(h[(1,0)], -0.096);
/// assert_relative_eq!(h[(1,1)], 0.128);
/// ```
pub fn hessian<G, T: DualNum<F>, F: DualNumFloat, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> (T, OVector<T, D>, OMatrix<T, D, D>)
where
    G: FnOnce(OVector<Dual2Vec<T, F, D>, D>) -> Dual2Vec<T, F, D>,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, U1, D>
        + Allocator<T, D, D>
        + Allocator<Dual2Vec<T, F, D>, D>,
{
    try_hessian(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [hessian] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_hessian<G, T: DualNum<F>, F: DualNumFloat, E, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> Result<(T, OVector<T, D>, OMatrix<T, D, D>), E>
where
    G: FnOnce(OVector<Dual2Vec<T, F, D>, D>) -> Result<Dual2Vec<T, F, D>, E>,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, U1, D>
        + Allocator<T, D, D>
        + Allocator<Dual2Vec<T, F, D>, D>,
{
    let mut x = x.map(Dual2Vec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.v1 = Derivative::derivative(c, r, i)
    }
    g(x).map(|res| {
        (
            res.re,
            res.v1.unwrap_generic(c, r).transpose(),
            res.v2.unwrap_generic(r, r),
        )
    })
}

/* chain rule */
impl<T: DualNum<F>, F: Float, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            &self.v1 * f1.clone(),
            &self.v2 * f1 + self.v1.tr_mul(&self.v1) * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Mul<&'a Dual2Vec<T, F, D>> for &'b Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    type Output = Dual2Vec<T, F, D>;
    #[inline]
    fn mul(self, other: &Dual2Vec<T, F, D>) -> Dual2Vec<T, F, D> {
        Dual2Vec::new(
            self.re.clone() * other.re.clone(),
            &other.v1 * self.re.clone() + &self.v1 * other.re.clone(),
            &other.v2 * self.re.clone()
                + self.v1.tr_mul(&other.v1)
                + other.v1.tr_mul(&self.v1)
                + &self.v2 * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Div<&'a Dual2Vec<T, F, D>> for &'b Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    type Output = Dual2Vec<T, F, D>;
    #[inline]
    fn div(self, other: &Dual2Vec<T, F, D>) -> Dual2Vec<T, F, D> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * inv.clone();
        Dual2Vec::new(
            self.re.clone() * inv.clone(),
            (&self.v1 * other.re.clone() - &other.v1 * self.re.clone()) * inv2.clone(),
            &self.v2 * inv.clone()
                - (&other.v2 * self.re.clone()
                    + self.v1.tr_mul(&other.v1)
                    + other.v1.tr_mul(&self.v1))
                    * inv2.clone()
                + other.v1.tr_mul(&other.v1)
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display, D: Dim> fmt::Display for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<T, U1, D> + Allocator<T, D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.v1.fmt(f, "ε1")?;
        self.v2.fmt(f, "ε1²")
    }
}

impl_second_derivatives2!(Dual2Vec, [v1, v2], [D]);
impl_dual2!(Dual2Vec, [v1, v2], [D]);
