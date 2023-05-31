use crate::{Derivative, DualNum, DualNumFloat};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, Dyn, OMatrix, OVector, U1};
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
        xi.v1 = Derivative::derivative_generic(c, r, i)
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

impl_second_derivatives!(Dual2Vec, [v1, v2], [D]);
impl_dual!(Dual2Vec, [v1, v2], [D]);
