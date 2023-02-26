use crate::{DualNum, DualNumFloat};
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use nalgebra::*;
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number for the calculations of gradients or Jacobians.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct DualVec<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: Derivative<T, F, D, U1>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Copy, const N: usize> Copy for DualVec<T, F, Const<N>> {}

pub type DualVec32<D> = DualVec<f32, f32, D>;
pub type DualVec64<D> = DualVec<f64, f64, D>;
pub type DualSVec32<const N: usize> = DualVec<f32, f32, Const<N>>;
pub type DualSVec64<const N: usize> = DualVec<f64, f64, Const<N>>;
pub type DualDVec32 = DualVec<f32, f32, Dyn>;
pub type DualDVec64 = DualVec<f64, f64, Dyn>;
pub type Dual<T, F> = DualVec<T, F, U1>;
pub type Dual32 = Dual<f32, f32>;
pub type Dual64 = Dual<f64, f64>;

impl<T: DualNum<F>, F, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: Derivative<T, F, D, U1>) -> Self {
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
        Self::new(re, Derivative::some(SVector::from_element(eps)))
    }
}

impl<T: DualNum<F> + Zero, F, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, Derivative::none())
    }
}

impl<T: DualNum<F> + One, F> Dual<T, F> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps.unwrap(), 10.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.eps = Derivative::some(SVector::from_element(T::one()));
        self
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
    try_first_derivative(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [first_derivative] for fallible functions.
pub fn try_first_derivative<G, T: DualNum<F>, F, E>(g: G, x: T) -> Result<(T, T), E>
where
    G: FnOnce(Dual<T, F>) -> Result<Dual<T, F>, E>,
{
    let x = Dual::from_re(x).derivative();
    g(x).map(|r| (r.re, r.eps.unwrap()))
}

/// Calculate the gradient of a scalar function
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualSVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<DualSVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g) = gradient(fun, v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// ```
///
/// The variable vector can also be dynamically sized
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualDVec64};
/// # use nalgebra::DVector;
/// let v = DVector::repeat(4, 2.0);
/// let fun = |v: DVector<DualDVec64>| v.iter().map(|v| v * v).sum::<DualDVec64>().sqrt();
/// let (f, g) = gradient(fun, v);
/// assert_eq!(f, 4.0);
/// assert_relative_eq!(g[0], 0.5);
/// assert_relative_eq!(g[1], 0.5);
/// assert_relative_eq!(g[2], 0.5);
/// assert_relative_eq!(g[3], 0.5);
/// ```
pub fn gradient<G, T: DualNum<F>, F: DualNumFloat, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> (T, OVector<T, D>)
where
    G: FnOnce(OVector<DualVec<T, F, D>, D>) -> DualVec<T, F, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<DualVec<T, F, D>, D>,
{
    try_gradient(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [gradient] for fallible functions.
pub fn try_gradient<G, T: DualNum<F>, F: DualNumFloat, E, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> Result<(T, OVector<T, D>), E>
where
    G: FnOnce(OVector<DualVec<T, F, D>, D>) -> Result<DualVec<T, F, D>, E>,
    DefaultAllocator: Allocator<T, D> + Allocator<DualVec<T, F, D>, D>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative(r, c, i);
    }
    g(x).map(|res| (res.re, res.eps.unwrap_generic(r, c)))
}

/// Calculate the Jacobian of a vector function.
/// ```
/// # use num_dual::{jacobian, DualSVec64, DualNum};
/// # use nalgebra::SVector;
/// let xy = SVector::from([5.0, 3.0, 2.0]);
/// let fun = |xy: SVector<DualSVec64<3>, 3>| SVector::from([
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
pub fn jacobian<G, T: DualNum<F>, F: DualNumFloat, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, N>,
) -> (OVector<T, M>, OMatrix<T, M, N>)
where
    G: FnOnce(OVector<DualVec<T, F, N>, N>) -> OVector<DualVec<T, F, N>, M>,
    DefaultAllocator: Allocator<DualVec<T, F, N>, M>
        + Allocator<T, M>
        + Allocator<T, N>
        + Allocator<T, M, N>
        + Allocator<T, nalgebra::Const<1>, N>
        + Allocator<DualVec<T, F, N>, N>
        + Allocator<OMatrix<T, U1, N>, M>,
{
    try_jacobian(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [jacobian] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_jacobian<G, T: DualNum<F>, F: DualNumFloat, E, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, N>,
) -> Result<(OVector<T, M>, OMatrix<T, M, N>), E>
where
    G: FnOnce(OVector<DualVec<T, F, N>, N>) -> Result<OVector<DualVec<T, F, N>, M>, E>,
    DefaultAllocator: Allocator<DualVec<T, F, N>, M>
        + Allocator<T, M>
        + Allocator<T, N>
        + Allocator<T, M, N>
        + Allocator<T, nalgebra::Const<1>, N>
        + Allocator<DualVec<T, F, N>, N>
        + Allocator<OMatrix<T, U1, N>, M>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative(r, c, i);
    }
    g(x).map(|res| {
        let eps = OMatrix::from_rows(
            res.map(|res| res.eps.unwrap_generic(r, c).transpose())
                .as_slice(),
        );
        (res.map(|r| r.re), eps)
    })
}

/* chain rule */
impl<T: DualNum<F>, F: Float, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, &self.eps * f1)
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Mul<&'a DualVec<T, F, D>> for &'b DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    type Output = DualVec<T, F, D>;
    #[inline]
    fn mul(self, other: &DualVec<T, F, D>) -> Self::Output {
        DualVec::new(
            self.re.clone() * other.re.clone(),
            &self.eps * other.re.clone() + &other.eps * self.re.clone(),
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Div<&'a DualVec<T, F, D>> for &'b DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    type Output = DualVec<T, F, D>;
    #[inline]
    fn div(self, other: &DualVec<T, F, D>) -> DualVec<T, F, D> {
        let inv = other.re.recip();
        DualVec::new(
            self.re.clone() * inv.clone(),
            (&self.eps * other.re.clone() - &other.eps * self.re.clone()) * inv.clone() * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F, D: Dim> fmt::Display for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.eps.fmt(f, "ε")
    }
}

impl_first_derivatives2!(DualVec, [eps], [D]);
impl_dual2!(DualVec, [eps], [D]);

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Derivative<T: DualNum<F>, F, R: Dim, C: Dim>(Option<OMatrix<T, R, C>>, PhantomData<F>)
where
    DefaultAllocator: Allocator<T, R, C>;

impl<T: DualNum<F> + Copy, F: Copy, const R: usize, const C: usize> Copy
    for Derivative<T, F, Const<R>, Const<C>>
{
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    pub fn new(derivative: Option<OMatrix<T, R, C>>) -> Self {
        Self(derivative, PhantomData)
    }

    pub fn some(derivative: OMatrix<T, R, C>) -> Self {
        Self::new(Some(derivative))
    }

    pub fn none() -> Self {
        Self::new(None)
    }

    #[allow(clippy::self_named_constructors)]
    pub fn derivative(r: R, c: C, i: usize) -> Self {
        let mut m = OMatrix::zeros_generic(r, c);
        m[i] = T::one();
        Self::some(m)
    }

    pub fn unwrap_generic(self, r: R, c: C) -> OMatrix<T, R, C> {
        self.0.unwrap_or_else(|| OMatrix::zeros_generic(r, c))
    }

    pub fn fmt(&self, f: &mut fmt::Formatter, symbol: &str) -> fmt::Result {
        if let Some(m) = self.0.as_ref() {
            write!(f, " + ")?;
            match m.shape() {
                (1, 1) => write!(f, "{}", m[0])?,
                (1, _) | (_, 1) => {
                    let x: Vec<_> = m.iter().map(T::to_string).collect();
                    write!(f, "[{}]", x.join(","))?
                }
                (_, _) => write!(f, "{}", m)?,
            };
            write!(f, "{symbol}")?;
        }
        write!(f, "")
    }
}

impl<T: DualNum<F>, F> Derivative<T, F, U1, U1> {
    pub fn unwrap(self) -> T {
        self.0.map_or_else(
            || T::zero(),
            |s| {
                let [[r]] = s.data.0;
                r
            },
        )
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Mul<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.map(|x| x * rhs))
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Mul<T> for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| x * rhs))
    }
}

impl<'a, 'b, T: DualNum<F>, F, R: Dim, C: Dim, R2: Dim, C2: Dim> Mul<&'b Derivative<T, F, R2, C2>>
    for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, R2, C2> + Allocator<T, R, C2>,
    ShapeConstraint: SameNumberOfRows<C, R2>,
{
    type Output = Derivative<T, F, R, C2>;

    fn mul(self, rhs: &Derivative<T, F, R2, C2>) -> Derivative<T, F, R, C2> {
        Derivative::new(self.0.as_ref().zip(rhs.0.as_ref()).map(|(s, r)| s * r))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    pub fn tr_mul<R2: Dim, C2: Dim>(
        &self,
        rhs: &Derivative<T, F, R2, C2>,
    ) -> Derivative<T, F, C, C2>
    where
        DefaultAllocator: Allocator<T, R2, C2> + Allocator<T, C, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        Derivative::new(
            self.0
                .as_ref()
                .zip(rhs.0.as_ref())
                .map(|(s, r)| s.tr_mul(r)),
        )
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Add for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(match (self.0, rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Add<&'a Derivative<T, F, R, C>>
    for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn add(self, rhs: &Derivative<T, F, R, C>) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(r.clone()),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Add for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn add(self, rhs: Self) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(r.clone()),
            (None, None) => None,
        })
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Sub for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(match (self.0, rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s),
            (None, Some(r)) => Some(-r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Sub<&'a Derivative<T, F, R, C>>
    for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn sub(self, rhs: &Derivative<T, F, R, C>) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(-r.clone()),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Sub for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(-r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Neg for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Neg for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> AddAssign for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn add_assign(&mut self, rhs: Self) {
        match (&mut self.0, rhs.0) {
            (Some(s), Some(r)) => *s += &r,
            (None, Some(r)) => self.0 = Some(r),
            (_, None) => (),
        };
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> SubAssign for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn sub_assign(&mut self, rhs: Self) {
        match (&mut self.0, rhs.0) {
            (Some(s), Some(r)) => *s -= &r,
            (None, Some(r)) => self.0 = Some(-&r),
            (_, None) => (),
        };
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> MulAssign<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn mul_assign(&mut self, rhs: T) {
        match &mut self.0 {
            Some(s) => *s *= rhs,
            None => (),
        }
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> DivAssign<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn div_assign(&mut self, rhs: T) {
        match &mut self.0 {
            Some(s) => *s /= rhs,
            None => (),
        }
    }
}
