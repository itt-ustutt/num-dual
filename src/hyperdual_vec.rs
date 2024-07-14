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

/// A vector hyper-dual number for the calculation of partial Hessians.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct HyperDualVec<T: DualNum<F>, F, M: Dim, N: Dim>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    /// Real part of the hyper-dual number
    pub re: T,
    /// Gradient part of the hyper-dual number
    pub eps1: Derivative<T, F, M, U1>,
    /// Gradient part of the hyper-dual number
    pub eps2: Derivative<T, F, U1, N>,
    /// Partial Hessian part of the hyper-dual number
    pub eps1eps2: Derivative<T, F, M, N>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Copy, const M: usize, const N: usize> Copy
    for HyperDualVec<T, F, Const<M>, Const<N>>
{
}

pub type HyperDualVec32<M, N> = HyperDualVec<f32, f32, M, N>;
pub type HyperDualVec64<M, N> = HyperDualVec<f64, f64, M, N>;
pub type HyperDualSVec32<const M: usize, const N: usize> =
    HyperDualVec<f32, f32, Const<M>, Const<N>>;
pub type HyperDualSVec64<const M: usize, const N: usize> =
    HyperDualVec<f64, f64, Const<M>, Const<N>>;
pub type HyperDualDVec32 = HyperDualVec<f32, f32, Dyn, Dyn>;
pub type HyperDualDVec64 = HyperDualVec<f64, f64, Dyn, Dyn>;

impl<T: DualNum<F>, F, M: Dim, N: Dim> HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    /// Create a new hyper-dual number from its fields.
    #[inline]
    pub fn new(
        re: T,
        eps1: Derivative<T, F, M, U1>,
        eps2: Derivative<T, F, U1, N>,
        eps1eps2: Derivative<T, F, M, N>,
    ) -> Self {
        Self {
            re,
            eps1,
            eps2,
            eps1eps2,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F, M: Dim, N: Dim> HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    /// Create a new hyper-dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(
            re,
            Derivative::none(),
            Derivative::none(),
            Derivative::none(),
        )
    }
}

/// Calculate second partial derivatives with respect to vectors.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{partial_hessian, DualNum, HyperDualSVec64};
/// # use nalgebra::SVector;
/// let x = SVector::from([4.0, 3.0]);
/// let y = SVector::from([5.0]);
/// let fun = |x: SVector<HyperDualSVec64<2, 1>, 2>, y: SVector<HyperDualSVec64<2, 1>, 1>|
///                 y[0] / (x[0].powi(2) + x[1].powi(2)).sqrt();
/// let (f, dfdx, dfdy, d2fdxdy) = partial_hessian(fun, x, y);
/// assert_eq!(f, 1.0);
/// assert_relative_eq!(dfdx[0], -0.16);
/// assert_relative_eq!(dfdx[1], -0.12);
/// assert_relative_eq!(dfdy[0], 0.2);
/// assert_relative_eq!(d2fdxdy[0], -0.032);
/// assert_relative_eq!(d2fdxdy[1], -0.024);
/// ```
#[allow(clippy::type_complexity)]
pub fn partial_hessian<G, T: DualNum<F>, F: DualNumFloat, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, M>,
    y: OVector<T, N>,
) -> (T, OVector<T, M>, OVector<T, N>, OMatrix<T, M, N>)
where
    G: FnOnce(
        OVector<HyperDualVec<T, F, M, N>, M>,
        OVector<HyperDualVec<T, F, M, N>, N>,
    ) -> HyperDualVec<T, F, M, N>,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    try_partial_hessian(|x, y| Ok::<_, Infallible>(g(x, y)), x, y).unwrap()
}

/// Variant of [partial_hessian] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_partial_hessian<G, T: DualNum<F>, F: DualNumFloat, E, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, M>,
    y: OVector<T, N>,
) -> Result<(T, OVector<T, M>, OVector<T, N>, OMatrix<T, M, N>), E>
where
    G: FnOnce(
        OVector<HyperDualVec<T, F, M, N>, M>,
        OVector<HyperDualVec<T, F, M, N>, N>,
    ) -> Result<HyperDualVec<T, F, M, N>, E>,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    let mut x = x.map(HyperDualVec::from_re);
    let mut y = y.map(HyperDualVec::from_re);
    let (m, _) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps1 = Derivative::derivative_generic(m, U1, i)
    }
    let (n, _) = y.shape_generic();
    for (i, yi) in y.iter_mut().enumerate() {
        yi.eps2 = Derivative::derivative_generic(U1, n, i)
    }
    g(x, y).map(|r| {
        (
            r.re,
            r.eps1.unwrap_generic(m, U1),
            r.eps2.unwrap_generic(U1, n).transpose(),
            r.eps1eps2.unwrap_generic(m, n),
        )
    })
}

/* chain rule */
impl<T: DualNum<F>, F: Float, M: Dim, N: Dim> HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            &self.eps1 * f1.clone(),
            &self.eps2 * f1.clone(),
            &self.eps1eps2 * f1 + &self.eps1 * &self.eps2 * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, M: Dim, N: Dim> Mul<&'a HyperDualVec<T, F, M, N>>
    for &'b HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn mul(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        HyperDualVec::new(
            self.re.clone() * other.re.clone(),
            &other.eps1 * self.re.clone() + &self.eps1 * other.re.clone(),
            &other.eps2 * self.re.clone() + &self.eps2 * other.re.clone(),
            &other.eps1eps2 * self.re.clone()
                + &self.eps1 * &other.eps2
                + &other.eps1 * &self.eps2
                + &self.eps1eps2 * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, M: Dim, N: Dim> Div<&'a HyperDualVec<T, F, M, N>>
    for &'b HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn div(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * &inv;
        HyperDualVec::new(
            self.re.clone() * &inv,
            (&self.eps1 * other.re.clone() - &other.eps1 * self.re.clone()) * inv2.clone(),
            (&self.eps2 * other.re.clone() - &other.eps2 * self.re.clone()) * inv2.clone(),
            &self.eps1eps2 * inv.clone()
                - (&other.eps1eps2 * self.re.clone()
                    + &self.eps1 * &other.eps2
                    + &other.eps1 * &self.eps2)
                    * inv2.clone()
                + &other.eps1
                    * &other.eps2
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display, M: Dim, N: Dim> fmt::Display for HyperDualVec<T, F, M, N>
where
    DefaultAllocator: Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.eps1.fmt(f, "ε1")?;
        self.eps2.fmt(f, "ε2")?;
        self.eps1eps2.fmt(f, "ε1ε2")
    }
}

impl_second_derivatives!(HyperDualVec, [eps1, eps2, eps1eps2], [M, N]);
impl_dual!(HyperDualVec, [eps1, eps2, eps1eps2], [M, N]);
