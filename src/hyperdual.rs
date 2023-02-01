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

/// A hyper dual number for the calculation of second partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct HyperDualVec<T: DualNum<F>, F, const M: usize, const N: usize> {
    /// Real part of the hyper dual number
    pub re: T,
    /// Partial derivative part of the hyper dual number
    pub eps1: SVector<T, M>,
    /// Partial derivative part of the hyper dual number
    pub eps2: RowSVector<T, N>,
    /// Second partial derivative part of the hyper dual number
    pub eps1eps2: SMatrix<T, M, N>,
    f: PhantomData<F>,
}

pub type HyperDualVec32<const M: usize, const N: usize> = HyperDualVec<f32, f32, M, N>;
pub type HyperDualVec64<const M: usize, const N: usize> = HyperDualVec<f64, f64, M, N>;
pub type HyperDual<T, F> = HyperDualVec<T, F, 1, 1>;
pub type HyperDual32 = HyperDual<f32, f32>;
pub type HyperDual64 = HyperDual<f64, f64>;

impl<T: DualNum<F>, F, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    /// Create a new hyper dual number from its fields.
    #[inline]
    pub fn new(
        re: T,
        eps1: SVector<T, M>,
        eps2: RowSVector<T, N>,
        eps1eps2: SMatrix<T, M, N>,
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

impl<T: DualNum<F>, F> HyperDual<T, F> {
    /// Create a new scalar hyper dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        Self::new(
            re,
            SVector::from([eps1]),
            SVector::from([eps2]),
            SMatrix::from([[eps1eps2]]),
        )
    }
}

impl<T: DualNum<F>, F, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    /// Create a new hyper dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        HyperDualVec::new(re, SVector::zero(), RowSVector::zero(), SMatrix::zero())
    }
}

impl<T: DualNum<F>, F, const N: usize> HyperDualVec<T, F, 1, N> {
    /// Derive a hyper dual number w.r.t. the first variable.
    #[inline]
    pub fn derive1(mut self) -> Self {
        self.eps1[0] = T::one();
        self
    }
}

impl<T: DualNum<F>, F, const M: usize> HyperDualVec<T, F, M, 1> {
    /// Derive a hyper dual number w.r.t. the 2nd variable.
    #[inline]
    pub fn derive2(mut self) -> Self {
        self.eps2[0] = T::one();
        self
    }
}

/// Calculate second partial derivatives with repsect to scalars.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{second_partial_derivative, DualNum, HyperDual64};
/// # use nalgebra::SVector;
/// let fun = |x: HyperDual64, y: HyperDual64| (x.powi(2) + y.powi(2)).sqrt();
/// let (f, dfdx, dfdy, d2fdxdy) = second_partial_derivative(fun, 4.0, 3.0);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(dfdx, 0.8);
/// assert_relative_eq!(dfdy, 0.6);
/// assert_relative_eq!(d2fdxdy, -0.096);
/// ```
pub fn second_partial_derivative<G, T: DualNum<F>, F>(g: G, x: T, y: T) -> (T, T, T, T)
where
    G: FnOnce(HyperDual<T, F>, HyperDual<T, F>) -> HyperDual<T, F>,
{
    try_second_partial_derivative(|x, y| Ok::<_, Infallible>(g(x, y)), x, y).unwrap()
}

/// Variant of [second_partial_derivative] for fallible functions.
pub fn try_second_partial_derivative<G, T: DualNum<F>, F, E>(
    g: G,
    x: T,
    y: T,
) -> Result<(T, T, T, T), E>
where
    G: FnOnce(HyperDual<T, F>, HyperDual<T, F>) -> Result<HyperDual<T, F>, E>,
{
    let mut x = HyperDual::from_re(x);
    let mut y = HyperDual::from_re(y);
    x.eps1[0] = T::one();
    y.eps2[0] = T::one();
    g(x, y).map(|r| (r.re, r.eps1[0], r.eps2[0], r.eps1eps2[0]))
}

/// Calculate second partial derivatives with repsect to vectors.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{partial_hessian, DualNum, HyperDualVec64};
/// # use nalgebra::SVector;
/// let x = SVector::from([4.0, 3.0]);
/// let y = SVector::from([5.0]);
/// let fun = |x: SVector<HyperDualVec64<2, 1>, 2>, y: SVector<HyperDualVec64<2, 1>, 1>|
///                 y[0] / (x[0].powi(2) + x[1].powi(2)).sqrt();
/// let (f, dfdx, dfdy, d2fdxdy) = partial_hessian(fun, x, y);
/// assert_eq!(f, 1.0);
/// assert_relative_eq!(dfdx[0], -0.16);
/// assert_relative_eq!(dfdx[1], -0.12);
/// assert_relative_eq!(dfdy[0], 0.2);
/// assert_relative_eq!(d2fdxdy[0], -0.032);
/// assert_relative_eq!(d2fdxdy[1], -0.024);
/// ```
pub fn partial_hessian<G, T: DualNum<F>, F: DualNumFloat, const M: usize, const N: usize>(
    g: G,
    x: SVector<T, M>,
    y: SVector<T, N>,
) -> (T, SVector<T, M>, SVector<T, N>, SMatrix<T, M, N>)
where
    G: FnOnce(
        SVector<HyperDualVec<T, F, M, N>, M>,
        SVector<HyperDualVec<T, F, M, N>, N>,
    ) -> HyperDualVec<T, F, M, N>,
{
    try_partial_hessian(|x, y| Ok::<_, Infallible>(g(x, y)), x, y).unwrap()
}

/// Variant of [partial_hessian] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_partial_hessian<G, T: DualNum<F>, F: DualNumFloat, E, const M: usize, const N: usize>(
    g: G,
    x: SVector<T, M>,
    y: SVector<T, N>,
) -> Result<(T, SVector<T, M>, SVector<T, N>, SMatrix<T, M, N>), E>
where
    G: FnOnce(
        SVector<HyperDualVec<T, F, M, N>, M>,
        SVector<HyperDualVec<T, F, M, N>, N>,
    ) -> Result<HyperDualVec<T, F, M, N>, E>,
{
    let mut x = x.map(HyperDualVec::from_re);
    let mut y = y.map(HyperDualVec::from_re);
    for i in 0..M {
        x[i].eps1[i] = T::one();
    }
    for j in 0..N {
        y[j].eps2[j] = T::one();
    }
    g(x, y).map(|r| (r.re, r.eps1, r.eps2.transpose(), r.eps1eps2))
}

/* chain rule */
impl<T: DualNum<F>, F: Float, const M: usize, const N: usize> HyperDualVec<T, F, M, N> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.eps1 * f1,
            self.eps2 * f1,
            self.eps1eps2 * f1 + self.eps1 * self.eps2 * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const M: usize, const N: usize>
    Mul<&'a HyperDualVec<T, F, M, N>> for &'b HyperDualVec<T, F, M, N>
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn mul(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        HyperDualVec::new(
            self.re * other.re,
            other.eps1 * self.re + self.eps1 * other.re,
            other.eps2 * self.re + self.eps2 * other.re,
            other.eps1eps2 * self.re
                + self.eps1 * other.eps2
                + other.eps1 * self.eps2
                + self.eps1eps2 * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, const M: usize, const N: usize>
    Div<&'a HyperDualVec<T, F, M, N>> for &'b HyperDualVec<T, F, M, N>
{
    type Output = HyperDualVec<T, F, M, N>;
    #[inline]
    fn div(self, other: &HyperDualVec<T, F, M, N>) -> HyperDualVec<T, F, M, N> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        HyperDualVec::new(
            self.re * inv,
            (self.eps1 * other.re - other.eps1 * self.re) * inv2,
            (self.eps2 * other.re - other.eps2 * self.re) * inv2,
            self.eps1eps2 * inv
                - (other.eps1eps2 * self.re + self.eps1 * other.eps2 + other.eps1 * self.eps2)
                    * inv2
                + other.eps1 * other.eps2 * ((T::one() + T::one()) * self.re * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display, const M: usize, const N: usize> fmt::Display
    for HyperDualVec<T, F, M, N>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε1ε2",
            self.re, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

impl_second_derivatives!(HyperDualVec, [M, N], [eps1, eps2, eps1eps2]);
impl_dual!(HyperDualVec, [M, N], [eps1, eps2, eps1eps2]);
