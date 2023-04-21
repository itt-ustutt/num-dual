use crate::{DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar hyper hyper dual number for the calculation of third partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct HyperHyperDual<T, F = T> {
    /// Real part of the hyper hyper dual number
    pub re: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps1: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps2: T,
    /// First partial derivative part of the hyper hyper dual number
    pub eps3: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps1eps2: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps1eps3: T,
    /// Second partial derivative part of the hyper hyper dual number
    pub eps2eps3: T,
    /// Third partial derivative part of the hyper hyper dual number
    pub eps1eps2eps3: T,
    f: PhantomData<F>,
}

pub type HyperHyperDual32 = HyperHyperDual<f32>;
pub type HyperHyperDual64 = HyperHyperDual<f64>;

impl<T: DualNum<F>, F> HyperHyperDual<T, F> {
    /// Create a new hyper hyper dual number from its fields.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        re: T,
        eps1: T,
        eps2: T,
        eps3: T,
        eps1eps2: T,
        eps1eps3: T,
        eps2eps3: T,
        eps1eps2eps3: T,
    ) -> Self {
        Self {
            re,
            eps1,
            eps2,
            eps3,
            eps1eps2,
            eps1eps3,
            eps2eps3,
            eps1eps2eps3,
            f: PhantomData,
        }
    }

    /// Set the partial derivative part w.r.t. the 1st variable to 1.
    #[inline]
    pub fn derivative1(mut self) -> Self {
        self.eps1 = T::one();
        self
    }

    /// Set the partial derivative part w.r.t. the 2nd variable to 1.
    #[inline]
    pub fn derivative2(mut self) -> Self {
        self.eps2 = T::one();
        self
    }

    /// Set the partial derivative part w.r.t. the 3rd variable to 1.
    #[inline]
    pub fn derivative3(mut self) -> Self {
        self.eps3 = T::one();
        self
    }

    /// Create a new hyper hyper dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(
            re,
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
        )
    }
}

/// Calculate third partial derivatives with respect to scalars.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{third_partial_derivative, DualNum, HyperHyperDual64};
/// # use nalgebra::SVector;
/// let fun = |x: HyperHyperDual64, y: HyperHyperDual64, z: HyperHyperDual64| (x.powi(2) + y.powi(2) + z.powi(2)).powi(3);
/// let (f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz) = third_partial_derivative(fun, 1.0, 2.0, 3.0);
/// println!("{:?}", third_partial_derivative(fun, 1.0, 2.0, 3.0));
/// assert_eq!(f, 2744.0);
/// assert_relative_eq!(dfdx, 1176.0);
/// assert_relative_eq!(dfdy, 2352.0);
/// assert_relative_eq!(dfdz, 3528.0);
/// assert_relative_eq!(d2fdxdy, 672.0);
/// assert_relative_eq!(d2fdxdz, 1008.0);
/// assert_relative_eq!(d2fdydz, 2016.0);
/// assert_relative_eq!(d3fdxdydz, 288.0);
/// ```
pub fn third_partial_derivative<G, T: DualNum<F>, F>(
    g: G,
    x: T,
    y: T,
    z: T,
) -> (T, T, T, T, T, T, T, T)
where
    G: FnOnce(
        HyperHyperDual<T, F>,
        HyperHyperDual<T, F>,
        HyperHyperDual<T, F>,
    ) -> HyperHyperDual<T, F>,
{
    try_third_partial_derivative(|x, y, z| Ok::<_, Infallible>(g(x, y, z)), x, y, z).unwrap()
}

/// Variant of [third_partial_derivative] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_third_partial_derivative<G, T: DualNum<F>, F, E>(
    g: G,
    x: T,
    y: T,
    z: T,
) -> Result<(T, T, T, T, T, T, T, T), E>
where
    G: FnOnce(
        HyperHyperDual<T, F>,
        HyperHyperDual<T, F>,
        HyperHyperDual<T, F>,
    ) -> Result<HyperHyperDual<T, F>, E>,
{
    let mut x = HyperHyperDual::from_re(x);
    let mut y = HyperHyperDual::from_re(y);
    let mut z = HyperHyperDual::from_re(z);
    x.eps1 = T::one();
    y.eps2 = T::one();
    z.eps3 = T::one();
    g(x, y, z).map(|r| {
        (
            r.re,
            r.eps1,
            r.eps2,
            r.eps3,
            r.eps1eps2,
            r.eps1eps3,
            r.eps2eps3,
            r.eps1eps2eps3,
        )
    })
}

/// Calculate the third partial derivative of a scalar function
/// with arbitrary many variables.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{third_partial_derivative_vec, DualNum, HyperHyperDual64};
/// # use nalgebra::SVector;
/// let fun = |x: &[HyperHyperDual64]| x[0].powi(3)*x[1].powi(2);
/// let (f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz) = third_partial_derivative_vec(fun, &[1.0, 2.0], 0, 0, 1);
/// # println!("{:?}", third_partial_derivative_vec(fun, &[1.0, 2.0, 3.0], 0, 0, 1));
/// assert_eq!(f, 4.0);
/// assert_relative_eq!(dfdx, 12.0);
/// assert_relative_eq!(dfdy, 12.0);
/// assert_relative_eq!(dfdz, 4.0);
/// assert_relative_eq!(d2fdxdy, 24.0);
/// assert_relative_eq!(d2fdxdz, 12.0);
/// assert_relative_eq!(d2fdydz, 12.0);
/// assert_relative_eq!(d3fdxdydz, 24.0);
/// ```
pub fn third_partial_derivative_vec<G, T: DualNum<F>, F>(
    g: G,
    x: &[T],
    i: usize,
    j: usize,
    k: usize,
) -> (T, T, T, T, T, T, T, T)
where
    G: FnOnce(&[HyperHyperDual<T, F>]) -> HyperHyperDual<T, F>,
{
    try_third_partial_derivative_vec(|x| Ok::<_, Infallible>(g(x)), x, i, j, k).unwrap()
}

/// Variant of [third_partial_derivative_vec] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_third_partial_derivative_vec<G, T: DualNum<F>, F, E>(
    g: G,
    x: &[T],
    i: usize,
    j: usize,
    k: usize,
) -> Result<(T, T, T, T, T, T, T, T), E>
where
    G: FnOnce(&[HyperHyperDual<T, F>]) -> Result<HyperHyperDual<T, F>, E>,
{
    let mut x: Vec<_> = x
        .iter()
        .map(|x| HyperHyperDual::from_re(x.clone()))
        .collect();
    x[i].eps1 = T::one();
    x[j].eps2 = T::one();
    x[k].eps3 = T::one();
    g(&x).map(|r| {
        (
            r.re,
            r.eps1,
            r.eps2,
            r.eps3,
            r.eps1eps2,
            r.eps1eps3,
            r.eps2eps3,
            r.eps1eps2eps3,
        )
    })
}

impl<T: DualNum<F>, F: Float> HyperHyperDual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        Self::new(
            f0,
            f1.clone() * &self.eps1,
            f1.clone() * &self.eps2,
            f1.clone() * &self.eps3,
            f1.clone() * &self.eps1eps2 + f2.clone() * &self.eps1 * &self.eps2,
            f1.clone() * &self.eps1eps3 + f2.clone() * &self.eps1 * &self.eps3,
            f1.clone() * &self.eps2eps3 + f2.clone() * &self.eps2 * &self.eps3,
            f1 * &self.eps1eps2eps3
                + f2 * (self.eps1.clone() * &self.eps2eps3
                    + self.eps2.clone() * &self.eps1eps3
                    + self.eps3.clone() * &self.eps1eps2)
                + f3 * self.eps1.clone() * &self.eps2 * &self.eps3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HyperHyperDual<T, F>> for &'b HyperHyperDual<T, F> {
    type Output = HyperHyperDual<T, F>;
    #[inline]
    fn mul(self, rhs: &HyperHyperDual<T, F>) -> HyperHyperDual<T, F> {
        HyperHyperDual::new(
            self.re.clone() * &rhs.re,
            self.eps1.clone() * &rhs.re + self.re.clone() * &rhs.eps1,
            self.eps2.clone() * &rhs.re + self.re.clone() * &rhs.eps2,
            self.eps3.clone() * &rhs.re + self.re.clone() * &rhs.eps3,
            self.eps1eps2.clone() * &rhs.re
                + self.eps1.clone() * &rhs.eps2
                + self.eps2.clone() * &rhs.eps1
                + self.re.clone() * &rhs.eps1eps2,
            self.eps1eps3.clone() * &rhs.re
                + self.eps1.clone() * &rhs.eps3
                + self.eps3.clone() * &rhs.eps1
                + self.re.clone() * &rhs.eps1eps3,
            self.eps2eps3.clone() * &rhs.re
                + self.eps2.clone() * &rhs.eps3
                + self.eps3.clone() * &rhs.eps2
                + self.re.clone() * &rhs.eps2eps3,
            self.eps1eps2eps3.clone() * &rhs.re
                + self.eps1.clone() * &rhs.eps2eps3
                + self.eps2.clone() * &rhs.eps1eps3
                + self.eps3.clone() * &rhs.eps1eps2
                + self.eps2eps3.clone() * &rhs.eps1
                + self.eps1eps3.clone() * &rhs.eps2
                + self.eps1eps2.clone() * &rhs.eps3
                + self.re.clone() * &rhs.eps1eps2eps3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HyperHyperDual<T, F>> for &'b HyperHyperDual<T, F> {
    type Output = HyperHyperDual<T, F>;
    #[inline]
    fn div(self, rhs: &HyperHyperDual<T, F>) -> HyperHyperDual<T, F> {
        let rec = T::one() / &rhs.re;
        let f0 = rec.clone();
        let f1 = -f0.clone() * &rec;
        let f2 = f1.clone() * &rec * F::from(-2.0).unwrap();
        let f3 = f2.clone() * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HyperHyperDual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε3 + {}ε1ε2 + {}ε1ε3 + {}ε2ε3 + {}ε1ε2ε3",
            self.re,
            self.eps1,
            self.eps2,
            self.eps3,
            self.eps1eps2,
            self.eps1eps3,
            self.eps2eps3,
            self.eps1eps2eps3
        )
    }
}

impl_third_derivatives!(
    HyperHyperDual,
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
impl_dual!(
    HyperHyperDual,
    [eps1, eps2, eps3, eps1eps2, eps1eps3, eps2eps3, eps1eps2eps3]
);
