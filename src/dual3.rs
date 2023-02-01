use crate::{DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

/// A scalar third order dual number for the calculation of third derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct Dual3<T, F = T> {
    /// Real part of the third order dual number
    pub re: T,
    /// First derivative part of the third order dual number
    pub v1: T,
    /// Second derivative part of the third order dual number
    pub v2: T,
    /// Third derivative part of the third order dual number
    pub v3: T,
    f: PhantomData<F>,
}

pub type Dual3_32 = Dual3<f32>;
pub type Dual3_64 = Dual3<f64>;

impl<T, F> Dual3<T, F> {
    /// Create a new third order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            re,
            v1,
            v2,
            v3,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual3<T, F> {
    /// Create a new third order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero(), T::zero())
    }

    /// Create a new third order dual number from the real part
    /// with the first derivative part set to 1.
    #[inline]
    pub fn derivative(re: T) -> Self {
        Self::new(re, T::one(), T::zero(), T::zero())
    }
}

/// Calculate the third derivative of a univariate function.
/// ```
/// # use num_dual::{third_derivative, DualNum};
/// let (f, df, d2f, d3f) = third_derivative(|x| x.powi(3), 5.0);
/// assert_eq!(f, 125.0);      // x³
/// assert_eq!(df, 75.0);      // 3x²
/// assert_eq!(d2f, 30.0);     // 6x
/// assert_eq!(d3f, 6.0);      // 6
/// ```
pub fn third_derivative<G, T: DualNum<F>, F>(g: G, x: T) -> (T, T, T, T)
where
    G: FnOnce(Dual3<T, F>) -> Dual3<T, F>,
{
    try_third_derivative(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [third_derivative] for fallible functions.
pub fn try_third_derivative<G, T: DualNum<F>, F, E>(g: G, x: T) -> Result<(T, T, T, T), E>
where
    G: FnOnce(Dual3<T, F>) -> Result<Dual3<T, F>, E>,
{
    let mut x = Dual3::from_re(x);
    x.v1 = T::one();
    g(x).map(|r| (r.re, r.v1, r.v2, r.v3))
}

impl<T: DualNum<F>, F: Float> Dual3<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        let three = T::one() + T::one() + T::one();
        Self::new(
            f0,
            f1 * self.v1,
            f2 * self.v1 * self.v1 + f1 * self.v2,
            f3 * self.v1 * self.v1 * self.v1 + three * f2 * self.v1 * self.v2 + f1 * self.v3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a Dual3<T, F>> for &'b Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn mul(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
        let two = T::one() + T::one();
        let three = two + T::one();
        Dual3::new(
            self.re * rhs.re,
            self.v1 * rhs.re + self.re * rhs.v1,
            self.v2 * rhs.re + two * self.v1 * rhs.v1 + self.re * rhs.v2,
            self.v3 * rhs.re
                + three * self.v2 * rhs.v1
                + three * self.v1 * rhs.v2
                + self.re * rhs.v3,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a Dual3<T, F>> for &'b Dual3<T, F> {
    type Output = Dual3<T, F>;
    #[inline]
    fn div(self, rhs: &Dual3<T, F>) -> Dual3<T, F> {
        let rec = T::one() / rhs.re;
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for Dual3<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}v1 + {}v2 + {}v3",
            self.re, self.v1, self.v2, self.v3
        )
    }
}

impl_third_derivatives!(Dual3, [], [v1, v2, v3]);
impl_dual!(Dual3, [], [v1, v2, v3]);
