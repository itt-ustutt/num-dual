use crate::{DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar second order dual number for the calculation of second derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual2<T: DualNum<F>, F> {
    /// Real part of the second order dual number
    pub re: T,
    /// First derivative part of the second order dual number
    pub v1: T,
    /// Second derivative part of the second order dual number
    pub v2: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    f: PhantomData<F>,
}

pub type Dual2_32 = Dual2<f32, f32>;
pub type Dual2_64 = Dual2<f64, f64>;

impl<T: DualNum<F>, F> Dual2<T, F> {
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: T, v2: T) -> Self {
        Self {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual2<T, F> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual2, DualNum};
    /// let x = Dual2::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);             // x²
    /// assert_eq!(x.v1, 10.0);    // 2x
    /// assert_eq!(x.v2, 2.0);     // 2
    /// ```
    ///
    /// Can also be used for higher order derivatives.
    /// ```
    /// # use num_dual::{Dual64, Dual2, DualNum};
    /// let x = Dual2::from_re(Dual64::from_re(5.0).derivative())
    ///     .derivative()
    ///     .powi(2);
    /// assert_eq!(x.re.re, 25.0);      // x²
    /// assert_eq!(x.re.eps, 10.0);     // 2x
    /// assert_eq!(x.v1.re, 10.0);      // 2x
    /// assert_eq!(x.v1.eps, 2.0);      // 2
    /// assert_eq!(x.v2.re, 2.0);       // 2
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.v1 = T::one();
        self
    }
}

impl<T: DualNum<F>, F> Dual2<T, F> {
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero())
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
/// let x = Dual64::new(5.0, 1.0);
/// let (f, df, d2f) = second_derivative(|x| x.powi(3), x);
/// assert_eq!(f.re, 125.0);    // x³
/// assert_eq!(f.eps, 75.0);    // 3x²
/// assert_eq!(df.re, 75.0);    // 3x²
/// assert_eq!(df.eps, 30.0);   // 6x
/// assert_eq!(d2f.re, 30.0);   // 6x
/// assert_eq!(d2f.eps, 6.0);   // 6
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
    g(x).map(|r| (r.re, r.v1, r.v2))
}

/* chain rule */
impl<T: DualNum<F>, F: Float> Dual2<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.v1.clone() * f1.clone(),
            self.v2.clone() * f1 + self.v1.clone() * self.v1.clone() * f2,
        )
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a Dual2<T, F>> for &'b Dual2<T, F> {
    type Output = Dual2<T, F>;
    #[inline]
    fn mul(self, other: &Dual2<T, F>) -> Dual2<T, F> {
        Dual2::new(
            self.re.clone() * other.re.clone(),
            other.v1.clone() * self.re.clone() + self.v1.clone() * other.re.clone(),
            other.v2.clone() * self.re.clone()
                + self.v1.clone() * other.v1.clone()
                + other.v1.clone() * self.v1.clone()
                + self.v2.clone() * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a Dual2<T, F>> for &'b Dual2<T, F> {
    type Output = Dual2<T, F>;
    #[inline]
    fn div(self, other: &Dual2<T, F>) -> Dual2<T, F> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * inv.clone();
        Dual2::new(
            self.re.clone() * inv.clone(),
            (self.v1.clone() * other.re.clone() - other.v1.clone() * self.re.clone())
                * inv2.clone(),
            self.v2.clone() * inv.clone()
                - (other.v2.clone() * self.re.clone()
                    + self.v1.clone() * other.v1.clone()
                    + other.v1.clone() * self.v1.clone())
                    * inv2.clone()
                + other.v1.clone()
                    * other.v1.clone()
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display> fmt::Display for Dual2<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε1 + {}ε1²", self.re, self.v1, self.v2)
    }
}

impl_second_derivatives!(Dual2, [v1, v2]);
impl_dual!(Dual2, [v1, v2]);
