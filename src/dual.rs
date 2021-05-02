use crate::{DualNum, DualNumFloat};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar dual number for the calculation of first derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct Dual<T, F = T> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: T,
    f: PhantomData<F>,
}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

impl<T, F> Dual<T, F> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Dual {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: Zero, F> Dual<T, F> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Dual::new(re, T::zero())
    }
}

impl<T: One, F> Dual<T, F> {
    /// Derive a dual number, i.e. set the derivative part to 1.
    /// ```
    /// # use num_hyperdual::{Dual, DualNum};
    /// let x = Dual::from_re(5.0).derive().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps, 10.0);
    /// ```
    #[inline]
    pub fn derive(mut self) -> Self {
        self.eps = T::one();
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float> Dual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, self.eps * f1)
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn mul(self, other: &Dual<T, F>) -> Self::Output {
        Dual::new(
            self.re * other.re,
            other.eps * self.re + self.eps * other.re,
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn div(self, other: &Dual<T, F>) -> Dual<T, F> {
        let inv = other.re.recip();
        let inv2 = inv * inv;
        Dual::new(
            self.re * inv,
            (self.eps * other.re - other.eps * self.re) * inv2,
        )
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for Dual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl_first_derivatives!(Dual, [], [eps]);
impl_dual!(Dual, [], [eps]);
