use crate::{DualNum, DualNumFloat, DualStruct};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A scalar dual number for the calculations of first derivatives.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dual<T: DualNum<F>, F> {
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    f: PhantomData<F>,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum<F>, F: DualNumFloat> ndarray::ScalarOperand for Dual<T, F> {}

pub type Dual32 = Dual<f32, f32>;
pub type Dual64 = Dual<f64, f64>;

impl<T: DualNum<F>, F> Dual<T, F> {
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F> + Zero, F> Dual<T, F> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero())
    }
}

impl<T: DualNum<F> + One, F> Dual<T, F> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps, 10.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.eps = T::one();
        self
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float> Dual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, self.eps.clone() * f1)
    }
}

/* product rule */
impl<T: DualNum<F>, F: Float> Mul<&Dual<T, F>> for &Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn mul(self, other: &Dual<T, F>) -> Self::Output {
        Dual::new(
            self.re.clone() * other.re.clone(),
            self.eps.clone() * other.re.clone() + other.eps.clone() * self.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum<F>, F: Float> Div<&Dual<T, F>> for &Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn div(self, other: &Dual<T, F>) -> Dual<T, F> {
        let inv = other.re.recip();
        Dual::new(
            self.re.clone() * inv.clone(),
            (self.eps.clone() * other.re.clone() - other.eps.clone() * self.re.clone())
                * inv.clone()
                * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F> fmt::Display for Dual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl_first_derivatives!(Dual, [eps]);
impl_dual!(Dual, [eps]);
impl_nalgebra!(Dual, [eps]);
