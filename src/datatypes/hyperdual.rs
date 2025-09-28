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

/// A scalar hyper-dual number for the calculation of second partial derivatives.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HyperDual<T: DualNum<F>, F> {
    /// Real part of the hyper-dual number
    pub re: T,
    /// Partial derivative part of the hyper-dual number
    pub eps1: T,
    /// Partial derivative part of the hyper-dual number
    pub eps2: T,
    /// Second partial derivative part of the hyper-dual number
    pub eps1eps2: T,
    #[cfg_attr(feature = "serde", serde(skip))]
    f: PhantomData<F>,
}

#[cfg(feature = "ndarray")]
impl<T: DualNum<F>, F: DualNumFloat> ndarray::ScalarOperand for HyperDual<T, F> {}

pub type HyperDual32 = HyperDual<f32, f32>;
pub type HyperDual64 = HyperDual<f64, f64>;

impl<T: DualNum<F>, F> HyperDual<T, F> {
    /// Create a new hyper-dual number from its fields.
    #[inline]
    pub fn new(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
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
}

impl<T: DualNum<F>, F> HyperDual<T, F> {
    /// Create a new hyper-dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero(), T::zero(), T::zero())
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float> HyperDual<T, F> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            self.eps1.clone() * f1.clone(),
            self.eps2.clone() * f1.clone(),
            self.eps1eps2.clone() * f1 + self.eps1.clone() * self.eps2.clone() * f2,
        )
    }
}

/* product rule */
impl<T: DualNum<F>, F: Float> Mul<&HyperDual<T, F>> for &HyperDual<T, F> {
    type Output = HyperDual<T, F>;
    #[inline]
    fn mul(self, other: &HyperDual<T, F>) -> HyperDual<T, F> {
        HyperDual::new(
            self.re.clone() * other.re.clone(),
            other.eps1.clone() * self.re.clone() + self.eps1.clone() * other.re.clone(),
            other.eps2.clone() * self.re.clone() + self.eps2.clone() * other.re.clone(),
            other.eps1eps2.clone() * self.re.clone()
                + self.eps1.clone() * other.eps2.clone()
                + other.eps1.clone() * self.eps2.clone()
                + self.eps1eps2.clone() * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum<F>, F: Float> Div<&HyperDual<T, F>> for &HyperDual<T, F> {
    type Output = HyperDual<T, F>;
    #[inline]
    fn div(self, other: &HyperDual<T, F>) -> HyperDual<T, F> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * &inv;
        HyperDual::new(
            self.re.clone() * &inv,
            (self.eps1.clone() * other.re.clone() - other.eps1.clone() * self.re.clone())
                * inv2.clone(),
            (self.eps2.clone() * other.re.clone() - other.eps2.clone() * self.re.clone())
                * inv2.clone(),
            self.eps1eps2.clone() * inv.clone()
                - (other.eps1eps2.clone() * self.re.clone()
                    + self.eps1.clone() * other.eps2.clone()
                    + other.eps1.clone() * self.eps2.clone())
                    * inv2.clone()
                + other.eps1.clone()
                    * other.eps2.clone()
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display> fmt::Display for HyperDual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.re, f)?;
        write!(f, " + ")?;
        fmt::Display::fmt(&self.eps1, f)?;
        write!(f, "ε1 + ")?;
        fmt::Display::fmt(&self.eps2, f)?;
        write!(f, "ε2 + ")?;
        fmt::Display::fmt(&self.eps1eps2, f)?;
        write!(f, "ε1ε2")
    }
}

impl_second_derivatives!(HyperDual, [eps1, eps2, eps1eps2]);
impl_dual!(HyperDual, [eps1, eps2, eps1eps2]);
