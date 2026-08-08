use crate::{Derivative, DualNum, DualNumFloat, DualStruct};
use nalgebra::allocator::Allocator;
use nalgebra::*;
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A vector dual number for the calculations of gradients or Jacobians.
#[derive(Clone, Debug)]
pub struct DualVec<T: Scalar, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: Derivative<T, D, U1>,
}

#[cfg(feature = "ndarray")]
impl<T: Scalar, D: Dim> ndarray::ScalarOperand for DualVec<T, D> where DefaultAllocator: Allocator<D>
{}

impl<T: Scalar + Copy, const N: usize> Copy for DualVec<T, Const<N>> {}

pub type DualSVec<D, const N: usize> = DualVec<D, Const<N>>;
pub type DualDVec<D> = DualVec<D, Dyn>;
pub type DualVec32<D> = DualVec<f32, D>;
pub type DualVec64<D> = DualVec<f64, D>;
pub type DualSVec32<const N: usize> = DualVec<f32, Const<N>>;
pub type DualSVec64<const N: usize> = DualVec<f64, Const<N>>;
pub type DualDVec32 = DualVec<f32, Dyn>;
pub type DualDVec64 = DualVec<f64, Dyn>;

impl<T: DualNum, D: Dim> DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: Derivative<T, D, U1>) -> Self {
        Self { re, eps }
    }
}

impl<T: DualNum, const N: usize> DualSVec<T, N> {
    /// Set the derivative part of variable `index` to 1.
    ///
    /// For most cases, the [`gradient`](crate::gradient) function provides a convenient interface
    /// to calculate derivatives. This function exists for the more edge cases
    /// where more control over the variables is required.
    /// ```
    /// # use num_dual::DualSVec64;
    /// # use nalgebra::{U1, U2, vector};
    /// let x: DualSVec64<2> = DualSVec64::from_re(5.0).derivative(0);
    /// let y: DualSVec64<2> = DualSVec64::from_re(3.0).derivative(1);
    /// let z = x * x * y;
    /// assert_eq!(z.re, 75.0);                                           // x²y
    /// assert_eq!(z.eps.unwrap_generic(U2, U1), vector![30.0, 25.0]);    // [2xy, x²]
    /// ```
    #[inline]
    pub fn derivative(mut self, index: usize) -> Self {
        self.eps = Derivative::derivative_generic(Const::<N>, U1, index);
        self
    }
}

impl<T: DualNum> DualDVec<T> {
    /// Set the derivative part of variable `index` to 1.
    ///
    /// For most cases, the [`gradient`](crate::gradient) function provides a convenient interface
    /// to calculate derivatives. This function exists for the more edge cases
    /// where more control over the variables is required.
    /// ```
    /// # use num_dual::DualDVec64;
    /// # use nalgebra::{Dyn, U1, dvector};
    /// let x: DualDVec64 = DualDVec64::from_re(5.0).derivative(2, 0);
    /// let y: DualDVec64 = DualDVec64::from_re(3.0).derivative(2, 1);
    /// let z = &x * &x * y;
    /// assert_eq!(z.re, 75.0);                                               // x²y
    /// assert_eq!(z.eps.unwrap_generic(Dyn(2), U1), dvector![30.0, 25.0]);   // [2xy, x²]
    /// ```
    #[inline]
    pub fn derivative(mut self, variables: usize, index: usize) -> Self {
        self.eps = Derivative::derivative_generic(Dyn(variables), U1, index);
        self
    }
}

impl<T: DualNum, D: Dim> DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, Derivative::none())
    }
}

/* chain rule */
impl<T: DualNum, D: Dim> DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, &self.eps * f1)
    }
}

/* product rule */
impl<T: DualNum, D: Dim> Mul<&DualVec<T, D>> for &DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Output = DualVec<T, D>;
    #[inline]
    fn mul(self, other: &DualVec<T, D>) -> Self::Output {
        DualVec::new(
            self.re.clone() * other.re.clone(),
            &self.eps * other.re.clone() + &other.eps * self.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum, D: Dim> Div<&DualVec<T, D>> for &DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Output = DualVec<T, D>;
    #[inline]
    fn div(self, other: &DualVec<T, D>) -> DualVec<T, D> {
        let inv = other.re.recip();
        DualVec::new(
            self.re.clone() * inv.clone(),
            (&self.eps * other.re.clone() - &other.eps * self.re.clone()) * inv.clone() * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum, D: Dim> fmt::Display for DualVec<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.eps.fmt(f, "ε")
    }
}

impl_first_derivatives!(DualVec, [eps], [D], [D]);
impl_dual!(DualVec, [eps], [D], [D]);
impl_nalgebra!(DualVec, [eps], [D], [D]);
