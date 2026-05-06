use crate::{Derivative, DualNum, DualNumFloat, DualStruct};
use nalgebra::allocator::Allocator;
use nalgebra::*;
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A vector second order dual number for the calculation of Hessians.
#[derive(Clone, Debug)]
pub struct Dual2Vec<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    /// Real part of the second order dual number
    pub re: T,
    /// Gradient part of the second order dual number
    pub v1: Derivative<T, F, U1, D>,
    /// Hessian part of the second order dual number
    pub v2: Derivative<T, F, D, D>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Copy, const N: usize> Copy for Dual2Vec<T, F, Const<N>> {}

#[cfg(feature = "ndarray")]
impl<T: DualNum<F>, F: DualNumFloat, D: Dim> ndarray::ScalarOperand for Dual2Vec<T, F, D> where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>
{
}

pub type Dual2SVec<T, F, const N: usize> = Dual2Vec<T, F, Const<N>>;
pub type Dual2DVec<T, F> = Dual2Vec<T, F, Dyn>;
pub type Dual2Vec32<D> = Dual2Vec<f32, f32, D>;
pub type Dual2Vec64<D> = Dual2Vec<f64, f64, D>;
pub type Dual2SVec32<const N: usize> = Dual2Vec<f32, f32, Const<N>>;
pub type Dual2SVec64<const N: usize> = Dual2Vec<f64, f64, Const<N>>;
pub type Dual2DVec32 = Dual2Vec<f32, f32, Dyn>;
pub type Dual2DVec64 = Dual2Vec<f64, f64, Dyn>;

impl<T: DualNum<F>, F, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    /// Create a new second order dual number from its fields.
    #[inline]
    pub fn new(re: T, v1: Derivative<T, F, U1, D>, v2: Derivative<T, F, D, D>) -> Self {
        Self {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F, const N: usize> Dual2SVec<T, F, N> {
    /// Set the derivative part of variable `index` to 1.
    ///
    /// For most cases, the [`hessian`](crate::hessian) function provides a convenient
    /// interface to calculate derivatives. This function exists for the more edge cases
    /// where more control over the variables is required.
    /// ```
    /// # use num_dual::Dual2SVec64;
    /// # use nalgebra::{U1, U2, matrix};
    /// let x: Dual2SVec64<2> = Dual2SVec64::from_re(5.0).derivative(0);
    /// let y: Dual2SVec64<2> = Dual2SVec64::from_re(3.0).derivative(1);
    /// let z = x * x * y;
    /// assert_eq!(z.re, 75.0);                                                 // x²y
    /// assert_eq!(z.v1.unwrap_generic(U1, U2), matrix![30.0, 25.0]);           // [2xy, x²]
    /// assert_eq!(z.v2.unwrap_generic(U2, U2), matrix![6.0, 10.0; 10.0, 0.0]); // [2y, 2x; 2x, 0]
    /// ```
    #[inline]
    pub fn derivative(mut self, index: usize) -> Self {
        self.v1 = Derivative::derivative_generic(U1, Const::<N>, index);
        self
    }
}

impl<T: DualNum<F>, F> Dual2DVec<T, F> {
    /// Set the derivative part of variable `index` to 1.
    ///
    /// For most cases, the [`hessian`](crate::hessian) function provides a convenient interface
    /// to calculate derivatives. This function exists for the more edge cases
    /// where more control over the variables is required.
    /// ```
    /// # use num_dual::Dual2DVec64;
    /// # use nalgebra::{Dyn, U1, dmatrix};
    /// let x: Dual2DVec64 = Dual2DVec64::from_re(5.0).derivative(2, 0);
    /// let y: Dual2DVec64 = Dual2DVec64::from_re(3.0).derivative(2, 1);
    /// let z = &x * &x * y;
    /// assert_eq!(z.re, 75.0);                                                          // x²y
    /// assert_eq!(z.v1.unwrap_generic(U1, Dyn(2)), dmatrix![30.0, 25.0]);               // [2xy, x²]
    /// assert_eq!(z.v2.unwrap_generic(Dyn(2), Dyn(2)), dmatrix![6.0, 10.0; 10.0, 0.0]); // [2y, 2x; 2x, 0]
    /// ```
    #[inline]
    pub fn derivative(mut self, variables: usize, index: usize) -> Self {
        self.v1 = Derivative::derivative_generic(U1, Dyn(variables), index);
        self
    }
}

impl<T: DualNum<F>, F, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    /// Create a new second order dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, Derivative::none(), Derivative::none())
    }
}

/* chain rule */
impl<T: DualNum<F>, F: Float, D: Dim> Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T) -> Self {
        Self::new(
            f0,
            &self.v1 * f1.clone(),
            &self.v2 * f1 + self.v1.tr_mul(&self.v1) * f2,
        )
    }
}

/* product rule */
impl<T: DualNum<F>, F: Float, D: Dim> Mul<&Dual2Vec<T, F, D>> for &Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    type Output = Dual2Vec<T, F, D>;
    #[inline]
    fn mul(self, other: &Dual2Vec<T, F, D>) -> Dual2Vec<T, F, D> {
        Dual2Vec::new(
            self.re.clone() * other.re.clone(),
            &other.v1 * self.re.clone() + &self.v1 * other.re.clone(),
            &other.v2 * self.re.clone()
                + self.v1.tr_mul(&other.v1)
                + other.v1.tr_mul(&self.v1)
                + &self.v2 * other.re.clone(),
        )
    }
}

/* quotient rule */
impl<T: DualNum<F>, F: Float, D: Dim> Div<&Dual2Vec<T, F, D>> for &Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    type Output = Dual2Vec<T, F, D>;
    #[inline]
    fn div(self, other: &Dual2Vec<T, F, D>) -> Dual2Vec<T, F, D> {
        let inv = other.re.recip();
        let inv2 = inv.clone() * inv.clone();
        Dual2Vec::new(
            self.re.clone() * inv.clone(),
            (&self.v1 * other.re.clone() - &other.v1 * self.re.clone()) * inv2.clone(),
            &self.v2 * inv.clone()
                - (&other.v2 * self.re.clone()
                    + self.v1.tr_mul(&other.v1)
                    + other.v1.tr_mul(&self.v1))
                    * inv2.clone()
                + other.v1.tr_mul(&other.v1)
                    * ((T::one() + T::one()) * self.re.clone() * inv2 * inv),
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F: fmt::Display, D: Dim> fmt::Display for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.v1.fmt(f, "ε1")?;
        self.v2.fmt(f, "ε1²")
    }
}

impl_second_derivatives!(Dual2Vec, [v1, v2], [D], [U1, D], [D, D]);
impl_dual!(Dual2Vec, [v1, v2], [D], [U1, D], [D, D]);
impl_nalgebra!(Dual2Vec, [v1, v2], [D], [U1, D], [D, D]);
