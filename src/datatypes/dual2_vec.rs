use crate::{Derivative, DualNum, DualNumFloat, DualStruct};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
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

/**
 * The SimdValue trait is for rearranging data into a form more suitable for Simd,
 * and rearranging it back into a usable form. It is not documented particularly well.
 *
 * The primary job of this SimdValue impl is to allow people to use `simba::simd::f32x4` etc,
 * instead of f32/f64. Those types implement nalgebra::SimdRealField/ComplexField, so they
 * behave like scalars. When we use them, we would have `DualVec<f32x4, f32, N>` etc, with our
 * F parameter set to `<T as SimdValue>::Element`. We will need to be able to split up that type
 * into four of DualVec in order to get out of simd-land. That's what the SimdValue trait is for.
 *
 * Ultimately, someone will have to to implement SimdRealField on DualVec and call the
 * simd_ functions of `<T as SimdRealField>`. That's future work for someone who finds
 * num_dual is not fast enough.
 *
 * Unfortunately, doing anything with SIMD is blocked on
 * <https://github.com/dimforge/simba/issues/44>.
 *
 */
impl<T, D: Dim> nalgebra::SimdValue for Dual2Vec<T, T::Element, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
    T: DualNum<T::Element> + SimdValue + Scalar,
    T::Element: DualNum<T::Element> + Scalar,
{
    // Say T = simba::f32x4. T::Element is f32. T::SimdBool is AutoSimd<[bool; 4]>.
    // AutoSimd<[f32; 4]> stores an actual [f32; 4], i.e. four floats in one slot.
    // So our DualVec<AutoSimd<[f32; 4], f32, N> has 4 * (1+N) floats in it, stored in blocks of
    // four. When we want to do any math on it but ignore its f32x4 storage mode, we need to break
    // that type into FOUR of DualVec<f32, f32, N>; then we do math on it, then we bring it back
    // together.
    //
    // Hence this definition of Element:
    type Element = Dual2Vec<T::Element, T::Element, D>;
    type SimdBool = T::SimdBool;

    const LANES: usize = T::LANES;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        // Need to make `lanes` copies of each of:
        // - the real part
        // - each of the N epsilon parts
        let re = T::splat(val.re);
        let v1 = Derivative::splat(val.v1);
        let v2 = Derivative::splat(val.v2);
        Self::new(re, v1, v2)
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        let re = self.re.extract(i);
        let v1 = self.v1.extract(i);
        let v2 = self.v2.extract(i);
        Self::Element {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        let re = unsafe { self.re.extract_unchecked(i) };
        let v1 = unsafe { self.v1.extract_unchecked(i) };
        let v2 = unsafe { self.v2.extract_unchecked(i) };
        Self::Element {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.re.replace(i, val.re);
        self.v1.replace(i, val.v1);
        self.v2.replace(i, val.v2);
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        unsafe { self.re.replace_unchecked(i, val.re) };
        unsafe { self.v1.replace_unchecked(i, val.v1) };
        unsafe { self.v2.replace_unchecked(i, val.v2) };
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        let re = self.re.select(cond, other.re);
        let v1 = self.v1.select(cond, other.v1);
        let v2 = self.v2.select(cond, other.v2);
        Self::new(re, v1, v2)
    }
}

/// Comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialEq, F: Float, D: Dim> PartialEq for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialOrd, F: Float, D: Dim> PartialOrd for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + approx::AbsDiffEq<Epsilon = T>, F: Float, D: Dim> approx::AbsDiffEq
    for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    type Epsilon = Self;
    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.re)
    }

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self::from_re(T::default_epsilon())
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + approx::RelativeEq<Epsilon = T>, F: Float, D: Dim> approx::RelativeEq
    for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        Self::from_re(T::default_max_relative())
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.re.relative_eq(&other.re, epsilon.re, max_relative.re)
    }
}
impl<T: DualNum<F> + UlpsEq<Epsilon = T>, F: Float, D: Dim> UlpsEq for Dual2Vec<T, F, D>
where
    DefaultAllocator: Allocator<U1, D> + Allocator<D, D>,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.re, &other.re, epsilon.re, max_ulps)
    }
}

impl<T, D: Dim> nalgebra::Field for Dual2Vec<T, T::Element, D>
where
    T: DualNum<T::Element> + SimdValue,
    T::Element: DualNum<T::Element> + Scalar + Float,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
{
}

use simba::scalar::{SubsetOf, SupersetOf};

impl<TSuper, FSuper, T, F, D: Dim> SubsetOf<Dual2Vec<TSuper, FSuper, D>> for Dual2Vec<T, F, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<T>,
    T: DualNum<F>,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
{
    #[inline(always)]
    fn to_superset(&self) -> Dual2Vec<TSuper, FSuper, D> {
        let re = TSuper::from_subset(&self.re);
        let v1 = Derivative::from_subset(&self.v1);
        let v2 = Derivative::from_subset(&self.v2);
        Dual2Vec {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
    #[inline(always)]
    fn from_superset(element: &Dual2Vec<TSuper, FSuper, D>) -> Option<Self> {
        let re = TSuper::to_subset(&element.re)?;
        let v1 = Derivative::to_subset(&element.v1)?;
        let v2 = Derivative::to_subset(&element.v2)?;
        Some(Self::new(re, v1, v2))
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &Dual2Vec<TSuper, FSuper, D>) -> Self {
        let re = TSuper::to_subset_unchecked(&element.re);
        let v1 = Derivative::to_subset_unchecked(&element.v1);
        let v2 = Derivative::to_subset_unchecked(&element.v2);
        Self::new(re, v1, v2)
    }
    #[inline(always)]
    fn is_in_subset(element: &Dual2Vec<TSuper, FSuper, D>) -> bool {
        TSuper::is_in_subset(&element.re)
            && <Derivative<_, _, _, _> as SupersetOf<Derivative<_, _, _, _>>>::is_in_subset(
                &element.v1,
            )
            && <Derivative<_, _, _, _> as SupersetOf<Derivative<_, _, _, _>>>::is_in_subset(
                &element.v2,
            )
    }
}

impl<TSuper, FSuper, D: Dim> SupersetOf<f32> for Dual2Vec<TSuper, FSuper, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<f32>,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
{
    #[inline(always)]
    fn is_in_subset(&self) -> bool {
        self.re.is_in_subset()
    }

    #[inline(always)]
    fn to_subset_unchecked(&self) -> f32 {
        self.re.to_subset_unchecked()
    }

    #[inline(always)]
    fn from_subset(element: &f32) -> Self {
        // Interpret as a purely real number
        let re = TSuper::from_subset(element);
        let v1 = Derivative::none();
        let v2 = Derivative::none();
        Self::new(re, v1, v2)
    }
}

impl<TSuper, FSuper, D: Dim> SupersetOf<f64> for Dual2Vec<TSuper, FSuper, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<f64>,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
{
    #[inline(always)]
    fn is_in_subset(&self) -> bool {
        self.re.is_in_subset()
    }

    #[inline(always)]
    fn to_subset_unchecked(&self) -> f64 {
        self.re.to_subset_unchecked()
    }

    #[inline(always)]
    fn from_subset(element: &f64) -> Self {
        // Interpret as a purely real number
        let re = TSuper::from_subset(element);
        let v1 = Derivative::none();
        let v2 = Derivative::none();
        Self::new(re, v1, v2)
    }
}

// We can't do a simd implementation until simba lets us implement SimdPartialOrd
// using _T_'s SimdBool. The blanket impl gets in the way. So we must constrain
// T to SimdValue<Element = T, SimdBool = bool>, which is basically the same as
// saying f32 or f64 only.
//
// Limitation of simba. See https://github.com/dimforge/simba/issues/44

use nalgebra::{ComplexField, RealField};
// This impl is modelled on `impl ComplexField for f32`. The imaginary part is nothing.
impl<T, D: Dim> ComplexField for Dual2Vec<T, T::Element, D>
where
    T: DualNum<T::Element> + SupersetOf<T> + AbsDiffEq<Epsilon = T> + Sync + Send,
    T::Element: DualNum<T::Element> + Scalar + DualNumFloat + Sync + Send,
    T: SupersetOf<T::Element>,
    T: SupersetOf<f32>,
    T: SupersetOf<f64>,
    T: SimdPartialOrd + PartialOrd,
    T: SimdValue<Element = T, SimdBool = bool>,
    T: RelativeEq + UlpsEq + AbsDiffEq,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<U1, D>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<D, U1>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<D, D>>::Buffer<T>: Sync + Send,
{
    type RealField = Self;

    #[inline]
    fn from_real(re: Self::RealField) -> Self {
        re
    }

    #[inline]
    fn real(self) -> Self::RealField {
        self
    }

    #[inline]
    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    #[inline]
    fn modulus(self) -> Self::RealField {
        self.abs()
    }

    #[inline]
    fn modulus_squared(self) -> Self::RealField {
        &self * &self
    }

    #[inline]
    fn argument(self) -> Self::RealField {
        Self::zero()
    }

    #[inline]
    fn norm1(self) -> Self::RealField {
        self.abs()
    }

    #[inline]
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }

    #[inline]
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }

    #[inline]
    fn floor(self) -> Self {
        panic!("called floor() on a dual number")
    }

    #[inline]
    fn ceil(self) -> Self {
        panic!("called ceil() on a dual number")
    }

    #[inline]
    fn round(self) -> Self {
        panic!("called round() on a dual number")
    }

    #[inline]
    fn trunc(self) -> Self {
        panic!("called trunc() on a dual number")
    }

    #[inline]
    fn fract(self) -> Self {
        panic!("called fract() on a dual number")
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        DualNum::mul_add(&self, a, b)
    }

    #[inline]
    fn abs(self) -> Self::RealField {
        Signed::abs(&self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        let sum_sq = self.powi(2) + other.powi(2);
        DualNum::sqrt(&sum_sq)
    }

    #[inline]
    fn recip(self) -> Self {
        DualNum::recip(&self)
    }

    #[inline]
    fn conjugate(self) -> Self {
        self
    }

    #[inline]
    fn sin(self) -> Self {
        DualNum::sin(&self)
    }

    #[inline]
    fn cos(self) -> Self {
        DualNum::cos(&self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        DualNum::sin_cos(&self)
    }

    #[inline]
    fn tan(self) -> Self {
        DualNum::tan(&self)
    }

    #[inline]
    fn asin(self) -> Self {
        DualNum::asin(&self)
    }

    #[inline]
    fn acos(self) -> Self {
        DualNum::acos(&self)
    }

    #[inline]
    fn atan(self) -> Self {
        DualNum::atan(&self)
    }

    #[inline]
    fn sinh(self) -> Self {
        DualNum::sinh(&self)
    }

    #[inline]
    fn cosh(self) -> Self {
        DualNum::cosh(&self)
    }

    #[inline]
    fn tanh(self) -> Self {
        DualNum::tanh(&self)
    }

    #[inline]
    fn asinh(self) -> Self {
        DualNum::asinh(&self)
    }

    #[inline]
    fn acosh(self) -> Self {
        DualNum::acosh(&self)
    }

    #[inline]
    fn atanh(self) -> Self {
        DualNum::atanh(&self)
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        DualNum::ln(&self) / DualNum::ln(&base)
    }

    #[inline]
    fn log2(self) -> Self {
        DualNum::log2(&self)
    }

    #[inline]
    fn log10(self) -> Self {
        DualNum::log10(&self)
    }

    #[inline]
    fn ln(self) -> Self {
        DualNum::ln(&self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        DualNum::ln_1p(&self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        DualNum::sqrt(&self)
    }

    #[inline]
    fn exp(self) -> Self {
        DualNum::exp(&self)
    }

    #[inline]
    fn exp2(self) -> Self {
        DualNum::exp2(&self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        DualNum::exp_m1(&self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        DualNum::powi(&self, n)
    }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        // n could be a dual.
        DualNum::powd(&self, n)
    }

    #[inline]
    fn powc(self, n: Self) -> Self {
        // same as powf, Self isn't complex
        self.powf(n)
    }

    #[inline]
    fn cbrt(self) -> Self {
        DualNum::cbrt(&self)
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.re.is_finite()
    }

    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        if self > Self::zero() {
            Some(DualNum::sqrt(&self))
        } else {
            None
        }
    }
}

impl<T, D: Dim> RealField for Dual2Vec<T, T::Element, D>
where
    T: DualNum<T::Element> + SupersetOf<T> + Sync + Send,
    T::Element: DualNum<T::Element> + Scalar + DualNumFloat,
    T: SupersetOf<T::Element>,
    T: SupersetOf<f32>,
    T: SupersetOf<f64>,
    T: SimdPartialOrd + PartialOrd,
    T: RelativeEq + AbsDiffEq<Epsilon = T>,
    T: SimdValue<Element = T, SimdBool = bool>,
    T: UlpsEq,
    T: AbsDiffEq,
    DefaultAllocator: Allocator<U1, D> + Allocator<D, U1> + Allocator<D, D>,
    <DefaultAllocator as Allocator<D>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<U1, D>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<D, U1>>::Buffer<T>: Sync + Send,
    <DefaultAllocator as Allocator<D, D>>::Buffer<T>: Sync + Send,
{
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if sign.re.is_sign_positive() {
            self.simd_abs()
        } else {
            -self.simd_abs()
        }
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        DualNum::atan2(&self, other)
    }

    #[inline]
    fn pi() -> Self {
        Self::from_re(<T as FloatConst>::PI())
    }

    #[inline]
    fn two_pi() -> Self {
        Self::from_re(<T as FloatConst>::TAU())
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_4())
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_3())
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_4())
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_6())
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_8())
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_1_PI())
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_2_PI())
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_2_SQRT_PI())
    }

    #[inline]
    fn e() -> Self {
        Self::from_re(<T as FloatConst>::E())
    }

    #[inline]
    fn log2_e() -> Self {
        Self::from_re(<T as FloatConst>::LOG2_E())
    }

    #[inline]
    fn log10_e() -> Self {
        Self::from_re(<T as FloatConst>::LOG10_E())
    }

    #[inline]
    fn ln_2() -> Self {
        Self::from_re(<T as FloatConst>::LN_2())
    }

    #[inline]
    fn ln_10() -> Self {
        Self::from_re(<T as FloatConst>::LN_10())
    }

    #[inline]
    fn is_sign_positive(&self) -> bool {
        self.re.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(&self) -> bool {
        self.re.is_sign_negative()
    }

    /// Got to be careful using this, because it throws away the derivatives of the one not chosen
    #[inline]
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// Got to be careful using this, because it throws away the derivatives of the one not chosen
    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    /// If the min/max values are constants and the clamping has an effect, you lose your gradients.
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::from_re(T::min_value()))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::from_re(T::max_value()))
    }
}
