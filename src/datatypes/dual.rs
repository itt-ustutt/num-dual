use crate::{DualNum, DualNumFloat, DualStruct};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::*;
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

/**
 * The SimdValue trait is for rearranging data into a form more suitable for Simd,
 * and rearranging it back into a usable form. It is not documented particularly well.
 *
 * The primary job of this SimdValue impl is to allow people to use `simba::simd::f32x4` etc,
 * instead of f32/f64. Those types implement nalgebra::SimdRealField/ComplexField, so they
 * behave like scalars. When we use them, we would have `Dual<f32x4, f32, N>` etc, with our
 * F parameter set to `<T as SimdValue>::Element`. We will need to be able to split up that type
 * into four of Dual in order to get out of simd-land. That's what the SimdValue trait is for.
 *
 * Ultimately, someone will have to to implement SimdRealField on Dual and call the
 * simd_ functions of `<T as SimdRealField>`. That's future work for someone who finds
 * num_dual is not fast enough.
 *
 * Unfortunately, doing anything with SIMD is blocked on
 * <https://github.com/dimforge/simba/issues/44>.
 *
 */
impl<T> nalgebra::SimdValue for Dual<T, T::Element>
where
    T: DualNum<T::Element> + SimdValue + Scalar,
    T::Element: DualNum<T::Element> + Scalar,
{
    // Say T = simba::f32x4. T::Element is f32. T::SimdBool is AutoSimd<[bool; 4]>.
    // AutoSimd<[f32; 4]> stores an actual [f32; 4], i.e. four floats in one slot.
    // So our Dual<AutoSimd<[f32; 4], f32, N> has 4 * (1+N) floats in it, stored in blocks of
    // four. When we want to do any math on it but ignore its f32x4 storage mode, we need to break
    // that type into FOUR of Dual<f32, f32, N>; then we do math on it, then we bring it back
    // together.
    //
    // Hence this definition of Element:
    type Element = Dual<T::Element, T::Element>;
    type SimdBool = T::SimdBool;

    const LANES: usize = T::LANES;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        // Need to make `lanes` copies of each of:
        // - the real part
        // - each of the N epsilon parts
        let re = T::splat(val.re);
        let eps = T::splat(val.eps);
        Self::new(re, eps)
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        let re = self.re.extract(i);
        let eps = self.eps.extract(i);
        Self::Element {
            re,
            eps,
            f: PhantomData,
        }
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        let re = self.re.extract_unchecked(i);
        let eps = self.eps.extract_unchecked(i);
        Self::Element {
            re,
            eps,
            f: PhantomData,
        }
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.re.replace(i, val.re);
        self.eps.replace(i, val.eps);
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.re.replace_unchecked(i, val.re);
        self.eps.replace_unchecked(i, val.eps);
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        let re = self.re.select(cond, other.re);
        let eps = self.eps.select(cond, other.eps);
        Self::new(re, eps)
    }
}

/// Comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialEq, F: Float> PartialEq for Dual<T, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialOrd, F: Float> PartialOrd for Dual<T, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + approx::AbsDiffEq<Epsilon = T>, F: Float> approx::AbsDiffEq for Dual<T, F> {
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
impl<T: DualNum<F> + approx::RelativeEq<Epsilon = T>, F: Float> approx::RelativeEq for Dual<T, F> {
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
impl<T: DualNum<F> + UlpsEq<Epsilon = T>, F: Float> UlpsEq for Dual<T, F> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.re, &other.re, epsilon.re, max_ulps)
    }
}

impl<T> nalgebra::Field for Dual<T, T::Element>
where
    T: DualNum<T::Element> + SimdValue,
    T::Element: DualNum<T::Element> + Scalar + Float,
{
}

use simba::scalar::{SubsetOf, SupersetOf};

impl<TSuper, FSuper, T, F> SubsetOf<Dual<TSuper, FSuper>> for Dual<T, F>
where
    TSuper: DualNum<FSuper> + SupersetOf<T>,
    T: DualNum<F>,
{
    #[inline(always)]
    fn to_superset(&self) -> Dual<TSuper, FSuper> {
        let re = TSuper::from_subset(&self.re);
        let eps = TSuper::from_subset(&self.eps);
        Dual {
            re,
            eps,
            f: PhantomData,
        }
    }
    #[inline(always)]
    fn from_superset(element: &Dual<TSuper, FSuper>) -> Option<Self> {
        let re = TSuper::to_subset(&element.re)?;
        let eps = TSuper::to_subset(&element.eps)?;
        Some(Self::new(re, eps))
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &Dual<TSuper, FSuper>) -> Self {
        let re = TSuper::to_subset_unchecked(&element.re);
        let eps = TSuper::to_subset_unchecked(&element.eps);
        Self::new(re, eps)
    }
    #[inline(always)]
    fn is_in_subset(element: &Dual<TSuper, FSuper>) -> bool {
        TSuper::is_in_subset(&element.re) && TSuper::is_in_subset(&element.eps)
    }
}

impl<TSuper, FSuper> SupersetOf<f32> for Dual<TSuper, FSuper>
where
    TSuper: DualNum<FSuper> + SupersetOf<f32>,
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
        let eps = TSuper::zero();
        Self::new(re, eps)
    }
}

impl<TSuper, FSuper> SupersetOf<f64> for Dual<TSuper, FSuper>
where
    TSuper: DualNum<FSuper> + SupersetOf<f64>,
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
        let eps = TSuper::zero();
        Self::new(re, eps)
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
impl<T> ComplexField for Dual<T, T::Element>
where
    T: DualNum<T::Element> + SupersetOf<T> + AbsDiffEq<Epsilon = T> + Sync + Send,
    T::Element: DualNum<T::Element> + Scalar + DualNumFloat + Sync + Send,
    T: SupersetOf<T::Element>,
    T: SupersetOf<f32>,
    T: SupersetOf<f64>,
    T: SimdPartialOrd + PartialOrd,
    T: SimdValue<Element = T, SimdBool = bool>,
    T: RelativeEq + UlpsEq + AbsDiffEq,
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
        self * self
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

impl<T> RealField for Dual<T, T::Element>
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
        if other > self {
            other
        } else {
            self
        }
    }

    /// Got to be careful using this, because it throws away the derivatives of the one not chosen
    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self {
            other
        } else {
            self
        }
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
