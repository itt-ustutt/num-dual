use crate::{DualNum, DualNumFloat};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::*;
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
#[derive(Copy, Clone, Debug)]
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
impl<T> nalgebra::SimdValue for Dual2<T, T::Element>
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
    type Element = Dual2<T::Element, T::Element>;
    type SimdBool = T::SimdBool;

    const LANES: usize = T::LANES;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        // Need to make `lanes` copies of each of:
        // - the real part
        // - each of the N epsilon parts
        let re = T::splat(val.re);
        let v1 = T::splat(val.v1);
        let v2 = T::splat(val.v2);
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
        let re = self.re.extract_unchecked(i);
        let v1 = self.v1.extract_unchecked(i);
        let v2 = self.v2.extract_unchecked(i);
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
        self.re.replace_unchecked(i, val.re);
        self.v1.replace_unchecked(i, val.v1);
        self.v2.replace_unchecked(i, val.v2);
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
impl<T: DualNum<F> + PartialEq, F: Float> PartialEq for Dual2<T, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialOrd, F: Float> PartialOrd for Dual2<T, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + approx::AbsDiffEq<Epsilon = T>, F: Float> approx::AbsDiffEq for Dual2<T, F> {
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
impl<T: DualNum<F> + approx::RelativeEq<Epsilon = T>, F: Float> approx::RelativeEq for Dual2<T, F> {
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
impl<T: DualNum<F> + UlpsEq<Epsilon = T>, F: Float> UlpsEq for Dual2<T, F> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.re, &other.re, epsilon.re, max_ulps)
    }
}

impl<T> nalgebra::Field for Dual2<T, T::Element>
where
    T: DualNum<T::Element> + SimdValue,
    T::Element: DualNum<T::Element> + Scalar + Float,
{
}

use simba::scalar::{SubsetOf, SupersetOf};

impl<TSuper, FSuper, T, F> SubsetOf<Dual2<TSuper, FSuper>> for Dual2<T, F>
where
    TSuper: DualNum<FSuper> + SupersetOf<T>,
    T: DualNum<F>,
{
    #[inline(always)]
    fn to_superset(&self) -> Dual2<TSuper, FSuper> {
        let re = TSuper::from_subset(&self.re);
        let v1 = TSuper::from_subset(&self.v1);
        let v2 = TSuper::from_subset(&self.v2);
        Dual2 {
            re,
            v1,
            v2,
            f: PhantomData,
        }
    }
    #[inline(always)]
    fn from_superset(element: &Dual2<TSuper, FSuper>) -> Option<Self> {
        let re = TSuper::to_subset(&element.re)?;
        let v1 = TSuper::to_subset(&element.v1)?;
        let v2 = TSuper::to_subset(&element.v2)?;
        Some(Self::new(re, v1, v2))
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &Dual2<TSuper, FSuper>) -> Self {
        let re = TSuper::to_subset_unchecked(&element.re);
        let v1 = TSuper::to_subset_unchecked(&element.v1);
        let v2 = TSuper::to_subset_unchecked(&element.v2);
        Self::new(re, v1, v2)
    }
    #[inline(always)]
    fn is_in_subset(element: &Dual2<TSuper, FSuper>) -> bool {
        TSuper::is_in_subset(&element.re)
            && TSuper::is_in_subset(&element.v1)
            && TSuper::is_in_subset(&element.v2)
    }
}

impl<TSuper, FSuper> SupersetOf<f32> for Dual2<TSuper, FSuper>
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
        let v1 = TSuper::zero();
        let v2 = TSuper::zero();
        Self::new(re, v1, v2)
    }
}

impl<TSuper, FSuper> SupersetOf<f64> for Dual2<TSuper, FSuper>
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
        let v1 = TSuper::zero();
        let v2 = TSuper::zero();
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
impl<T> ComplexField for Dual2<T, T::Element>
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

impl<T> RealField for Dual2<T, T::Element>
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
        let re = self.re.atan2(other.re);
        let den = self.re.powi(2) + other.re.powi(2);

        let da = other.re / den;
        let db = -self.re / den;
        let v1 = self.v1 * da + other.v1 * db;

        let daa = db * da * (T::one() + T::one());
        let dab = db * db - da * da;
        let dbb = -daa;
        let ca = self.v1 * daa + other.v1 * dab;
        let cb = self.v1 * dab + other.v1 * dbb;
        let v2 = self.v2 * da + other.v2 * db + ca * self.v1 + cb * other.v1;

        Self::new(re, v1, v2)
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

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_atan2() {
        let x = Dual2_64::from(2.0).derivative();
        let y = Dual2_64::from(-3.0);
        let z = x.atan2(y);
        let z2 = (x / y).atan();
        assert_relative_eq!(z.v1, z2.v1, epsilon = 1e-14);
        assert_relative_eq!(z.v2, z2.v2, epsilon = 1e-14);
    }
}
