use crate::{Derivative, DualNum, DualNumFloat};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::allocator::Allocator;
use nalgebra::*;
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::convert::Infallible;
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number for the calculations of gradients or Jacobians.
#[derive(Clone, Debug)]
pub struct DualVec<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Real part of the dual number
    pub re: T,
    /// Derivative part of the dual number
    pub eps: Derivative<T, F, D, U1>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Copy, const N: usize> Copy for DualVec<T, F, Const<N>> {}

pub type DualVec32<D> = DualVec<f32, f32, D>;
pub type DualVec64<D> = DualVec<f64, f64, D>;
pub type DualSVec32<const N: usize> = DualVec<f32, f32, Const<N>>;
pub type DualSVec64<const N: usize> = DualVec<f64, f64, Const<N>>;
pub type DualDVec32 = DualVec<f32, f32, Dyn>;
pub type DualDVec64 = DualVec<f64, f64, Dyn>;
pub type Dual<T, F> = DualVec<T, F, U1>;
pub type Dual32 = Dual<f32, f32>;
pub type Dual64 = Dual<f64, f64>;

impl<T: DualNum<F>, F, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new dual number from its fields.
    #[inline]
    pub fn new(re: T, eps: Derivative<T, F, D, U1>) -> Self {
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: DualNum<F>, F> Dual<T, F> {
    /// Create a new scalar dual number from its fields.
    #[inline]
    pub fn new_scalar(re: T, eps: T) -> Self {
        Self::new(re, Derivative::some(SVector::from_element(eps)))
    }
}

impl<T: DualNum<F> + Zero, F, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, Derivative::none())
    }
}

impl<T: DualNum<F> + One, F> Dual<T, F> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps.unwrap(), 10.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.eps = Derivative::derivative();
        self
    }
}

/// Calculate the first derivative of a scalar function.
/// ```
/// # use num_dual::{first_derivative, DualNum};
/// let (f, df) = first_derivative(|x| x.powi(2), 5.0);
/// assert_eq!(f, 25.0);
/// assert_eq!(df, 10.0);
/// ```
pub fn first_derivative<G, T: DualNum<F>, F>(g: G, x: T) -> (T, T)
where
    G: FnOnce(Dual<T, F>) -> Dual<T, F>,
{
    try_first_derivative(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [first_derivative] for fallible functions.
pub fn try_first_derivative<G, T: DualNum<F>, F, E>(g: G, x: T) -> Result<(T, T), E>
where
    G: FnOnce(Dual<T, F>) -> Result<Dual<T, F>, E>,
{
    let x = Dual::from_re(x).derivative();
    g(x).map(|r| (r.re, r.eps.unwrap()))
}

/// Calculate the gradient of a scalar function
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualSVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<DualSVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g) = gradient(fun, v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// ```
///
/// The variable vector can also be dynamically sized
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualDVec64};
/// # use nalgebra::DVector;
/// let v = DVector::repeat(4, 2.0);
/// let fun = |v: DVector<DualDVec64>| v.iter().map(|v| v * v).sum::<DualDVec64>().sqrt();
/// let (f, g) = gradient(fun, v);
/// assert_eq!(f, 4.0);
/// assert_relative_eq!(g[0], 0.5);
/// assert_relative_eq!(g[1], 0.5);
/// assert_relative_eq!(g[2], 0.5);
/// assert_relative_eq!(g[3], 0.5);
/// ```
pub fn gradient<G, T: DualNum<F>, F: DualNumFloat, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> (T, OVector<T, D>)
where
    G: FnOnce(OVector<DualVec<T, F, D>, D>) -> DualVec<T, F, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<DualVec<T, F, D>, D>,
{
    try_gradient(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [gradient] for fallible functions.
pub fn try_gradient<G, T: DualNum<F>, F: DualNumFloat, E, D: Dim>(
    g: G,
    x: OVector<T, D>,
) -> Result<(T, OVector<T, D>), E>
where
    G: FnOnce(OVector<DualVec<T, F, D>, D>) -> Result<DualVec<T, F, D>, E>,
    DefaultAllocator: Allocator<T, D> + Allocator<DualVec<T, F, D>, D>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative_generic(r, c, i);
    }
    g(x).map(|res| (res.re, res.eps.unwrap_generic(r, c)))
}

/// Calculate the Jacobian of a vector function.
/// ```
/// # use num_dual::{jacobian, DualSVec64, DualNum};
/// # use nalgebra::SVector;
/// let xy = SVector::from([5.0, 3.0, 2.0]);
/// let fun = |xy: SVector<DualSVec64<3>, 3>| SVector::from([
///                      xy[0] * xy[1].powi(3) * xy[2],
///                      xy[0].powi(2) * xy[1] * xy[2].powi(2)
///                     ]);
/// let (f, jac) = jacobian(fun, xy);
/// assert_eq!(f[0], 270.0);          // xy³z
/// assert_eq!(f[1], 300.0);          // x²yz²
/// assert_eq!(jac[(0,0)], 54.0);     // y³z
/// assert_eq!(jac[(0,1)], 270.0);    // 3xy²z
/// assert_eq!(jac[(0,2)], 135.0);    // xy³
/// assert_eq!(jac[(1,0)], 120.0);    // 2xyz²
/// assert_eq!(jac[(1,1)], 100.0);    // x²z²
/// assert_eq!(jac[(1,2)], 300.0);     // 2x²yz
/// ```
pub fn jacobian<G, T: DualNum<F>, F: DualNumFloat, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, N>,
) -> (OVector<T, M>, OMatrix<T, M, N>)
where
    G: FnOnce(OVector<DualVec<T, F, N>, N>) -> OVector<DualVec<T, F, N>, M>,
    DefaultAllocator: Allocator<DualVec<T, F, N>, M>
        + Allocator<T, M>
        + Allocator<T, N>
        + Allocator<T, M, N>
        + Allocator<T, nalgebra::Const<1>, N>
        + Allocator<DualVec<T, F, N>, N>
        + Allocator<OMatrix<T, U1, N>, M>,
{
    try_jacobian(|x| Ok::<_, Infallible>(g(x)), x).unwrap()
}

/// Variant of [jacobian] for fallible functions.
#[allow(clippy::type_complexity)]
pub fn try_jacobian<G, T: DualNum<F>, F: DualNumFloat, E, M: Dim, N: Dim>(
    g: G,
    x: OVector<T, N>,
) -> Result<(OVector<T, M>, OMatrix<T, M, N>), E>
where
    G: FnOnce(OVector<DualVec<T, F, N>, N>) -> Result<OVector<DualVec<T, F, N>, M>, E>,
    DefaultAllocator: Allocator<DualVec<T, F, N>, M>
        + Allocator<T, M>
        + Allocator<T, N>
        + Allocator<T, M, N>
        + Allocator<T, nalgebra::Const<1>, N>
        + Allocator<DualVec<T, F, N>, N>
        + Allocator<OMatrix<T, U1, N>, M>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative_generic(r, c, i);
    }
    g(x).map(|res| {
        let eps = OMatrix::from_rows(
            res.map(|res| res.eps.unwrap_generic(r, c).transpose())
                .as_slice(),
        );
        (res.map(|r| r.re), eps)
    })
}

/* chain rule */
impl<T: DualNum<F>, F: Float, D: Dim> DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, &self.eps * f1)
    }
}

/* product rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Mul<&'a DualVec<T, F, D>> for &'b DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    type Output = DualVec<T, F, D>;
    #[inline]
    fn mul(self, other: &DualVec<T, F, D>) -> Self::Output {
        DualVec::new(
            self.re.clone() * other.re.clone(),
            &self.eps * other.re.clone() + &other.eps * self.re.clone(),
        )
    }
}

/* quotient rule */
impl<'a, 'b, T: DualNum<F>, F: Float, D: Dim> Div<&'a DualVec<T, F, D>> for &'b DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    type Output = DualVec<T, F, D>;
    #[inline]
    fn div(self, other: &DualVec<T, F, D>) -> DualVec<T, F, D> {
        let inv = other.re.recip();
        DualVec::new(
            self.re.clone() * inv.clone(),
            (&self.eps * other.re.clone() - &other.eps * self.re.clone()) * inv.clone() * inv,
        )
    }
}

/* string conversions */
impl<T: DualNum<F>, F, D: Dim> fmt::Display for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.re)?;
        self.eps.fmt(f, "ε")
    }
}

impl_first_derivatives!(DualVec, [eps], [D]);
impl_dual!(DualVec, [eps], [D]);

/**
 * The SimdValue trait is for rearranging data into a form more suitable for Simd,
 * and rearranging it back into a usable form. It is not documented particularly well.
 *
 * The primary job of this SimdValue impl is to allow people to use `simba::simd::f32x4` etc,
 * instead of f32/f64. Those types implement nalgebra::SimdRealField/ComplexField, so they
 * behave like scalars. When we use them, we would have `DualVec<f32x4, f32, N>` etc, with our
 * F parameter set to <T as SimdValue>::Element. We will need to be able to split up that type
 * into four of DualVec in order to get out of simd-land. That's what the SimdValue trait is for.
 *
 * Ultimately, someone will have to to implement SimdRealField on DualVec and call the
 * simd_ functions of <T as SimdRealField>. That's future work for someone who finds
 * num_dual is not fast enough.
 *
 * Unfortunately, doing anything with SIMD is blocked on
 * <https://github.com/dimforge/simba/issues/44>.
 *
 */
impl<T, D: Dim> nalgebra::SimdValue for DualVec<T, T::Element, D>
where
    DefaultAllocator: Allocator<T, D> + Allocator<T::Element, D>,
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
    type Element = DualVec<T::Element, T::Element, D>;
    type SimdBool = T::SimdBool;

    #[inline]
    fn lanes() -> usize {
        T::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        // Need to make `lanes` copies of each of:
        // - the real part
        // - each of the N epsilon parts
        let re = T::splat(val.re);
        let eps = Derivative::splat(val.eps);
        Self {
            re,
            eps,
            f: PhantomData,
        }
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
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

/// Comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialEq, F: Float, D: Dim> PartialEq for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + PartialOrd, F: Float, D: Dim> PartialOrd for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}
/// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
/// same execution path as real-valued code would.
impl<T: DualNum<F> + approx::AbsDiffEq<Epsilon = T>, F: Float, D: Dim> approx::AbsDiffEq
    for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
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
    for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
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
impl<T: DualNum<F> + UlpsEq<Epsilon = T>, F: Float, D: Dim> UlpsEq for DualVec<T, F, D>
where
    DefaultAllocator: Allocator<T, D>,
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

impl<T, D: Dim> nalgebra::Field for DualVec<T, T::Element, D>
where
    T: DualNum<T::Element> + SimdValue,
    T::Element: DualNum<T::Element> + Scalar + Float,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, U1, D> + Allocator<T, D, U1> + Allocator<T, D, D>,
    DefaultAllocator: Allocator<T::Element, D>
        + Allocator<T::Element, U1, D>
        + Allocator<T::Element, D, U1>
        + Allocator<T::Element, D, D>,
{
}

use simba::scalar::{SubsetOf, SupersetOf};

impl<TSuper, FSuper, T, F, D: Dim> SubsetOf<DualVec<TSuper, FSuper, D>> for DualVec<T, F, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<T>,
    T: DualNum<F>,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, U1, D> + Allocator<T, D, U1> + Allocator<T, D, D>,
    DefaultAllocator: Allocator<TSuper, D>
        + Allocator<TSuper, U1, D>
        + Allocator<TSuper, D, U1>
        + Allocator<TSuper, D, D>,
{
    #[inline(always)]
    fn to_superset(&self) -> DualVec<TSuper, FSuper, D> {
        let re = TSuper::from_subset(&self.re);
        let eps = Derivative::from_subset(&self.eps);
        DualVec {
            re,
            eps,
            f: PhantomData,
        }
    }
    #[inline(always)]
    fn from_superset(element: &DualVec<TSuper, FSuper, D>) -> Option<Self> {
        let re = TSuper::to_subset(&element.re)?;
        let eps = Derivative::to_subset(&element.eps)?;
        Some(Self {
            re,
            eps,
            f: PhantomData,
        })
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &DualVec<TSuper, FSuper, D>) -> Self {
        let re = TSuper::to_subset_unchecked(&element.re);
        let eps = Derivative::to_subset_unchecked(&element.eps);
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
    #[inline(always)]
    fn is_in_subset(element: &DualVec<TSuper, FSuper, D>) -> bool {
        TSuper::is_in_subset(&element.re)
            && <Derivative<_, _, _, _> as SupersetOf<Derivative<_, _, _, _>>>::is_in_subset(
                &element.eps,
            )
    }
}

impl<TSuper, FSuper, D: Dim> SupersetOf<f32> for DualVec<TSuper, FSuper, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<f32>,
    DefaultAllocator: Allocator<TSuper, D>
        + Allocator<TSuper, U1, D>
        + Allocator<TSuper, D, U1>
        + Allocator<TSuper, D, D>,
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
        let eps = Derivative::none();
        Self {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<TSuper, FSuper, D: Dim> SupersetOf<f64> for DualVec<TSuper, FSuper, D>
where
    TSuper: DualNum<FSuper> + SupersetOf<f64>,
    DefaultAllocator: Allocator<TSuper, D>
        + Allocator<TSuper, U1, D>
        + Allocator<TSuper, D, U1>
        + Allocator<TSuper, D, D>,
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
        let eps = Derivative::none();
        Self {
            re,
            eps,
            f: PhantomData,
        }
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
impl<T, D: Dim> ComplexField for DualVec<T, T::Element, D>
where
    T: DualNum<T::Element> + SupersetOf<T> + AbsDiffEq<Epsilon = T> + Sync + Send,
    T::Element: DualNum<T::Element> + Scalar + DualNumFloat + Sync + Send,
    T: SupersetOf<T::Element>,
    T: SupersetOf<f64>,
    T: SimdPartialOrd + PartialOrd,
    T: SimdValue<Element = T, SimdBool = bool>,
    T: RelativeEq + UlpsEq + AbsDiffEq,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, U1, D> + Allocator<T, D, U1> + Allocator<T, D, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Sync + Send,
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

impl<T, D: Dim> RealField for DualVec<T, T::Element, D>
where
    T: DualNum<T::Element> + SupersetOf<T> + Sync + Send,
    T::Element: DualNum<T::Element> + Scalar + DualNumFloat,
    T: SupersetOf<T::Element>,
    T: SupersetOf<f64>,
    T: SimdPartialOrd + PartialOrd,
    T: RelativeEq + AbsDiffEq<Epsilon = T>,
    T: SimdValue<Element = T, SimdBool = bool>,
    T: UlpsEq,
    T: AbsDiffEq,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, U1, D> + Allocator<T, D, U1> + Allocator<T, D, D>,
    <DefaultAllocator as Allocator<T, D>>::Buffer: Sync + Send,
{
    #[inline]
    fn copysign(self, _sign: Self) -> Self {
        todo!("copysign not yet implemented on dual numbers")
    }

    #[inline]
    fn atan2(self, _other: Self) -> Self {
        todo!("atan2 not yet implemented on dual numbers")
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
