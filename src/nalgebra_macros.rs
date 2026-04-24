macro_rules! impl_approx {
    ($struct:ident$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        /// Like PartialEq, comparisons are only made based on the real part. This allows the code to follow the
        /// same execution path as real-valued code would.
        impl<T: DualNum<F> + approx::AbsDiffEq<Epsilon = T>, F$($(, $dim: Dim)*)?> approx::AbsDiffEq
            for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
        {
            // Would make much more sense to have type Epsilon = F, but there is a trait bound in nalgebra::RealField that requires Self...
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
        impl<T: DualNum<F> + approx::RelativeEq<Epsilon = T>, F$($(, $dim: Dim)*)?> approx::RelativeEq
            for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
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

        impl<T: DualNum<F> + approx::UlpsEq<Epsilon = T>, F$($(, $dim: Dim)*)?> approx::UlpsEq for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
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
    };
}

macro_rules! impl_simd_value {
    ($struct:ident, [$($im:ident),*]$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
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
        impl<T$($(, $dim: Dim)*)?> nalgebra::SimdValue for $struct<T, T::Element$($(, $dim)*)?>
        where
            T: DualNum<T::Element> + nalgebra::SimdValue + nalgebra::Scalar,
            T::Element: DualNum<T::Element> + nalgebra::Scalar,
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
        {
            // Say T = simba::f32x4. T::Element is f32. T::SimdBool is AutoSimd<[bool; 4]>.
            // AutoSimd<[f32; 4]> stores an actual [f32; 4], i.e. four floats in one slot.
            // So our Dual<AutoSimd<[f32; 4], f32, N> has 4 * (1+N) floats in it, stored in blocks of
            // four. When we want to do any math on it but ignore its f32x4 storage mode, we need to break
            // that type into FOUR of Dual<f32, f32, N>; then we do math on it, then we bring it back
            // together.
            //
            // Hence this definition of Element:
            type Element = $struct<T::Element, T::Element$($(, $dim)*)?>;
            type SimdBool = T::SimdBool;

            const LANES: usize = T::LANES;

            #[inline]
            fn splat(val: Self::Element) -> Self {
                // Need to make `lanes` copies of each of:
                // - the real part
                // - each of the N epsilon parts
                Self::new(T::splat(val.re), $(nalgebra::SimdValue::splat(val.$im),)*)
            }

            #[inline]
            fn extract(&self, i: usize) -> Self::Element {
                Self::Element::new(self.re.extract(i), $(self.$im.extract(i),)*)
            }

            #[inline]
            unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
                unsafe {
                    Self::Element::new(self.re.extract_unchecked(i), $(self.$im.extract_unchecked(i),)*)
                }
            }

            #[inline]
            fn replace(&mut self, i: usize, val: Self::Element) {
                self.re.replace(i, val.re);
                $(self.$im.replace(i, val.$im);)*
            }

            #[inline]
            unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
                unsafe { self.re.replace_unchecked(i, val.re) };
                $(unsafe { self.$im.replace_unchecked(i, val.$im) };)*
            }

            #[inline]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                Self::new(self.re.select(cond, other.re), $(self.$im.select(cond, other.$im),)*)
            }
        }
    };
}

macro_rules! impl_subset {
    ($struct:ident$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        // one of the weirdest things among the nalgebra trait bounds
        impl<T: DualNum<F> + Clone, F: Clone$($(, $dim: Dim)*)?> simba::scalar::SubsetOf<Self> for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
        {
            #[inline(always)]
            fn to_superset(&self) -> Self {
                self.clone()
            }
            #[inline(always)]
            fn from_superset(element: &Self) -> Option<Self> {
                Some(element.clone())
            }
            #[inline(always)]
            fn from_superset_unchecked(element: &Self) -> Self {
                element.clone()
            }
            #[inline(always)]
            fn is_in_subset(_: &Self) -> bool {
                true
            }
        }
    };
}

macro_rules! impl_superset {
    ($struct:ident$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        impl<T: DualNum<F> + simba::scalar::SupersetOf<f32>, F$($(, $dim: Dim)*)?> simba::scalar::SupersetOf<f32>
            for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
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
                Self::from_re(T::from_subset(element))
            }
        }

        impl<T: DualNum<F> + simba::scalar::SupersetOf<f64>, F$($(, $dim: Dim)*)?> simba::scalar::SupersetOf<f64>
            for $struct<T, F$($(, $dim)*)?>
        where
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
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
                Self::from_re(T::from_subset(element))
            }
        }
    };
}

macro_rules! impl_complex_field {
    ($struct:ident$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        impl<T: DualNum<T::Element>$($(, $dim: Dim)*)?> nalgebra::Field for $struct<T, T::Element$($(, $dim)*)?>
        where
            T: nalgebra::SimdValue,
            T::Element: DualNum<T::Element> + nalgebra::Scalar + Float,
            $($(DefaultAllocator: Allocator<$($ddim,)*>),*)?
        {}

        // This impl is modelled on `impl ComplexField for f32`. The imaginary part is nothing.
        impl<T: DualNum<T::Element>$($(, $dim: Dim)*)?> nalgebra::ComplexField for $struct<T, T::Element$($(, $dim)*)?>
        where
            T: nalgebra::Scalar + DualNumFloat,
            T: simba::scalar::SupersetOf<T>,
            T: simba::scalar::SupersetOf<f32> + simba::scalar::SupersetOf<f64>,
            T: nalgebra::SimdValue<Element = T, SimdBool = bool>,
            T: approx::RelativeEq<Epsilon = T> + approx::UlpsEq + approx::AbsDiffEq,
            $($(DefaultAllocator: Allocator<$($ddim,)*>,)*
            $(<DefaultAllocator as Allocator<$($ddim,)*>>::Buffer<T>: Sync + Send),*)?
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

            #[inline]
            fn signum(self) -> Self {
                Signed::signum(&self)
            }
        }
    };
}

macro_rules! impl_real_field {
    ($struct:ident$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        impl<T: DualNum<T::Element>$($(, $dim: Dim)*)?> nalgebra::RealField for $struct<T, T::Element$($(, $dim)*)?>
        where
            T: nalgebra::Scalar + DualNumFloat,
            T: simba::scalar::SupersetOf<T>,
            T: simba::scalar::SupersetOf<f32> + simba::scalar::SupersetOf<f64>,
            T: nalgebra::SimdValue<Element = T, SimdBool = bool>,
            T: approx::RelativeEq<Epsilon = T> + approx::UlpsEq + approx::AbsDiffEq,
            $($(DefaultAllocator: Allocator<$($ddim,)*>,)*
            $(<DefaultAllocator as Allocator<$($ddim,)*>>::Buffer<T>: Sync + Send),*)?
        {
            #[inline]
            fn copysign(self, sign: Self) -> Self {
                if sign.re.is_sign_positive() {
                    nalgebra::SimdComplexField::simd_abs(self)
                } else {
                    -nalgebra::SimdComplexField::simd_abs(self)
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
                Self::from_re(<T as FloatConst>::FRAC_PI_2())
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
    };
}

#[macro_export]
macro_rules! impl_nalgebra {
    ($struct:ident, [$($im:ident),*]$(, [$($dim:tt),*]$(, [$($ddim:tt),*])*)?) => {
        impl_approx!($struct$(, [$($dim),*]$(, [$($ddim),*])*)?);
        impl_simd_value!($struct, [$($im),*]$(, [$($dim),*]$(, [$($ddim),*])*)?);
        impl_subset!($struct$(, [$($dim),*]$(, [$($ddim),*])*)?);
        impl_superset!($struct$(, [$($dim),*]$(, [$($ddim),*])*)?);
        impl_complex_field!($struct$(, [$($dim),*]$(, [$($ddim),*])*)?);
        impl_real_field!($struct$(, [$($dim),*]$(, [$($ddim),*])*)?);
    };
}
