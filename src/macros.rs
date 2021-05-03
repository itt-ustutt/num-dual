#[macro_use]

macro_rules! impl_from_f {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<T: Copy + Zero + AddAssign + From<F>, F, $(const $const: usize,)*> From<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn from(float: F) -> Self {
                Self::from_re(T::from(float))
            }
        }
    };
}

macro_rules! impl_zero_one {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> Zero for $struct<T, F$(, $const)*> {
            #[inline]
            fn zero() -> Self {
                Self::from_re(T::zero())
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.re.is_zero() && $(self.$im.is_zero()) &&*
            }
        }

        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> One for $struct<T, F$(, $const)*> {
            #[inline]
            fn one() -> Self {
                Self::from_re(T::one())
            }

            #[inline]
            fn is_one(&self) -> bool {
                self.re.is_one() && $(self.$im.is_zero()) &&*
            }
        }
    };
}

macro_rules! impl_add_sub_rem {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<'a, 'b, T: DualNum<F>, F: Float, $(const $const: usize,)*> Add<&'a $struct<T, F$(, $const)*>>
            for &'b $struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn add(self, other: &$struct<T, F$(, $const)*>) -> $struct<T, F$(, $const)*> {
                Self::Output::new(self.re + other.re, $(self.$im + other.$im,)*)
            }
        }

        impl<'a, 'b, T: DualNum<F>, F: Float, $(const $const: usize,)*> Sub<&'a $struct<T, F$(, $const)*>>
            for &'b $struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn sub(self, other: &$struct<T, F$(, $const)*>) -> $struct<T, F$(, $const)*> {
                Self::Output::new(self.re - other.re, $(self.$im - other.$im,)*)
            }
        }

        impl<'a, 'b, T, F, $(const $const: usize,)*> Rem<&'a $struct<T, F$(, $const)*>> for &'b $struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn rem(self, _other: &$struct<T, F$(, $const)*>) -> $struct<T, F$(, $const)*> {
                unimplemented!()
            }
        }
    };
}

macro_rules! forward_binop {
    ($struct:ident, [$($const:tt),*], $trt:ident, $operator:tt, $mth:ident) => {
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> $trt<$struct<T, F$(, $const)*>> for &$struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F$(, $const)*>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> $trt<&$struct<T, F$(, $const)*>> for $struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn $mth(self, rhs: &$struct<T, F$(, $const)*>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> $trt for $struct<T, F$(, $const)*>
        {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F$(, $const)*>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

macro_rules! impl_neg {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<T: Copy + Neg<Output = T>, F: Float, $(const $const: usize,)*> Neg for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                -&self
            }
        }

        impl<T: Copy + Neg<Output = T>, F: Float, $(const $const: usize,)*> Neg for &$struct<T, F$(, $const)*> {
            type Output = $struct<T, F$(, $const)*>;
            #[inline]
            fn neg(self) -> Self::Output {
                <$struct<T, F$(, $const)*>>::new(-self.re, $(-self.$im),*)
            }
        }
    };
}

macro_rules! impl_assign_ops {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> MulAssign for $struct<T, F$(, $const)*>
        {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self = *self * other;
            }
        }

        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> DivAssign for $struct<T, F$(, $const)*>
        {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                *self = *self / other;
            }
        }

        impl<T: Copy + AddAssign, F: Float, $(const $const: usize,)*> AddAssign for $struct<T, F$(, $const)*> {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.re += other.re;
                $(self.$im += other.$im;)*
            }
        }

        impl<T: Copy + SubAssign, F: Float, $(const $const: usize,)*> SubAssign for $struct<T, F$(, $const)*> {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.re -= other.re;
                $(self.$im -= other.$im;)*
            }
        }

        impl<T, F: Float, $(const $const: usize,)*> RemAssign for $struct<T, F$(, $const)*> {
            #[inline]
            fn rem_assign(&mut self, _other: Self) {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_scalar_op {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> Mul<F> for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn mul(mut self, other: F) -> Self {
                self.scale(other);
                self
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> MulAssign<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn mul_assign(&mut self, other: F) {
                self.scale(other);
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> Div<F> for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn div(mut self, other: F) -> Self {
                self.scale(other.recip());
                self
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> DivAssign<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn div_assign(&mut self, other: F) {
                self.scale(other.recip());
            }
        }

        impl<T: AddAssign<F>, F: Float, $(const $const: usize,)*> Add<F> for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn add(mut self, other: F) -> Self {
                self.re += other;
                self
            }
        }

        impl<T: AddAssign<F>, F: Float, $(const $const: usize,)*> AddAssign<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn add_assign(&mut self, other: F)  {
                self.re += other;
            }
        }

        impl<T: SubAssign<F>, F: Float, $(const $const: usize,)*> Sub<F> for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn sub(mut self, other: F) -> Self {
                self.re -= other;
                self
            }
        }

        impl<T: SubAssign<F>, F: Float, $(const $const: usize,)*> SubAssign<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn sub_assign(&mut self, other: F)  {
                self.re -= other;
            }
        }

        impl<T, F: Float, $(const $const: usize,)*> Rem<F> for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn rem(self, _other: F) -> Self {
                unimplemented!()
            }
        }

        impl<T, F: Float, $(const $const: usize,)*> RemAssign<F> for $struct<T, F$(, $const)*> {
            #[inline]
            fn rem_assign(&mut self, _other: F) {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_inv {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> Inv for $struct<T, F$(, $const)*> {
            type Output = Self;
            #[inline]
            fn inv(self) -> Self {
                self.recip()
            }
        }
    };
}

macro_rules! impl_iterator {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> Sum for $struct<T, F$(, $const)*> {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::zero(), |acc, c| acc + c)
            }
        }

        impl<'a, T: DualNum<F>, F: Float, $(const $const: usize,)*> Sum<&'a $struct<T, F$(, $const)*>>
            for $struct<T, F$(, $const)*>
        {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a $struct<T, F$(, $const)*>>,
            {
                iter.fold(Self::zero(), |acc, c| acc + c)
            }
        }
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> Product for $struct<T, F$(, $const)*> {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::one(), |acc, c| acc * c)
            }
        }
        impl<'a, T: DualNum<F>, F: Float, $(const $const: usize,)*> Product<&'a $struct<T, F$(, $const)*>>
            for $struct<T, F$(, $const)*>
        {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a $struct<T, F$(, $const)*>>,
            {
                iter.fold(Self::one(), |acc, c| acc * c)
            }
        }
    };
}

macro_rules! impl_from_primitive {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: Float + FromPrimitive, $(const $const: usize,)*> FromPrimitive for $struct<T, F$(, $const)*> {
            #[inline]
            fn from_isize(n: isize) -> Option<Self> {
                F::from_isize(n).map(|f| f.into())
            }

            #[inline]
            fn from_i8(n: i8) -> Option<Self> {
                F::from_i8(n).map(|f| f.into())
            }

            #[inline]
            fn from_i16(n: i16) -> Option<Self> {
                F::from_i16(n).map(|f| f.into())
            }

            #[inline]
            fn from_i32(n: i32) -> Option<Self> {
                F::from_i32(n).map(|f| f.into())
            }

            #[inline]
            fn from_i64(n: i64) -> Option<Self> {
                F::from_i64(n).map(|f| f.into())
            }

            #[inline]
            fn from_i128(n: i128) -> Option<Self> {
                F::from_i128(n).map(|f| f.into())
            }

            #[inline]
            fn from_usize(n: usize) -> Option<Self> {
                F::from_usize(n).map(|f| f.into())
            }

            #[inline]
            fn from_u8(n: u8) -> Option<Self> {
                F::from_u8(n).map(|f| f.into())
            }

            #[inline]
            fn from_u16(n: u16) -> Option<Self> {
                F::from_u16(n).map(|f| f.into())
            }

            #[inline]
            fn from_u32(n: u32) -> Option<Self> {
                F::from_u32(n).map(|f| f.into())
            }

            #[inline]
            fn from_u64(n: u64) -> Option<Self> {
                F::from_u64(n).map(|f| f.into())
            }

            #[inline]
            fn from_u128(n: u128) -> Option<Self> {
                F::from_u128(n).map(|f| f.into())
            }

            #[inline]
            fn from_f32(n: f32) -> Option<Self> {
                F::from_f32(n).map(|f| f.into())
            }

            #[inline]
            fn from_f64(n: f64) -> Option<Self> {
                F::from_f64(n).map(|f| f.into())
            }
        }
    };
}

macro_rules! impl_signed {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $(const $const: usize,)*> Signed for $struct<T, F$(, $const)*> {
            #[inline]
            fn abs(&self) -> Self {
                if self.is_positive() {
                    *self
                } else {
                    -self
                }
            }

            #[inline]
            fn abs_sub(&self, other: &Self) -> Self {
                if self.re() > other.re() {
                    self - other
                } else {
                    Self::zero()
                }
            }

            #[inline]
            fn signum(&self) -> Self {
                if self.is_positive() {
                    Self::one()
                } else if self.is_zero() {
                    Self::zero()
                } else {
                    -Self::one()
                }
            }

            #[inline]
            fn is_positive(&self) -> bool {
                self.re().is_positive()
            }

            #[inline]
            fn is_negative(&self) -> bool {
                self.re().is_negative()
            }
        }
    };
}

macro_rules! impl_float_const {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F>, F: Float + FloatConst, $(const $const: usize,)*> FloatConst for $struct<T, F$(, $const)*> {
            fn E() -> Self {
                Self::from(F::E())
            }

            fn FRAC_1_PI() -> Self {
                Self::from(F::FRAC_1_PI())
            }

            fn FRAC_1_SQRT_2() -> Self {
                Self::from(F::FRAC_1_SQRT_2())
            }

            fn FRAC_2_PI() -> Self {
                Self::from(F::FRAC_2_PI())
            }

            fn FRAC_2_SQRT_PI() -> Self {
                Self::from(F::FRAC_2_SQRT_PI())
            }

            fn FRAC_PI_2() -> Self {
                Self::from(F::FRAC_PI_2())
            }

            fn FRAC_PI_3() -> Self {
                Self::from(F::FRAC_PI_3())
            }

            fn FRAC_PI_4() -> Self {
                Self::from(F::FRAC_PI_4())
            }

            fn FRAC_PI_6() -> Self {
                Self::from(F::FRAC_PI_6())
            }

            fn FRAC_PI_8() -> Self {
                Self::from(F::FRAC_PI_8())
            }

            fn LN_10() -> Self {
                Self::from(F::LN_10())
            }

            fn LN_2() -> Self {
                Self::from(F::LN_2())
            }

            fn LOG10_E() -> Self {
                Self::from(F::LOG10_E())
            }

            fn LOG2_E() -> Self {
                Self::from(F::LOG2_E())
            }

            fn PI() -> Self {
                Self::from(F::PI())
            }

            fn SQRT_2() -> Self {
                Self::from(F::SQRT_2())
            }
        }
    };
}

macro_rules! impl_num {
    ($struct:ident, [$($const:tt),*]) => {
        impl<T: DualNum<F> + Signed, F: Float, $(const $const: usize,)*> Num for $struct<T, F$(, $const)*> {
            type FromStrRadixErr = F::FromStrRadixErr;
            #[inline]
            fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_dual {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl_from_f!($struct, [$($const),*], [$($im),*]);
        impl_zero_one!($struct, [$($const),*], [$($im),*]);
        impl_add_sub_rem!($struct, [$($const),*], [$($im),*]);
        forward_binop!($struct, [$($const),*], Add, +, add);
        forward_binop!($struct, [$($const),*], Sub, -, sub);
        forward_binop!($struct, [$($const),*], Mul, *, mul);
        forward_binop!($struct, [$($const),*], Div, /, div);
        forward_binop!($struct, [$($const),*], Rem, %, rem);
        impl_neg!($struct, [$($const),*], [$($im),*]);
        impl_assign_ops!($struct, [$($const),*], [$($im),*]);
        impl_scalar_op!($struct, [$($const),*]);
        impl_inv!($struct, [$($const),*]);
        impl_iterator!($struct, [$($const),*]);
        impl_from_primitive!($struct, [$($const),*]);
        impl_signed!($struct, [$($const),*]);
        impl_num!($struct, [$($const),*]);
        impl_float_const!($struct, [$($const),*]);
    };
}
