macro_rules! impl_from_f2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F, $($dim: Dim,)*> From<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T$(, $dim)*>,
        {
            #[inline]
            fn from(float: F) -> Self {
                Self::from_re(T::from(float))
            }
        }
    };
}

macro_rules! impl_zero_one2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Zero for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T$(, $dim)*>,
        {
            #[inline]
            fn zero() -> Self {
                Self::from_re(T::zero())
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.re.is_zero()
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> One for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T$(, $dim)*>,
        {
            #[inline]
            fn one() -> Self {
                Self::from_re(T::one())
            }

            #[inline]
            fn is_one(&self) -> bool {
                self.re.is_one()
            }
        }
    };
}

macro_rules! impl_add_sub_rem2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl<'a, 'b, T: DualNum<F>, F: Float, $($dim: Dim,)*> Add<&'a $struct<T, F$(, $dim)*>>
            for &'b $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn add(self, other: &$struct<T, F$(, $dim)*>) -> $struct<T, F$(, $dim)*> {
                Self::Output::new(self.re.clone() + &other.re, $(&self.$im + &other.$im,)*)
            }
        }

        impl<'a, 'b, T: DualNum<F>, F: Float, $($dim: Dim,)*> Sub<&'a $struct<T, F$(, $dim)*>>
            for &'b $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn sub(self, other: &$struct<T, F$(, $dim)*>) -> $struct<T, F$(, $dim)*> {
                Self::Output::new(self.re.clone() - &other.re, $(&self.$im - &other.$im,)*)
            }
        }

        impl<'a, 'b, T: DualNum<F>, F, $($dim: Dim,)*> Rem<&'a $struct<T, F$(, $dim)*>> for &'b $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn rem(self, _other: &$struct<T, F$(, $dim)*>) -> $struct<T, F$(, $dim)*> {
                unimplemented!()
            }
        }
    };
}

macro_rules! forward_binop2 {
    ($struct:ident, [$($dim:tt),*], $trt:ident, $operator:tt, $mth:ident) => {
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> $trt<$struct<T, F$(, $dim)*>> for &$struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F$(, $dim)*>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> $trt<&$struct<T, F$(, $dim)*>> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn $mth(self, rhs: &$struct<T, F$(, $dim)*>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> $trt for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F$(, $dim)*>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

macro_rules! impl_neg2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Neg for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                -&self
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Neg for &$struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = $struct<T, F$(, $dim)*>;
            #[inline]
            fn neg(self) -> Self::Output {
                <$struct<T, F$(, $dim)*>>::new(-self.re.clone(), $(-&self.$im),*)
            }
        }
    };
}

macro_rules! impl_assign_ops2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> MulAssign for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self = self.clone() * other;
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> DivAssign for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                *self = self.clone() / other;
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> AddAssign for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.re += other.re;
                $(self.$im += other.$im;)*
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> SubAssign for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.re -= other.re;
                $(self.$im -= other.$im;)*
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> RemAssign for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn rem_assign(&mut self, _other: Self) {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_scalar_op2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> Mul<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn mul(mut self, other: F) -> Self {
                self *= other;
                self
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> MulAssign<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn mul_assign(&mut self, other: F) {
                self.re *= other;
                $(self.$im *= T::from(other);)*
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> Div<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn div(mut self, other: F) -> Self {
                self /= other;
                self
            }
        }

        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> DivAssign<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn div_assign(&mut self, other: F) {
                self.re /= other;
                $(self.$im /= T::from(other);)*
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Add<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn add(mut self, other: F) -> Self {
                self.re += other;
                self
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> AddAssign<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn add_assign(&mut self, other: F)  {
                self.re += other;
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Sub<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn sub(mut self, other: F) -> Self {
                self.re -= other;
                self
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> SubAssign<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn sub_assign(&mut self, other: F)  {
                self.re -= other;
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Rem<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn rem(self, _other: F) -> Self {
                unimplemented!()
            }
        }

        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> RemAssign<F> for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn rem_assign(&mut self, _other: F) {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_inv2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> Inv for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            type Output = Self;
            #[inline]
            fn inv(self) -> Self {
                self.recip()
            }
        }
    };
}

macro_rules! impl_iterator2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Sum for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::zero(), |acc, c| acc + c)
            }
        }

        impl<'a, T: DualNum<F>, F: Float, $($dim: Dim,)*> Sum<&'a $struct<T, F$(, $dim)*>>
            for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a $struct<T, F$(, $dim)*>>,
            {
                iter.fold(Self::zero(), |acc, c| acc + c)
            }
        }
        impl<T: DualNum<F>, F: Float, $($dim: Dim,)*> Product for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::one(), |acc, c| acc * c)
            }
        }
        impl<'a, T: DualNum<F>, F: Float, $($dim: Dim,)*> Product<&'a $struct<T, F$(, $dim)*>>
            for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a $struct<T, F$(, $dim)*>>,
            {
                iter.fold(Self::one(), |acc, c| acc * c)
            }
        }
    };
}

macro_rules! impl_from_primitive2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: Float + FromPrimitive, $($dim: Dim,)*> FromPrimitive for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>,
        {
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

macro_rules! impl_signed2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: DualNumFloat, $($dim: Dim,)*> Signed for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>
        {
            #[inline]
            fn abs(&self) -> Self {
                if self.is_positive() {
                    self.clone()
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
                self.re.is_positive()
            }

            #[inline]
            fn is_negative(&self) -> bool {
                self.re.is_negative()
            }
        }
    };
}

macro_rules! impl_float_const2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F>, F: Float + FloatConst, $($dim: Dim,)*> FloatConst for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>
        {
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

macro_rules! impl_num2 {
    ($struct:ident, [$($dim:tt),*]) => {
        impl<T: DualNum<F> + Signed, F: Float, $($dim: Dim,)*> Num for $struct<T, F$(, $dim)*>
        where
            DefaultAllocator: Allocator<T, D>
        {
            type FromStrRadixErr = F::FromStrRadixErr;
            #[inline]
            fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                unimplemented!()
            }
        }
    };
}

macro_rules! impl_dual2 {
    ($struct:ident, [$($dim:tt),*], [$($im:ident),*]) => {
        impl_from_f2!($struct, [$($dim),*], [$($im),*]);
        impl_zero_one2!($struct, [$($dim),*]);
        impl_add_sub_rem2!($struct, [$($dim),*], [$($im),*]);
        forward_binop2!($struct, [$($dim),*], Add, +, add);
        forward_binop2!($struct, [$($dim),*], Sub, -, sub);
        forward_binop2!($struct, [$($dim),*], Mul, *, mul);
        forward_binop2!($struct, [$($dim),*], Div, /, div);
        forward_binop2!($struct, [$($dim),*], Rem, %, rem);
        impl_neg2!($struct, [$($dim),*], [$($im),*]);
        impl_assign_ops2!($struct, [$($dim),*], [$($im),*]);
        impl_scalar_op2!($struct, [$($dim),*], [$($im),*]);
        impl_inv2!($struct, [$($dim),*]);
        impl_iterator2!($struct, [$($dim),*]);
        impl_from_primitive2!($struct, [$($dim),*]);
        impl_signed2!($struct, [$($dim),*]);
        impl_num2!($struct, [$($dim),*]);
        impl_float_const2!($struct, [$($dim),*]);
    };
}
