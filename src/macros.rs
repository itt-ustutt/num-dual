#[macro_use]

macro_rules! forward_binop {
    ($struct:ident, $trt:ident, $operator:tt, $mth:ident) => {
        impl<T: DualNum<F>, F: Float> $trt<$struct<T, F>> for &$struct<T, F>
        {
            type Output = $struct<T, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: DualNum<F>, F: Float> $trt<&$struct<T, F>> for $struct<T, F>
        {
            type Output = $struct<T, F>;
            #[inline]
            fn $mth(self, rhs: &$struct<T, F>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: DualNum<F>, F: Float> $trt for $struct<T, F>
        {
            type Output = $struct<T, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T, F>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

macro_rules! forward_binop_vec {
    ($struct:ident, $trt:ident, $operator:tt, $mth:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, F: Float> $trt<$struct<T0, T1, F>> for &$struct<T0, T1, F>
        {
            type Output = $struct<T0, T1, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T0, T1, F>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T0: DualNum<F>, T1: DualVec<T0, F>, F: Float> $trt<&$struct<T0, T1, F>> for $struct<T0, T1, F>
        {
            type Output = $struct<T0, T1, F>;
            #[inline]
            fn $mth(self, rhs: &$struct<T0, T1, F>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T0: DualNum<F>, T1: DualVec<T0, F>, F: Float> $trt for $struct<T0, T1, F>
        {
            type Output = $struct<T0, T1, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T0, T1, F>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

macro_rules! forward_binop_mat {
    ($struct:ident, $trt:ident, $operator:tt, $mth:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, T2: DualVec<T0, F>, F: Float> $trt<$struct<T0, T1, T2, F>> for &$struct<T0, T1, T2, F>
        where T1: OuterProduct<Output=T2>
        {
            type Output = $struct<T0, T1, T2, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T0, T1, T2, F>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T0: DualNum<F>, T1: DualVec<T0, F>, T2: DualVec<T0, F>, F: Float> $trt<&$struct<T0, T1, T2, F>> for $struct<T0, T1, T2, F>
        where T1: OuterProduct<Output=T2>
        {
            type Output = $struct<T0, T1, T2, F>;
            #[inline]
            fn $mth(self, rhs: &$struct<T0, T1, T2, F>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T0: DualNum<F>, T1: DualVec<T0, F>, T2: DualVec<T0, F>, F: Float> $trt for $struct<T0, T1, T2, F>
        where T1: OuterProduct<Output=T2>
        {
            type Output = $struct<T0, T1, T2, F>;
            #[inline]
            fn $mth(self, rhs: $struct<T0, T1, T2, F>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

macro_rules! impl_from_primitive {
    ($struct:ident) => {
        impl<T: DualNum<F>, F: Float + FromPrimitive> FromPrimitive for $struct<T, F> {
            fn from_isize(n: isize) -> Option<Self> {
                F::from_isize(n).map(|f| f.into())
            }

            fn from_i8(n: i8) -> Option<Self> {
                F::from_i8(n).map(|f| f.into())
            }

            fn from_i16(n: i16) -> Option<Self> {
                F::from_i16(n).map(|f| f.into())
            }

            fn from_i32(n: i32) -> Option<Self> {
                F::from_i32(n).map(|f| f.into())
            }

            fn from_i64(n: i64) -> Option<Self> {
                F::from_i64(n).map(|f| f.into())
            }

            fn from_i128(n: i128) -> Option<Self> {
                F::from_i128(n).map(|f| f.into())
            }

            fn from_usize(n: usize) -> Option<Self> {
                F::from_usize(n).map(|f| f.into())
            }

            fn from_u8(n: u8) -> Option<Self> {
                F::from_u8(n).map(|f| f.into())
            }

            fn from_u16(n: u16) -> Option<Self> {
                F::from_u16(n).map(|f| f.into())
            }

            fn from_u32(n: u32) -> Option<Self> {
                F::from_u32(n).map(|f| f.into())
            }

            fn from_u64(n: u64) -> Option<Self> {
                F::from_u64(n).map(|f| f.into())
            }

            fn from_u128(n: u128) -> Option<Self> {
                F::from_u128(n).map(|f| f.into())
            }

            fn from_f32(n: f32) -> Option<Self> {
                F::from_f32(n).map(|f| f.into())
            }

            fn from_f64(n: f64) -> Option<Self> {
                F::from_f64(n).map(|f| f.into())
            }
        }
    };
}

macro_rules! impl_from_primitive_vec {
    ($struct:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, F: Float + FromPrimitive> FromPrimitive
            for $struct<T0, T1, F>
        {
            fn from_isize(n: isize) -> Option<Self> {
                F::from_isize(n).map(|f| f.into())
            }

            fn from_i8(n: i8) -> Option<Self> {
                F::from_i8(n).map(|f| f.into())
            }

            fn from_i16(n: i16) -> Option<Self> {
                F::from_i16(n).map(|f| f.into())
            }

            fn from_i32(n: i32) -> Option<Self> {
                F::from_i32(n).map(|f| f.into())
            }

            fn from_i64(n: i64) -> Option<Self> {
                F::from_i64(n).map(|f| f.into())
            }

            fn from_i128(n: i128) -> Option<Self> {
                F::from_i128(n).map(|f| f.into())
            }

            fn from_usize(n: usize) -> Option<Self> {
                F::from_usize(n).map(|f| f.into())
            }

            fn from_u8(n: u8) -> Option<Self> {
                F::from_u8(n).map(|f| f.into())
            }

            fn from_u16(n: u16) -> Option<Self> {
                F::from_u16(n).map(|f| f.into())
            }

            fn from_u32(n: u32) -> Option<Self> {
                F::from_u32(n).map(|f| f.into())
            }

            fn from_u64(n: u64) -> Option<Self> {
                F::from_u64(n).map(|f| f.into())
            }

            fn from_u128(n: u128) -> Option<Self> {
                F::from_u128(n).map(|f| f.into())
            }

            fn from_f32(n: f32) -> Option<Self> {
                F::from_f32(n).map(|f| f.into())
            }

            fn from_f64(n: f64) -> Option<Self> {
                F::from_f64(n).map(|f| f.into())
            }
        }
    };
}

macro_rules! impl_from_primitive_mat {
    ($struct:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, T2: DualVec<T0, F>, F: Float + FromPrimitive>
            FromPrimitive for $struct<T0, T1, T2, F>
        {
            fn from_isize(n: isize) -> Option<Self> {
                F::from_isize(n).map(|f| f.into())
            }

            fn from_i8(n: i8) -> Option<Self> {
                F::from_i8(n).map(|f| f.into())
            }

            fn from_i16(n: i16) -> Option<Self> {
                F::from_i16(n).map(|f| f.into())
            }

            fn from_i32(n: i32) -> Option<Self> {
                F::from_i32(n).map(|f| f.into())
            }

            fn from_i64(n: i64) -> Option<Self> {
                F::from_i64(n).map(|f| f.into())
            }

            fn from_i128(n: i128) -> Option<Self> {
                F::from_i128(n).map(|f| f.into())
            }

            fn from_usize(n: usize) -> Option<Self> {
                F::from_usize(n).map(|f| f.into())
            }

            fn from_u8(n: u8) -> Option<Self> {
                F::from_u8(n).map(|f| f.into())
            }

            fn from_u16(n: u16) -> Option<Self> {
                F::from_u16(n).map(|f| f.into())
            }

            fn from_u32(n: u32) -> Option<Self> {
                F::from_u32(n).map(|f| f.into())
            }

            fn from_u64(n: u64) -> Option<Self> {
                F::from_u64(n).map(|f| f.into())
            }

            fn from_u128(n: u128) -> Option<Self> {
                F::from_u128(n).map(|f| f.into())
            }

            fn from_f32(n: f32) -> Option<Self> {
                F::from_f32(n).map(|f| f.into())
            }

            fn from_f64(n: f64) -> Option<Self> {
                F::from_f64(n).map(|f| f.into())
            }
        }
    };
}

macro_rules! impl_signed {
    ($struct:ident) => {
        impl<T: DualNum<F>, F: Float + Signed> Signed for $struct<T, F> {
            fn abs(&self) -> Self {
                if self.is_positive() {
                    *self
                } else {
                    -self
                }
            }

            fn abs_sub(&self, other: &Self) -> Self {
                if self.re() > other.re() {
                    self - other
                } else {
                    Self::zero()
                }
            }

            fn signum(&self) -> Self {
                if self.is_positive() {
                    Self::one()
                } else if self.is_zero() {
                    Self::zero()
                } else {
                    -Self::one()
                }
            }

            fn is_positive(&self) -> bool {
                self.re().is_positive()
            }

            fn is_negative(&self) -> bool {
                self.re().is_negative()
            }
        }
    };
}

macro_rules! impl_float_const {
    ($struct:ident) => {
        impl<T: DualNum<F>, F: Float + FloatConst> FloatConst for $struct<T, F> {
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

macro_rules! impl_signed_vec {
    ($struct:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, F: Float + Signed> Signed for $struct<T0, T1, F> {
            fn abs(&self) -> Self {
                if self.is_positive() {
                    *self
                } else {
                    -self
                }
            }

            fn abs_sub(&self, other: &Self) -> Self {
                if self.re() > other.re() {
                    self - other
                } else {
                    Self::zero()
                }
            }

            fn signum(&self) -> Self {
                if self.is_positive() {
                    Self::one()
                } else if self.is_zero() {
                    Self::zero()
                } else {
                    -Self::one()
                }
            }

            fn is_positive(&self) -> bool {
                self.re().is_positive()
            }

            fn is_negative(&self) -> bool {
                self.re().is_negative()
            }
        }
    };
}

macro_rules! impl_signed_mat {
    ($struct:ident) => {
        impl<T0: DualNum<F>, T1: DualVec<T0, F>, T2: DualVec<T0, F>, F: Float + Signed> Signed
            for $struct<T0, T1, T2, F>
        where
            T1: OuterProduct<Output = T2>,
        {
            fn abs(&self) -> Self {
                if self.is_positive() {
                    *self
                } else {
                    -self
                }
            }

            fn abs_sub(&self, other: &Self) -> Self {
                if self.re() > other.re() {
                    self - other
                } else {
                    Self::zero()
                }
            }

            fn signum(&self) -> Self {
                if self.is_positive() {
                    Self::one()
                } else if self.is_zero() {
                    Self::zero()
                } else {
                    -Self::one()
                }
            }

            fn is_positive(&self) -> bool {
                self.re().is_positive()
            }

            fn is_negative(&self) -> bool {
                self.re().is_negative()
            }
        }
    };
}
