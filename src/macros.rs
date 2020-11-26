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
