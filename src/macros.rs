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