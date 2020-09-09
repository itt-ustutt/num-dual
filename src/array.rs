use crate::{Dual, HyperDual, HD3};
use ndarray::*;
// use num_traits::Inv;
// use std::ops::*;

impl<T: Clone + 'static> ScalarOperand for Dual<T> {}
impl<T: Clone + 'static> ScalarOperand for HyperDual<T> {}
impl<T: Clone + 'static> ScalarOperand for HD3<T> {}

// macro_rules! impl_scalar_lhs_op {
//     ($scalar:ty, $operator:tt, $trt:ident, $mth:ident, $trt2:ident, $mth2:ident, $trt_inv:ident, $inv:ident) => {
//         impl<'a, S, D> $trt<&'a ArrayBase<S, D>> for $scalar
//         where
//             S: Data,
//             S::Elem: $trt<$scalar, Output = $scalar> + Copy,
//             D: Dimension,
//         {
//             type Output = Array<$scalar, D>;
//             fn $mth(self, rhs: &ArrayBase<S, D>) -> Array<$scalar, D> {
//                 rhs.map(|elt| *elt $operator self)
//             }
//         }

//         impl<'a, S, D> $trt2<&'a ArrayBase<S, D>> for $scalar
//         where
//             S: Data,
//             S::Elem: $trt<$scalar, Output = $scalar> + $trt_inv<Output = S::Elem> + Copy,
//             D: Dimension,
//         {
//             type Output = Array<$scalar, D>;
//             fn $mth2(self, rhs: &ArrayBase<S, D>) -> Array<$scalar, D> {
//                 rhs.map(|elt| elt.$inv() $operator self)
//             }
//         }

//         impl<S, D> $trt<ArrayBase<S, D>> for $scalar
//         where
//             S: Data,
//             S::Elem: $trt<$scalar, Output = $scalar> + Clone,
//             D: Dimension,
//         {
//             type Output = Array<$scalar, D>;
//             fn $mth(self, rhs: ArrayBase<S, D>) -> Array<$scalar, D> {
//                 rhs.mapv(|elt| elt $operator self)
//             }
//         }

//         impl<S, D> $trt2<ArrayBase<S, D>> for $scalar
//         where
//             S: Data,
//             S::Elem: $trt<$scalar, Output = $scalar> + $trt_inv<Output = S::Elem> + Clone,
//             D: Dimension,
//         {
//             type Output = Array<$scalar, D>;
//             fn $mth2(self, rhs: ArrayBase<S, D>) -> Array<$scalar, D> {
//                 rhs.mapv(|elt| elt.$inv() $operator self)
//             }
//         }
//     };
// }

// impl_scalar_lhs_op!(HyperDual64, *, Mul, mul, Div, div, Inv, inv);
// impl_scalar_lhs_op!(HyperDual64, +, Add, add, Sub, sub, Neg, neg);

// impl_scalar_lhs_op!(HyperDual32, *, Mul, mul, Div, div, Inv, inv);
// impl_scalar_lhs_op!(HyperDual32, +, Add, add, Sub, sub, Neg, neg);

#[macro_export]
macro_rules! hd64 {
    ($x:expr) => {
        HyperDual64::from($x)
    };
}
