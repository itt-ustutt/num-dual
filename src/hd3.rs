use crate::DualNumMethods;
use num_traits::{Float, Inv, One, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct HD3<T>(pub [T; 4]);

pub type HD3_32 = HD3<f32>;
pub type HD3_64 = HD3<f64>;

impl<T: Float> HD3<T> {
    #[inline]
    pub fn new(x: [T; 4]) -> Self {
        Self(x)
    }

    #[inline]
    pub fn derive(mut self) -> Self {
        self.0[1] = T::one();
        self
    }

    #[inline]
    pub fn f0(&self) -> T {
        self.0[0].clone()
    }

    #[inline]
    pub fn f0_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    #[inline]
    fn chain_rule(&self, f0: T, f1: T, f2: T, f3: T) -> Self {
        let three = T::one() + T::one() + T::one();
        Self([
            f0,
            f1 * self.0[1],
            f2 * self.0[1] * self.0[1] + f1 * self.0[2],
            f3 * self.0[1] * self.0[1] * self.0[1]
                + three * f2 * self.0[1] * self.0[2]
                + f1 * self.0[3],
        ])
    }
}

impl<T: Float> From<T> for HD3<T> {
    fn from(float: T) -> Self {
        let mut res = [T::zero(); 4];
        res[0] = float;
        Self(res)
    }
}

impl<T: Float> Zero for HD3<T> {
    #[inline]
    fn zero() -> Self {
        Self([T::zero(); 4])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<T: Float> One for HD3<T> {
    #[inline]
    fn one() -> Self {
        let mut res = [T::zero(); 4];
        res[0] = T::one();
        Self(res)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.iter().skip(1).all(|x| x.is_zero()) && self.0[0].is_one()
    }
}

impl<'a, 'b, T: Float> Mul<&'a HD3<T>> for &'b HD3<T> {
    type Output = HD3<T>;
    #[inline]
    fn mul(self, rhs: &HD3<T>) -> HD3<T> {
        let two = T::one() + T::one();
        let three = two + T::one();
        HD3([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[0] + self.0[0] * rhs.0[1],
            self.0[2] * rhs.0[0] + two * self.0[1] * rhs.0[1] + self.0[0] * rhs.0[2],
            self.0[3] * rhs.0[0]
                + three * self.0[2] * rhs.0[1]
                + three * self.0[1] * rhs.0[2]
                + self.0[0] * rhs.0[3],
        ])
    }
}

impl<'a, 'b, T: Float> Div<&'a HD3<T>> for &'b HD3<T> {
    type Output = HD3<T>;
    #[inline]
    fn div(self, rhs: &HD3<T>) -> HD3<T> {
        let rec = T::one() / rhs.0[0];
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = T::from(-2.0).unwrap() * f1 * rec;
        let f3 = T::from(-3.0).unwrap() * f2 * rec;
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

impl<'a, 'b, T: Float> Add<&'a HD3<T>> for &'b HD3<T> {
    type Output = HD3<T>;
    #[inline]
    fn add(self, rhs: &HD3<T>) -> HD3<T> {
        HD3([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl<'a, 'b, T: Float> Sub<&'a HD3<T>> for &'b HD3<T> {
    type Output = HD3<T>;
    #[inline]
    fn sub(self, rhs: &HD3<T>) -> HD3<T> {
        HD3([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

macro_rules! forward_binop {
    ($trt:ident, $operator:tt, $mth:ident) => {
        impl<T: Float> $trt<HD3<T>> for &HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: HD3<T>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: Float> $trt<&HD3<T>> for HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: &HD3<T>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: Float> $trt for HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: HD3<T>) -> Self::Output {
                &self $operator &rhs
            }
        }
    };
}

forward_binop!(Mul, *, mul);
forward_binop!(Div, /, div);
forward_binop!(Add, +, add);
forward_binop!(Sub, -, sub);

macro_rules! impl_scalar_op {
    ($trt:ident, $operator:tt, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Float> $trt<T> for HD3<T>
        {
            type Output = Self;
            #[inline]
            fn $mth(self, rhs: T) -> Self {
                &self $operator rhs
            }
        }

        impl<T: Float> $trt<&T> for HD3<T>
        {
            type Output = Self;
            #[inline]
            fn $mth(self, rhs: &T) -> Self {
                &self $operator rhs
            }
        }

        impl<T: Float> $trt<T> for &HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: T) -> HD3<T> {
                HD3([self.0[0] $operator rhs,
                     self.0[1] $operator rhs,
                     self.0[2] $operator rhs,
                     self.0[3] $operator rhs])
            }
        }

        impl<T: Float> $trt<&T> for &HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: &T) -> HD3<T> {
                self $operator *rhs
            }
        }

        // impl<T: Float> $trt_assign<T> for HD3<T>
        // {
        //     fn $mth_assign(&mut self, rhs: T) {
        //         self.0.iter_mut().for_each(|x| *x $op_assign rhs);
        //     }
        // }
    };
}

macro_rules! impl_scalar_addition_op {
    ($trt:ident, $operator:tt, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Float> $trt<T> for HD3<T>
        {
            type Output = Self;
            #[inline]
            fn $mth(mut self, rhs: T) -> Self {
                *self.f0_mut() = self.f0() $operator rhs;
                self
            }
        }

        impl<T: Float> $trt<&T> for HD3<T>
        {
            type Output = Self;
            #[inline]
            fn $mth(mut self, rhs: &T) -> Self {
                *self.f0_mut() = self.f0() $operator *rhs;
                self
            }
        }

        impl<T: Float> $trt<T> for &HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: T) -> HD3<T> {
                let mut res = HD3(self.0.clone());
                *res.f0_mut() = self.f0() $operator rhs;
                res
            }
        }

        impl<T: Float> $trt<&T> for &HD3<T>
        {
            type Output = HD3<T>;
            #[inline]
            fn $mth(self, rhs: &T) -> HD3<T> {
                let mut res = HD3(self.0.clone());
                *res.f0_mut() = self.f0() $operator *rhs;
                res
            }
        }

        // impl<T: Float> $trt_assign<T> for HD3<T>
        // {
        //     fn $mth_assign(&mut self, rhs: T) {
        //         *self.f0_mut() $op_assign rhs;
        //     }
        // }
    };
}

impl_scalar_op!(Mul, *, mul, MulAssign, *=, mul_assign);
impl_scalar_op!(Div, /, div, DivAssign, /=, div_assign);
impl_scalar_addition_op!(Add, +, add, AddAssign, +=, add_assign);
impl_scalar_addition_op!(Sub, -, sub, SubAssign, -=, sub_assign);

// macro_rules! impl_assign_op {
//     ($trt:ident, $operator:tt, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {

//         impl<D: Derivative> $trt_assign<&HD3<D>> for HD3<D>
//         {
//             fn $mth_assign(&mut self, rhs: &HD3<D>) {
//                 let res = &*self $operator rhs;
//                 self.0.iter_mut().zip(res.0.iter()).for_each(|(s, &r)| *s = r);
//             }
//         }

//         impl<D: Derivative> $trt_assign for HD3<D>
//         {
//             fn $mth_assign(&mut self, rhs: HD3<D>) {
//                 *self $op_assign &rhs;
//             }
//         }
//     };
// }

// macro_rules! impl_addition_assign_op {
//     ($trt:ident, $operator:tt, $mth:ident) => {
//         impl<D: Derivative> $trt<&HD3<D>> for HD3<D>
//         {
//             fn $mth(&mut self, rhs: &HD3<D>) {
//                 self.0.iter_mut().zip(rhs.0.iter()).for_each(|(s, &r)| *s $operator r);
//             }
//         }

//         impl<D: Derivative> $trt<HD3<D>> for HD3<D>
//         {
//             fn $mth(&mut self, rhs: HD3<D>) {
//                 self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(s, r)| *s $operator r);
//             }
//         }
//     };
// }

// impl_assign_op!(Mul, *, MulAssign, *=, mul_assign);
// // impl_assign_op!(Div, /, DivAssign, /=, div_assign);
// impl_addition_assign_op!(AddAssign, +=, add_assign);
// impl_addition_assign_op!(SubAssign, -=, sub_assign);

// impl<D: Derivative> DivAssign<&HD3<D>> for HD3<D> {
//     fn div_assign(&mut self, rhs: &HD3<D>) {
//         self.0
//             .iter_mut()
//             .zip(rhs.0.iter())
//             .for_each(|(s, &r)| *s /= r);
//     }
// }

// impl<D: Derivative> DivAssign<HD3<D>> for HD3<D> {
//     fn div_assign(&mut self, rhs: HD3<D>) {
//         self.0
//             .iter_mut()
//             .zip(rhs.0.into_iter())
//             .for_each(|(s, r)| *s /= r);
//     }
// }

impl<T: Float> Neg for HD3<T> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.neg());
        self
    }
}

impl<T: Float> Neg for &HD3<T> {
    type Output = HD3<T>;
    #[inline]
    fn neg(self) -> Self::Output {
        HD3([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl<T: Float> DualNumMethods<T> for HD3<T> {
    #[inline]
    fn re(&self) -> T {
        self.0[0]
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().recip();
    /// assert!((res.0[0] - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.0[1] - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.0[2] - 1.15740740740741).abs() < 1e-10);
    /// assert!((res.0[3] - -2.89351851851852).abs() < 1e-10);
    /// ```
    #[inline]
    fn recip(&self) -> Self {
        let rec = self.0[0].recip();
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = T::from(-2.0).unwrap() * f1 * rec;
        let f3 = T::from(-3.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().powi(6);
    /// assert!((res.0[0] - 2.98598400000000).abs() < 1e-10);
    /// assert!((res.0[1] - 14.9299200000000).abs() < 1e-10);
    /// assert!((res.0[2] - 62.2080000000000).abs() < 1e-10);
    /// assert!((res.0[3] - 207.360000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powi(&self, n: i32) -> Self {
        let f3 = self.0[0].powi(n - 3);
        let f2 = f3 * self.0[0];
        let f1 = f2 * self.0[0];
        let f0 = f1 * self.0[0];
        self.chain_rule(
            f0,
            T::from(n).unwrap() * f1,
            T::from(n * (n - 1)).unwrap() * f2,
            T::from(n * (n - 1) * (n - 2)).unwrap() * f3,
        )
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().powf(4.2);
    /// assert!((res.0[0] - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.0[1] - 7.52712759108966).abs() < 1e-10);
    /// assert!((res.0[2] - 20.0723402429058).abs() < 1e-10);
    /// assert!((res.0[3] - 36.7992904453272).abs() < 1e-10);
    /// ```
    #[inline]
    fn powf(&self, n: T) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].powf(n);
        let f1 = n * f0 * rec;
        let f2 = (n - T::one()) * f1 * rec;
        let f3 = (n - T::one() - T::one()) * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sqrt();
    /// assert!((res.0[0] - 1.09544511501033).abs() < 1e-10);
    /// assert!((res.0[1] - 0.456435464587638).abs() < 1e-10);
    /// assert!((res.0[2] - -0.190181443578183).abs() < 1e-10);
    /// assert!((res.0[3] - 0.237726804472728).abs() < 1e-10);
    /// ```
    #[inline]
    fn sqrt(&self) -> Self {
        let rec = self.0[0].recip();
        let half = T::from(0.5).unwrap();
        let f0 = self.0[0].sqrt();
        let f1 = half * f0 * rec;
        let f2 = -half * f1 * rec;
        let f3 = (-T::one() - half) * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().cbrt();
    /// assert!((res.0[0] - 1.06265856918261).abs() < 1e-10);
    /// assert!((res.0[1] - 0.295182935884059).abs() < 1e-10);
    /// assert!((res.0[2] - -0.163990519935588).abs() < 1e-10);
    /// assert!((res.0[3] - 0.227764611021650).abs() < 1e-10);
    /// ```
    #[inline]
    fn cbrt(&self) -> Self {
        let rec = self.0[0].recip();
        let third = T::from(1.0 / 3.0).unwrap();
        let f0 = self.0[0].cbrt();
        let f1 = third * f0 * rec;
        let f2 = (third - T::one()) * f1 * rec;
        let f3 = (third - T::one() - T::one()) * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().exp();
    /// assert!((res.0[0] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[1] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[2] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[3] - 3.32011692273655).abs() < 1e-10);
    /// ```

    #[inline]
    fn exp(&self) -> Self {
        let f = self.0[0].exp();
        self.chain_rule(f, f, f, f)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().exp2();
    /// assert!((res.0[0] - 2.29739670999407).abs() < 1e-10);
    /// assert!((res.0[1] - 1.59243405216008).abs() < 1e-10);
    /// assert!((res.0[2] - 1.10379117348241).abs() < 1e-10);
    /// assert!((res.0[3] - 0.765089739826287).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp2(&self) -> Self {
        let ln2 = T::from(2.0).unwrap().ln();
        let f0 = self.0[0].exp2();
        let f1 = f0 * ln2;
        let f2 = f1 * ln2;
        let f3 = f2 * ln2;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().exp_m1();
    /// assert!((res.0[0] - 2.32011692273655).abs() < 1e-10);
    /// assert!((res.0[1] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[2] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[3] - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp_m1(&self) -> Self {
        let f0 = self.0[0].exp_m1();
        let f1 = self.0[0].exp();
        self.chain_rule(f0, f1, f1, f1)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().ln();
    /// assert!((res.0[0] - 0.182321556793955).abs() < 1e-10);
    /// assert!((res.0[1] - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.0[2] - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.0[3] - 1.15740740740741).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln(&self) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].ln();
        let f1 = rec;
        let f2 = -f1 * rec;
        let f3 = T::from(-2.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().log(4.2);
    /// assert!((res.0[0] - 0.127045866345188).abs() < 1e-10);
    /// assert!((res.0[1] - 0.580685888982970).abs() < 1e-10);
    /// assert!((res.0[2] - -0.483904907485808).abs() < 1e-10);
    /// assert!((res.0[3] - 0.806508179143013).abs() < 1e-10);
    /// ```
    #[inline]
    fn log(&self, base: T) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].log(base);
        let f1 = rec / base.ln();
        let f2 = -f1 * rec;
        let f3 = T::from(-2.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().log2();
    /// assert!((res.0[0] - 0.263034405833794).abs() < 1e-10);
    /// assert!((res.0[1] - 1.20224586740747).abs() < 1e-10);
    /// assert!((res.0[2] - -1.00187155617289).abs() < 1e-10);
    /// assert!((res.0[3] - 1.66978592695482).abs() < 1e-10);
    /// ```
    #[inline]
    fn log2(&self) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].log2();
        let f1 = rec / (T::one() + T::one()).ln();
        let f2 = -f1 * rec;
        let f3 = T::from(-2.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().log10();
    /// assert!((res.0[0] - 0.0791812460476248).abs() < 1e-10);
    /// assert!((res.0[1] - 0.361912068252710).abs() < 1e-10);
    /// assert!((res.0[2] - -0.301593390210592).abs() < 1e-10);
    /// assert!((res.0[3] - 0.502655650350986).abs() < 1e-10);
    /// ```
    #[inline]
    fn log10(&self) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].log10();
        let f1 = rec / T::from(10.0).unwrap().ln();
        let f2 = -f1 * rec;
        let f3 = T::from(-2.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().ln_1p();
    /// assert!((res.0[0] - 0.788457360364270).abs() < 1e-10);
    /// assert!((res.0[1] - 0.454545454545455).abs() < 1e-10);
    /// assert!((res.0[2] - -0.206611570247934).abs() < 1e-10);
    /// assert!((res.0[3] - 0.187828700225394).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln_1p(&self) -> Self {
        let rec = (T::one() + self.0[0]).recip();
        let f0 = self.0[0].ln_1p();
        let f1 = rec;
        let f2 = -f1 * rec;
        let f3 = T::from(-2.0).unwrap() * f2 * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sin();
    /// assert!((res.0[0] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[1] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[2] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[3] - -0.362357754476674).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin(&self) -> Self {
        let (s, c) = self.0[0].sin_cos();
        self.chain_rule(s, c, -s, -c)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().cos();
    /// assert!((res.0[0] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[1] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[2] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[3] - 0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn cos(&self) -> Self {
        let (s, c) = self.0[0].sin_cos();
        self.chain_rule(c, -s, -c, s)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let (res_sin, res_cos) = HD3_64::from(1.2).derive().sin_cos();
    /// assert!((res_sin.0[0] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.0[1] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_sin.0[2] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.0[3] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[0] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[1] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.0[2] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[3] - 0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin_cos(&self) -> (Self, Self) {
        let (s, c) = self.0[0].sin_cos();
        (self.chain_rule(s, c, -s, -c), self.chain_rule(c, -s, -c, s))
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().tan();
    /// assert!((res.0[0] - 2.57215162212632).abs() < 1e-10);
    /// assert!((res.0[1] - 7.61596396720705).abs() < 1e-10);
    /// assert!((res.0[2] - 39.1788281446144).abs() < 1e-10);
    /// assert!((res.0[3] - 317.553587029949).abs() < 1e-10);
    /// ```
    #[inline]
    fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(0.2).derive().asin();
    /// assert!((res.0[0] - 0.201357920790331).abs() < 1e-10);
    /// assert!((res.0[1] - 1.02062072615966).abs() < 1e-10);
    /// assert!((res.0[2] - 0.212629317949929).abs() < 1e-10);
    /// assert!((res.0[3] - 1.19603991346835).abs() < 1e-10);
    /// ```
    #[inline]
    fn asin(&self) -> Self {
        let rec = (T::one() - self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].asin();
        let f1 = rec.sqrt();
        let f2 = self.0[0] * f1 * rec;
        let f3 = ((T::one() + T::one()) * self.0[0] * self.0[0] + T::one()) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(0.2).derive().acos();
    /// assert!((res.0[0] - 1.36943840600457).abs() < 1e-10);
    /// assert!((res.0[1] - -1.02062072615966).abs() < 1e-10);
    /// assert!((res.0[2] - -0.212629317949929).abs() < 1e-10);
    /// assert!((res.0[3] - -1.19603991346835).abs() < 1e-10);
    /// ```
    #[inline]
    fn acos(&self) -> Self {
        let rec = (T::one() - self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].acos();
        let f1 = -rec.sqrt();
        let f2 = self.0[0] * f1 * rec;
        let f3 = ((T::one() + T::one()) * self.0[0] * self.0[0] + T::one()) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(0.2).derive().atan();
    /// assert!((res.0[0] - 0.197395559849881).abs() < 1e-10);
    /// assert!((res.0[1] - 0.961538461538461).abs() < 1e-10);
    /// assert!((res.0[2] - -0.369822485207101).abs() < 1e-10);
    /// assert!((res.0[3] - -1.56463359126081).abs() < 1e-10);
    /// ```
    #[inline]
    fn atan(&self) -> Self {
        let two = T::one() + T::one();
        let rec = (T::one() + self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].atan();
        let f1 = rec;
        let f2 = -two * self.0[0] * f1 * rec;
        let f3 = (T::from(6.0).unwrap() * self.0[0] * self.0[0] - two) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sinh();
    /// assert!((res.0[0] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[1] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[2] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[3] - 1.81065556732437).abs() < 1e-10);
    /// ```
    #[inline]
    fn sinh(&self) -> Self {
        let s = self.0[0].sinh();
        let c = self.0[0].cosh();
        self.chain_rule(s, c, s, c)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().cosh();
    /// assert!((res.0[0] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[1] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[2] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[3] - 1.50946135541217).abs() < 1e-10);
    /// ```
    #[inline]
    fn cosh(&self) -> Self {
        let s = self.0[0].sinh();
        let c = self.0[0].cosh();
        self.chain_rule(c, s, c, s)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().tanh();
    /// assert!((res.0[0] - 0.833654607012155).abs() < 1e-10);
    /// assert!((res.0[1] - 0.305019996207409).abs() < 1e-10);
    /// assert!((res.0[2] - -0.508562650138273).abs() < 1e-10);
    /// assert!((res.0[3] - 0.661856796311429).abs() < 1e-10);
    /// ```
    #[inline]
    fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().asinh();
    /// assert!((res.0[0] - 1.01597313417969).abs() < 1e-10);
    /// assert!((res.0[1] - 0.640184399664480).abs() < 1e-10);
    /// assert!((res.0[2] - -0.314844786720236).abs() < 1e-10);
    /// assert!((res.0[3] - 0.202154439560807).abs() < 1e-10);
    /// ```
    #[inline]
    fn asinh(&self) -> Self {
        let rec = (T::one() + self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].asinh();
        let f1 = rec.sqrt();
        let f2 = -self.0[0] * f1 * rec;
        let f3 = ((T::one() + T::one()) * self.0[0] * self.0[0] - T::one()) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().acosh();
    /// assert!((res.0[0] - 0.622362503714779).abs() < 1e-10);
    /// assert!((res.0[1] - 1.50755672288882).abs() < 1e-10);
    /// assert!((res.0[2] - -4.11151833515132).abs() < 1e-10);
    /// assert!((res.0[3] - 30.2134301901271).abs() < 1e-10);
    /// ```
    #[inline]
    fn acosh(&self) -> Self {
        let rec = (self.0[0] * self.0[0] - T::one()).recip();
        let f0 = self.0[0].acosh();
        let f1 = rec.sqrt();
        let f2 = -self.0[0] * f1 * rec;
        let f3 = ((T::one() + T::one()) * self.0[0] * self.0[0] + T::one()) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(0.2).derive().atanh();
    /// assert!((res.0[0] - 0.202732554054082).abs() < 1e-10);
    /// assert!((res.0[1] - 1.04166666666667).abs() < 1e-10);
    /// assert!((res.0[2] - 0.434027777777778).abs() < 1e-10);
    /// assert!((res.0[3] - 2.53182870370370).abs() < 1e-10);
    /// ```
    #[inline]
    fn atanh(&self) -> Self {
        let two = T::one() + T::one();
        let rec = (T::one() - self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].atanh();
        let f1 = rec;
        let f2 = two * self.0[0] * f1 * rec;
        let f3 = (T::from(6.0).unwrap() * self.0[0] * self.0[0] + two) * f1 * rec * rec;
        self.chain_rule(f0, f1, f2, f3)
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sph_j0();
    /// assert!((res.0[0] - 0.776699238306022).abs() < 1e-10);
    /// assert!((res.0[1] - -0.345284569857790).abs() < 1e-10);
    /// assert!((res.0[2] - -0.201224955209705).abs() < 1e-10);
    /// assert!((res.0[3] - 0.201097592627034).abs() < 1e-10);
    /// ```
    #[inline]
    fn sph_j0(&self) -> Self {
        if self.0[0] < T::epsilon() {
            Self::one() - self * self / T::from(6.0).unwrap()
        } else {
            self.sin() / self
        }
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sph_j1();
    /// assert!((res.0[0] - 0.345284569857790).abs() < 1e-10);
    /// assert!((res.0[1] - 0.201224955209705).abs() < 1e-10);
    /// assert!((res.0[2] - -0.201097592627034).abs() < 1e-10);
    /// assert!((res.0[3] - -0.106373929549242).abs() < 1e-10);
    /// ```
    #[inline]
    fn sph_j1(&self) -> Self {
        if self.0[0] < T::epsilon() {
            self / T::from(3.0).unwrap()
        } else {
            let (s, c) = self.sin_cos();
            (s - self * c) / (self * self)
        }
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().sph_j2();
    /// assert!((res.0[0] - 0.0865121863384538).abs() < 1e-10);
    /// assert!((res.0[1] - 0.129004104011656).abs() < 1e-10);
    /// assert!((res.0[2] - 0.0589484167190109).abs() < 1e-10);
    /// assert!((res.0[3] - -0.111341070273404).abs() < 1e-10);
    /// ```
    #[inline]
    fn sph_j2(&self) -> Self {
        if self.0[0] < T::epsilon() {
            self * self / T::from(15.0).unwrap()
        } else {
            let (s, c) = self.sin_cos();
            let s2 = self * self;
            ((&s - self * c) * T::from(3.0).unwrap() - &s2 * s) / (s2 * self)
        }
    }
}

impl<T: Float> Inv for HD3<T> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

impl<T: Float> Inv for &HD3<T> {
    type Output = HD3<T>;
    fn inv(self) -> HD3<T> {
        self.recip()
    }
}

/* iterator methods */
impl<T: Float> Sum for HD3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: 'a + Float> Sum<&'a HD3<T>> for HD3<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HD3<T>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: Float> Product for HD3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: 'a + Float> Product<&'a HD3<T>> for HD3<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HD3<T>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

/* string conversions */
impl<T> fmt::Display for HD3<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "re: {}, dx: {}, dx2: {}, dx3: {}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}
