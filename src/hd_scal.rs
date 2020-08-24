use crate::DualNum;
use num_integer::binomial;
use num_traits::{Float, Inv, One, Zero};
use smallvec::{smallvec, Array, SmallVec};
use std::iter::{Product, Sum};
use std::ops::*;

pub trait Derivative<T: Float>: PartialEq + Clone {
    type Arr: Array<Item = T> + PartialEq + Clone;
    const NDIM: usize;

    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self>;
}

#[derive(PartialEq, Clone)]
pub struct D0();
#[derive(PartialEq, Clone)]
pub struct D1();
#[derive(PartialEq, Clone)]
pub struct D2();
#[derive(PartialEq, Clone)]
pub struct D3();
#[derive(PartialEq, Clone)]
pub struct D4();
#[derive(PartialEq, Clone)]
pub struct D5();

impl<T: Float> Derivative<T> for D0 {
    type Arr = [T; 1];
    const NDIM: usize = 1;

    #[inline]
    fn chain_rule(_: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        f
    }
}

impl<T: Float> Derivative<T> for D1 {
    type Arr = [T; 2];
    const NDIM: usize = 2;

    #[inline]
    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        let mut res = SmallVec::with_capacity(2);
        res.push(f.0[0]);
        res.push(f.0[1] * x.0[1]);
        HDScal(res)
    }
}

impl<T: Float> Derivative<T> for D2 {
    type Arr = [T; 3];
    const NDIM: usize = 3;

    #[inline]
    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        let mut res = SmallVec::with_capacity(3);
        res.push(f.0[0]);
        res.push(f.0[1] * x.0[1]);
        res.push(f.0[2] * x.0[1].powi(2) + f.0[1] * x.0[2]);
        HDScal(res)
    }
}

impl<T: Float> Derivative<T> for D3 {
    type Arr = [T; 4];
    const NDIM: usize = 4;

    #[inline]
    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        let three = T::one() + T::one() + T::one();
        let mut res = SmallVec::with_capacity(4);
        res.push(f.0[0]);
        res.push(f.0[1] * x.0[1]);
        res.push(f.0[2] * x.0[1].powi(2) + f.0[1] * x.0[2]);
        res.push(f.0[3] * x.0[1].powi(3) + three * f.0[2] * x.0[1] * x.0[2] + f.0[1] * x.0[3]);
        HDScal(res)
    }
}

impl<T: Float> Derivative<T> for D4 {
    type Arr = [T; 5];
    const NDIM: usize = 5;

    #[inline]
    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        let three = T::one() + T::one() + T::one();
        let mut res = SmallVec::with_capacity(5);
        res.push(f.0[0]);
        res.push(f.0[1] * x.0[1]);
        res.push(f.0[2] * x.0[1].powi(2) + f.0[1] * x.0[2]);
        res.push(f.0[3] * x.0[1].powi(3) + three * f.0[2] * x.0[1] * x.0[2] + f.0[1] * x.0[3]);
        res.push(
            f.0[4] * x.0[1].powi(4)
                + T::from(6.0).unwrap() * f.0[3] * x.0[1].powi(2) * x.0[2]
                + T::from(4.0).unwrap() * f.0[2] * x.0[1] * x.0[3]
                + three * f.0[2] * x.0[2].powi(2)
                + f.0[1] * x.0[4],
        );
        HDScal(res)
    }
}

impl<T: Float> Derivative<T> for D5 {
    type Arr = [T; 6];
    const NDIM: usize = 6;

    #[inline]
    fn chain_rule(x: &HDScal<T, Self>, f: HDScal<T, Self>) -> HDScal<T, Self> {
        let three = T::one() + T::one() + T::one();
        let ten = T::from(10.0).unwrap();
        let mut res = SmallVec::with_capacity(6);
        res.push(f.0[0]);
        res.push(f.0[1] * x.0[1]);
        res.push(f.0[2] * x.0[1].powi(2) + f.0[1] * x.0[2]);
        res.push(f.0[3] * x.0[1].powi(3) + three * f.0[2] * x.0[1] * x.0[2] + f.0[1] * x.0[3]);
        res.push(
            f.0[4] * x.0[1].powi(4)
                + T::from(6.0).unwrap() * f.0[3] * x.0[1].powi(2) * x.0[2]
                + T::from(4.0).unwrap() * f.0[2] * x.0[1] * x.0[3]
                + three * f.0[2] * x.0[2].powi(2)
                + f.0[1] * x.0[4],
        );
        res.push(
            f.0[5] * x.0[1].powi(5)
                + ten * f.0[4] * x.0[1].powi(3) * x.0[2]
                + ten * f.0[3] * x.0[1].powi(2) * x.0[3]
                + T::from(15.0).unwrap() * f.0[3] * x.0[1] * x.0[2].powi(2)
                + ten * f.0[2] * x.0[2] * x.0[3]
                + T::from(5.0).unwrap() * f.0[2] * x.0[1] * x.0[4]
                + f.0[1] * x.0[5],
        );
        HDScal(res)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct HDScal<T: Float, D: Derivative<T>>(pub SmallVec<D::Arr>);

pub type HDScal32<D> = HDScal<f32, D>;
pub type HDScal64<D> = HDScal<f64, D>;

impl<T: Float, D: Derivative<T>> HDScal<T, D> {
    #[inline]
    pub fn new(x: T) -> Self {
        let mut res = smallvec![T::zero(); D::NDIM];
        res[0] = x;
        Self(res)
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
}

impl<T: Float, D: Derivative<T>> From<T> for HDScal<T, D> {
    fn from(float: T) -> Self {
        Self::new(float)
    }
}

impl<T: Float, D: Derivative<T>> Zero for HDScal<T, D> {
    #[inline]
    fn zero() -> Self {
        Self(smallvec![T::zero(); D::NDIM])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<T: Float, D: Derivative<T>> One for HDScal<T, D> {
    #[inline]
    fn one() -> Self {
        let mut res = smallvec![T::zero(); D::NDIM];
        res[0] = T::one();
        Self(res)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.iter().skip(1).all(|x| x.is_zero()) && self.0[0].is_one()
    }
}

impl<'a, 'b, T: Float, D: Derivative<T>> Mul<&'a HDScal<T, D>> for &'b HDScal<T, D> {
    type Output = HDScal<T, D>;
    #[inline]
    fn mul(self, rhs: &HDScal<T, D>) -> HDScal<T, D> {
        let n = self.0.len();
        let mut res = SmallVec::with_capacity(n);
        for i in 0..n {
            let mut s = T::zero();
            for j in 0..=i {
                s = s + self.0[j] * rhs.0[i - j] * T::from(binomial(i, j)).unwrap();
            }
            res.push(s);
        }
        HDScal(res)
    }
}

impl<'a, 'b, T: Float, D: Derivative<T>> Div<&'a HDScal<T, D>> for &'b HDScal<T, D> {
    type Output = HDScal<T, D>;
    #[inline]
    fn div(self, rhs: &HDScal<T, D>) -> HDScal<T, D> {
        self * &rhs.recip()
    }
}

impl<'a, 'b, T: Float, D: Derivative<T>> Add<&'a HDScal<T, D>> for &'b HDScal<T, D> {
    type Output = HDScal<T, D>;
    #[inline]
    fn add(self, rhs: &HDScal<T, D>) -> HDScal<T, D> {
        HDScal(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&s, &r)| s + r)
                .collect(),
        )
    }
}

impl<'a, 'b, T: Float, D: Derivative<T>> Sub<&'a HDScal<T, D>> for &'b HDScal<T, D> {
    type Output = HDScal<T, D>;
    #[inline]
    fn sub(self, rhs: &HDScal<T, D>) -> HDScal<T, D> {
        HDScal(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&s, &r)| s - r)
                .collect(),
        )
    }
}

macro_rules! forward_binop {
    ($trt:ident, $operator:tt, $mth:ident) => {
        impl<T: Float, D: Derivative<T>> $trt<HDScal<T, D>> for &HDScal<T, D>
        {
            type Output = HDScal<T, D>;
            #[inline]
            fn $mth(self, rhs: HDScal<T, D>) -> Self::Output {
                self $operator &rhs
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<&HDScal<T, D>> for HDScal<T, D>
        {
            type Output = HDScal<T, D>;
            #[inline]
            fn $mth(self, rhs: &HDScal<T, D>) -> Self::Output {
                &self $operator rhs
            }
        }

        impl<T: Float, D: Derivative<T>> $trt for HDScal<T, D>
        {
            type Output = HDScal<T, D>;
            #[inline]
            fn $mth(self, rhs: HDScal<T, D>) -> Self::Output {
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
        impl<T: Float, D: Derivative<T>> $trt<T> for HDScal<T,D>
        {
            type Output = Self;
            #[inline]
            fn $mth(self, rhs: T) -> Self {
                &self $operator rhs
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<&T> for HDScal<T,D>
        {
            type Output = Self;
            #[inline]
            fn $mth(self, rhs: &T) -> Self {
                &self $operator rhs
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<T> for &HDScal<T,D>
        {
            type Output = HDScal<T,D>;
            #[inline]
            fn $mth(self, rhs: T) -> HDScal<T,D> {
                HDScal(self.0.iter().map(|x| *x $operator rhs).collect())
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<&T> for &HDScal<T,D>
        {
            type Output = HDScal<T,D>;
            #[inline]
            fn $mth(self, rhs: &T) -> HDScal<T,D> {
                HDScal(self.0.iter().map(|x| *x $operator *rhs).collect())
            }
        }

        // impl<T: Float, D: Derivative<T>> $trt_assign<T> for HDScal<T,D>
        // {
        //     fn $mth_assign(&mut self, rhs: T) {
        //         self.0.iter_mut().for_each(|x| *x $op_assign rhs);
        //     }
        // }
    };
}

macro_rules! impl_scalar_addition_op {
    ($trt:ident, $operator:tt, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Float, D: Derivative<T>> $trt<T> for HDScal<T,D>
        {
            type Output = Self;
            #[inline]
            fn $mth(mut self, rhs: T) -> Self {
                *self.f0_mut() = self.f0() $operator rhs;
                self
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<&T> for HDScal<T,D>
        {
            type Output = Self;
            #[inline]
            fn $mth(mut self, rhs: &T) -> Self {
                *self.f0_mut() = self.f0() $operator *rhs;
                self
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<T> for &HDScal<T,D>
        {
            type Output = HDScal<T,D>;
            #[inline]
            fn $mth(self, rhs: T) -> HDScal<T,D> {
                let mut res = HDScal(self.0.clone());
                *res.f0_mut() = self.f0() $operator rhs;
                res
            }
        }

        impl<T: Float, D: Derivative<T>> $trt<&T> for &HDScal<T,D>
        {
            type Output = HDScal<T,D>;
            #[inline]
            fn $mth(self, rhs: &T) -> HDScal<T,D> {
                let mut res = HDScal(self.0.clone());
                *res.f0_mut() = self.f0() $operator *rhs;
                res
            }
        }

        // impl<T: Float, D: Derivative<T>> $trt_assign<T> for HDScal<T,D>
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

//         impl<D: Derivative> $trt_assign<&HDScal<D>> for HDScal<D>
//         {
//             fn $mth_assign(&mut self, rhs: &HDScal<D>) {
//                 let res = &*self $operator rhs;
//                 self.0.iter_mut().zip(res.0.iter()).for_each(|(s, &r)| *s = r);
//             }
//         }

//         impl<D: Derivative> $trt_assign for HDScal<D>
//         {
//             fn $mth_assign(&mut self, rhs: HDScal<D>) {
//                 *self $op_assign &rhs;
//             }
//         }
//     };
// }

// macro_rules! impl_addition_assign_op {
//     ($trt:ident, $operator:tt, $mth:ident) => {
//         impl<D: Derivative> $trt<&HDScal<D>> for HDScal<D>
//         {
//             fn $mth(&mut self, rhs: &HDScal<D>) {
//                 self.0.iter_mut().zip(rhs.0.iter()).for_each(|(s, &r)| *s $operator r);
//             }
//         }

//         impl<D: Derivative> $trt<HDScal<D>> for HDScal<D>
//         {
//             fn $mth(&mut self, rhs: HDScal<D>) {
//                 self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(s, r)| *s $operator r);
//             }
//         }
//     };
// }

// impl_assign_op!(Mul, *, MulAssign, *=, mul_assign);
// // impl_assign_op!(Div, /, DivAssign, /=, div_assign);
// impl_addition_assign_op!(AddAssign, +=, add_assign);
// impl_addition_assign_op!(SubAssign, -=, sub_assign);

// impl<D: Derivative> DivAssign<&HDScal<D>> for HDScal<D> {
//     fn div_assign(&mut self, rhs: &HDScal<D>) {
//         self.0
//             .iter_mut()
//             .zip(rhs.0.iter())
//             .for_each(|(s, &r)| *s /= r);
//     }
// }

// impl<D: Derivative> DivAssign<HDScal<D>> for HDScal<D> {
//     fn div_assign(&mut self, rhs: HDScal<D>) {
//         self.0
//             .iter_mut()
//             .zip(rhs.0.into_iter())
//             .for_each(|(s, r)| *s /= r);
//     }
// }

impl<T: Float, D: Derivative<T>> Neg for HDScal<T, D> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.neg());
        self
    }
}

impl<T: Float, D: Derivative<T>> Neg for &HDScal<T, D> {
    type Output = HDScal<T, D>;
    #[inline]
    fn neg(self) -> Self::Output {
        HDScal(self.0.iter().map(|x| -*x).collect())
    }
}

impl<T: Float, D: Derivative<T>> HDScal<T, D> {
    #[inline]
    fn apply_function<F, DF>(&self, f: F, df: DF) -> HDScal<T, D>
    where
        F: Fn(T) -> T,
        DF: Fn(&HDScal<T, D>) -> HDScal<T, D>,
    {
        if self.0.len() == 1 {
            return HDScal(smallvec![f(self.f0())]);
        }
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(f(self.f0()));

        let deriv = {
            let mut x: SmallVec<D::Arr> = self.0.clone().into_iter().skip(1).collect();
            x.push(T::zero());
            Self(x)
        };

        let diff = {
            let mut x: SmallVec<D::Arr> = (0..(self.0.len() - 1)).map(|i| self.0[i]).collect();
            x.push(T::zero());
            df(&Self(x))
        };

        for x in (diff * deriv).0.into_iter() {
            res.push(x);
        }
        Self(res)
    }
}

impl<T: Float, D: Derivative<T>> DualNum<T> for HDScal<T, D> {
    #[inline]
    fn re(&self) -> T {
        self.0[0]
    }

    /// Returns `1/self`
    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().recip();
    /// assert!((res.0[0] - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.0[1] - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.0[2] - 1.15740740740741).abs() < 1e-10);
    /// assert!((res.0[3] - -2.89351851851852).abs() < 1e-10);
    /// assert!((res.0[4] - 9.64506172839506).abs() < 1e-10);
    /// assert!((res.0[5] - -40.1877572016461).abs() < 1e-10);
    /// ```
    #[inline]
    fn recip(&self) -> Self {
        let rec = self.0[0].recip();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = rec;
        res.push(f);
        for i in 0..self.0.len() as i32 - 1 {
            f = T::from(-1 - i).unwrap() * f * rec;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().powi(6);
    /// assert!((res.0[0] - 2.98598400000000).abs() < 1e-10);
    /// assert!((res.0[1] - 14.9299200000000).abs() < 1e-10);
    /// assert!((res.0[2] - 62.2080000000000).abs() < 1e-10);
    /// assert!((res.0[3] - 207.360000000000).abs() < 1e-10);
    /// assert!((res.0[4] - 518.400000000000).abs() < 1e-10);
    /// assert!((res.0[5] - 864.000000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powi(&self, n: i32) -> Self {
        let rec = self.0[0].recip();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = self.0[0].powi(n);
        res.push(f);
        for i in 0..self.0.len() as i32 - 1 {
            f = T::from(n - i).unwrap() * f * rec;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().powf(4.2);
    /// assert!((res.0[0] - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.0[1] - 7.52712759108966).abs() < 1e-10);
    /// assert!((res.0[2] - 20.0723402429058).abs() < 1e-10);
    /// assert!((res.0[3] - 36.7992904453272).abs() < 1e-10);
    /// assert!((res.0[4] - 36.7992904453272).abs() < 1e-10);
    /// assert!((res.0[5] - 6.13321507422121).abs() < 1e-10);
    /// ```
    #[inline]
    fn powf(&self, n: T) -> Self {
        let rec = self.0[0].recip();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = self.0[0].powf(n);
        res.push(f);
        for i in 0..self.0.len() - 1 {
            f = (n - T::from(i).unwrap()) * f * rec;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sqrt();
    /// assert!((res.0[0] - 1.09544511501033).abs() < 1e-10);
    /// assert!((res.0[1] - 0.456435464587638).abs() < 1e-10);
    /// assert!((res.0[2] - -0.190181443578183).abs() < 1e-10);
    /// assert!((res.0[3] - 0.237726804472728).abs() < 1e-10);
    /// assert!((res.0[4] - -0.495264175984851).abs() < 1e-10);
    /// assert!((res.0[5] - 1.44452051328915).abs() < 1e-10);
    /// ```
    #[inline]
    fn sqrt(&self) -> Self {
        let rec = self.0[0].recip();
        let half = T::from(0.5).unwrap();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = self.0[0].sqrt();
        res.push(f);
        for i in 0..self.0.len() - 1 {
            f = (half - T::from(i).unwrap()) * f * rec;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().cbrt();
    /// assert!((res.0[0] - 1.06265856918261).abs() < 1e-10);
    /// assert!((res.0[1] - 0.295182935884059).abs() < 1e-10);
    /// assert!((res.0[2] - -0.163990519935588).abs() < 1e-10);
    /// assert!((res.0[3] - 0.227764611021650).abs() < 1e-10);
    /// assert!((res.0[4] - -0.506143580048112).abs() < 1e-10);
    /// assert!((res.0[5] - 1.54654982792479).abs() < 1e-10);
    /// ```
    #[inline]
    fn cbrt(&self) -> Self {
        let rec = self.0[0].recip();
        let third = T::from(1.0 / 3.0).unwrap();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = self.0[0].cbrt();
        res.push(f);
        for i in 0..self.0.len() - 1 {
            f = (third - T::from(i).unwrap()) * f * rec;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().exp_m1();
    /// assert!((res.0[0] - 2.32011692273655).abs() < 1e-10);
    /// assert!((res.0[1] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[2] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[3] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[4] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[5] - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        let f = self.0[0].exp();
        for _ in 0..self.0.len() {
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().exp2();
    /// assert!((res.0[0] - 2.29739670999407).abs() < 1e-10);
    /// assert!((res.0[1] - 1.59243405216008).abs() < 1e-10);
    /// assert!((res.0[2] - 1.10379117348241).abs() < 1e-10);
    /// assert!((res.0[3] - 0.765089739826287).abs() < 1e-10);
    /// assert!((res.0[4] - 0.530319796035933).abs() < 1e-10);
    /// assert!((res.0[5] - 0.367589671417432).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp2(&self) -> Self {
        let ln2 = T::from(2.0).unwrap().ln();
        let mut res = SmallVec::with_capacity(self.0.len());
        let mut f = self.0[0].exp2();
        res.push(f);
        for _ in 1..self.0.len() {
            f = f * ln2;
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().exp_m1();
    /// assert!((res.0[0] - 2.32011692273655).abs() < 1e-10);
    /// assert!((res.0[1] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[2] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[3] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[4] - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.0[5] - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp_m1(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].exp_m1());
        let f = self.0[0].exp();
        for _ in 1..self.0.len() {
            res.push(f);
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().ln();
    /// assert!((res.0[0] - 0.182321556793955).abs() < 1e-10);
    /// assert!((res.0[1] - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.0[2] - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.0[3] - 1.15740740740741).abs() < 1e-10);
    /// assert!((res.0[4] - -2.89351851851852).abs() < 1e-10);
    /// assert!((res.0[5] - 9.64506172839506).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].ln());
        if self.0.len() > 1 {
            let rec = self.0[0].recip();
            let mut f = rec;
            res.push(f);
            for i in 2..self.0.len() as i32 {
                f = T::from(1 - i).unwrap() * f * rec;
                res.push(f);
            }
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().log(4.2);
    /// assert!((res.0[0] - 0.127045866345188).abs() < 1e-10);
    /// assert!((res.0[1] - 0.580685888982970).abs() < 1e-10);
    /// assert!((res.0[2] - -0.483904907485808).abs() < 1e-10);
    /// assert!((res.0[3] - 0.806508179143013).abs() < 1e-10);
    /// assert!((res.0[4] - -2.01627044785753).abs() < 1e-10);
    /// assert!((res.0[5] - 6.72090149285845).abs() < 1e-10);
    /// ```
    #[inline]
    fn log(&self, base: T) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].log(base));
        if self.0.len() > 1 {
            let rec = self.0[0].recip();
            let mut f = rec / base.ln();
            res.push(f);
            for i in 2..self.0.len() as i32 {
                f = T::from(1 - i).unwrap() * f * rec;
                res.push(f);
            }
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().log2();
    /// assert!((res.0[0] - 0.263034405833794).abs() < 1e-10);
    /// assert!((res.0[1] - 1.20224586740747).abs() < 1e-10);
    /// assert!((res.0[2] - -1.00187155617289).abs() < 1e-10);
    /// assert!((res.0[3] - 1.66978592695482).abs() < 1e-10);
    /// assert!((res.0[4] - -4.17446481738705).abs() < 1e-10);
    /// assert!((res.0[5] - 13.9148827246235).abs() < 1e-10);
    /// ```
    #[inline]
    fn log2(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].log2());
        if self.0.len() > 1 {
            let rec = self.0[0].recip();
            let mut f = rec / (T::one() + T::one()).ln();
            res.push(f);
            for i in 2..self.0.len() as i32 {
                f = T::from(1 - i).unwrap() * f * rec;
                res.push(f);
            }
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().log10();
    /// assert!((res.0[0] - 0.0791812460476248).abs() < 1e-10);
    /// assert!((res.0[1] - 0.361912068252710).abs() < 1e-10);
    /// assert!((res.0[2] - -0.301593390210592).abs() < 1e-10);
    /// assert!((res.0[3] - 0.502655650350986).abs() < 1e-10);
    /// assert!((res.0[4] - -1.25663912587746).abs() < 1e-10);
    /// assert!((res.0[5] - 4.18879708625822).abs() < 1e-10);
    /// ```
    #[inline]
    fn log10(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].log10());
        if self.0.len() > 1 {
            let rec = self.0[0].recip();
            let mut f = rec / T::from(10.0).unwrap().ln();
            res.push(f);
            for i in 2..self.0.len() as i32 {
                f = T::from(1 - i).unwrap() * f * rec;
                res.push(f);
            }
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().ln_1p();
    /// assert!((res.0[0] - 0.788457360364270).abs() < 1e-10);
    /// assert!((res.0[1] - 0.454545454545455).abs() < 1e-10);
    /// assert!((res.0[2] - -0.206611570247934).abs() < 1e-10);
    /// assert!((res.0[3] - 0.187828700225394).abs() < 1e-10);
    /// assert!((res.0[4] - -0.256130045761901).abs() < 1e-10);
    /// assert!((res.0[5] - 0.465690992294366).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln_1p(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        res.push(self.0[0].ln_1p());
        if self.0.len() > 1 {
            let rec = (T::one() + self.0[0]).recip();
            let mut f = rec;
            res.push(f);
            for i in 2..self.0.len() as i32 {
                f = T::from(1 - i).unwrap() * f * rec;
                res.push(f);
            }
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sin();
    /// assert!((res.0[0] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[1] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[2] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[3] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[4] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[5] - 0.362357754476674).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        let (s, c) = self.0[0].sin_cos();
        for i in 0..self.0.len() {
            res.push({
                match i % 4 {
                    0 => s,
                    1 => c,
                    2 => -s,
                    3 => -c,
                    _ => unreachable!(),
                }
            });
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().cos();
    /// assert!((res.0[0] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[1] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[2] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[3] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.0[4] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.0[5] - -0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn cos(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        let (s, c) = self.0[0].sin_cos();
        for i in 0..self.0.len() {
            res.push({
                match i % 4 {
                    0 => c,
                    1 => -s,
                    2 => -c,
                    3 => s,
                    _ => unreachable!(),
                }
            });
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let (res_sin, res_cos): (HDScal64<D5>, HDScal64<D5>) = HDScal64::new(1.2).derive().sin_cos();
    /// assert!((res_sin.0[0] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.0[1] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_sin.0[2] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.0[3] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res_sin.0[4] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.0[5] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[0] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[1] - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.0[2] - -0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[3] - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.0[4] - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.0[5] - -0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin_cos(&self) -> (Self, Self) {
        let mut res_sin = SmallVec::with_capacity(self.0.len());
        let mut res_cos = SmallVec::with_capacity(self.0.len());
        let (s, c) = self.0[0].sin_cos();
        for i in 0..self.0.len() {
            res_sin.push({
                match i % 4 {
                    0 => s,
                    1 => c,
                    2 => -s,
                    3 => -c,
                    _ => unreachable!(),
                }
            });
            res_cos.push({
                match i % 4 {
                    0 => c,
                    1 => -s,
                    2 => -c,
                    3 => s,
                    _ => unreachable!(),
                }
            });
        }
        (
            D::chain_rule(self, HDScal(res_sin)),
            D::chain_rule(self, HDScal(res_cos)),
        )
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().tan();
    /// assert!((res.0[0] - 2.57215162212632).abs() < 1e-10);
    /// assert!((res.0[1] - 7.61596396720705).abs() < 1e-10);
    /// assert!((res.0[2] - 39.1788281446144).abs() < 1e-10);
    /// assert!((res.0[3] - 317.553587029949).abs() < 1e-10);
    /// assert!((res.0[4] - 3423.89920854292).abs() < 1e-10);
    /// assert!((res.0[5] - 46171.2726670323).abs() < 1e-10);
    /// ```
    #[inline]
    fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(0.2).derive().asin();
    /// assert!((res.0[0] - 0.201357920790331).abs() < 1e-10);
    /// assert!((res.0[1] - 1.02062072615966).abs() < 1e-10);
    /// assert!((res.0[2] - 0.212629317949929).abs() < 1e-10);
    /// assert!((res.0[3] - 1.19603991346835).abs() < 1e-10);
    /// assert!((res.0[4] - 2.13183040132090).abs() < 1e-10);
    /// assert!((res.0[5] - 14.3217935240254).abs() < 1e-10);
    /// ```
    #[inline]
    fn asin(&self) -> Self {
        self.apply_function(T::asin, |x| (-x * x + T::one()).sqrt().recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(0.2).derive().acos();
    /// assert!((res.0[0] - 1.36943840600457).abs() < 1e-10);
    /// assert!((res.0[1] - -1.02062072615966).abs() < 1e-10);
    /// assert!((res.0[2] - -0.212629317949929).abs() < 1e-10);
    /// assert!((res.0[3] - -1.19603991346835).abs() < 1e-10);
    /// assert!((res.0[4] - -2.13183040132090).abs() < 1e-10);
    /// assert!((res.0[5] - -14.3217935240254).abs() < 1e-10);
    /// ```
    #[inline]
    fn acos(&self) -> Self {
        self.apply_function(T::acos, |x| -(-x * x + T::one()).sqrt().recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(0.2).derive().atan();
    /// assert!((res.0[0] - 0.197395559849881).abs() < 1e-10);
    /// assert!((res.0[1] - 0.961538461538461).abs() < 1e-10);
    /// assert!((res.0[2] - -0.369822485207101).abs() < 1e-10);
    /// assert!((res.0[3] - -1.56463359126081).abs() < 1e-10);
    /// assert!((res.0[4] - 3.93893771226498).abs() < 1e-10);
    /// assert!((res.0[5] - 11.9935603418325).abs() < 1e-10);
    /// ```

    #[inline]
    fn atan(&self) -> Self {
        self.apply_function(T::atan, |x| (x * x + T::one()).recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sinh();
    /// assert!((res.0[0] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[1] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[2] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[3] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[4] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[5] - 1.81065556732437).abs() < 1e-10);
    /// ```
    #[inline]
    fn sinh(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        let s = self.0[0].sinh();
        let c = self.0[0].cosh();
        for i in 0..self.0.len() {
            res.push({
                match i % 2 {
                    0 => s,
                    1 => c,
                    _ => unreachable!(),
                }
            });
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().cosh();
    /// assert!((res.0[0] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[1] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[2] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[3] - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.0[4] - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.0[5] - 1.50946135541217).abs() < 1e-10);
    /// ```
    #[inline]
    fn cosh(&self) -> Self {
        let mut res = SmallVec::with_capacity(self.0.len());
        let s = self.0[0].sinh();
        let c = self.0[0].cosh();
        for i in 0..self.0.len() {
            res.push({
                match i % 2 {
                    0 => c,
                    1 => s,
                    _ => unreachable!(),
                }
            });
        }
        D::chain_rule(self, HDScal(res))
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().tanh();
    /// assert!((res.0[0] - 0.833654607012155).abs() < 1e-10);
    /// assert!((res.0[1] - 0.305019996207409).abs() < 1e-10);
    /// assert!((res.0[2] - -0.508562650138273).abs() < 1e-10);
    /// assert!((res.0[3] - 0.661856796311429).abs() < 1e-10);
    /// assert!((res.0[4] - -0.172789269156221).abs() < 1e-10);
    /// assert!((res.0[5] - -2.87875913415137).abs() < 1e-10);
    /// ```
    #[inline]
    fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().asinh();
    /// assert!((res.0[0] - 1.01597313417969).abs() < 1e-10);
    /// assert!((res.0[1] - 0.640184399664480).abs() < 1e-10);
    /// assert!((res.0[2] - -0.314844786720236).abs() < 1e-10);
    /// assert!((res.0[3] - 0.202154439560807).abs() < 1e-10);
    /// assert!((res.0[4] - 0.0190379137361067).abs() < 1e-10);
    /// assert!((res.0[5] - -0.811191980094493).abs() < 1e-10);
    /// ```
    #[inline]
    fn asinh(&self) -> Self {
        self.apply_function(T::asinh, |x| (x * x + T::one()).sqrt().recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().acosh();
    /// assert!((res.0[0] - 0.622362503714779).abs() < 1e-10);
    /// assert!((res.0[1] - 1.50755672288882).abs() < 1e-10);
    /// assert!((res.0[2] - -4.11151833515132).abs() < 1e-10);
    /// assert!((res.0[3] - 30.2134301901271).abs() < 1e-10);
    /// assert!((res.0[4] - -374.623881363994).abs() < 1e-10);
    /// assert!((res.0[5] - 6533.90848124184).abs() < 1e-10);
    /// ```
    #[inline]
    fn acosh(&self) -> Self {
        self.apply_function(T::acosh, |x| (x * x - T::one()).sqrt().recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(0.2).derive().atanh();
    /// assert!((res.0[0] - 0.202732554054082).abs() < 1e-10);
    /// assert!((res.0[1] - 1.04166666666667).abs() < 1e-10);
    /// assert!((res.0[2] - 0.434027777777778).abs() < 1e-10);
    /// assert!((res.0[3] - 2.53182870370370).abs() < 1e-10);
    /// assert!((res.0[4] - 5.87745949074074).abs() < 1e-10);
    /// assert!((res.0[5] - 41.4436246141975).abs() < 1e-10);
    /// ```
    #[inline]
    fn atanh(&self) -> Self {
        self.apply_function(T::atanh, |x| (-x * x + T::one()).recip())
    }

    /// ```
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sph_j0();
    /// assert!((res.0[0] - 0.776699238306022).abs() < 1e-10);
    /// assert!((res.0[1] - -0.345284569857790).abs() < 1e-10);
    /// assert!((res.0[2] - -0.201224955209705).abs() < 1e-10);
    /// assert!((res.0[3] - 0.201097592627034).abs() < 1e-10);
    /// assert!((res.0[4] - 0.106373929549242).abs() < 1e-10);
    /// assert!((res.0[5] - -0.141259911057947).abs() < 1e-10);
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
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sph_j1();
    /// assert!((res.0[0] - 0.345284569857790).abs() < 1e-10);
    /// assert!((res.0[1] - 0.201224955209705).abs() < 1e-10);
    /// assert!((res.0[2] - -0.201097592627034).abs() < 1e-10);
    /// assert!((res.0[3] - -0.106373929549242).abs() < 1e-10);
    /// assert!((res.0[4] - 0.141259911057947).abs() < 1e-10);
    /// assert!((res.0[5] - 0.0703996830162854).abs() < 1e-10);
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
    /// # use num_hyperdual::hd_scal::{HDScal64, D5};
    /// # use num_hyperdual::DualNum;
    /// let res: HDScal64<D5> = HDScal64::new(1.2).derive().sph_j2();
    /// assert!((res.0[0] - 0.0865121863384538).abs() < 1e-10);
    /// assert!((res.0[1] - 0.129004104011656).abs() < 1e-10);
    /// assert!((res.0[2] - 0.0589484167190109).abs() < 1e-10);
    /// assert!((res.0[3] - -0.111341070273404).abs() < 1e-10);
    /// assert!((res.0[4] - -0.0524125597498053).abs() < 1e-10);
    /// assert!((res.0[5] - 0.0924200777676286).abs() < 1e-10);
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

impl<T: Float, D: Derivative<T>> Inv for HDScal<T, D> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

impl<T: Float, D: Derivative<T>> Inv for &HDScal<T, D> {
    type Output = HDScal<T, D>;
    fn inv(self) -> HDScal<T, D> {
        self.recip()
    }
}

/* iterator methods */
impl<T: Float, D: Derivative<T>> Sum for HDScal<T, D> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: 'a + Float, D: Derivative<T>> Sum<&'a HDScal<T, D>> for HDScal<T, D> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HDScal<T, D>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: Float, D: Derivative<T>> Product for HDScal<T, D> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: 'a + Float, D: Derivative<T>> Product<&'a HDScal<T, D>> for HDScal<T, D> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HDScal<T, D>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}
