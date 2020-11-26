use crate::{Dual32, Dual64, DualNum, DualNumMethods};
use num_traits::{Float, FromPrimitive, Inv, Num, One, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HD3<T, F = T>(pub [T; 4], PhantomData<F>);

pub type HD3_32 = HD3<f32>;
pub type HD3_64 = HD3<f64>;
pub type HD3Dual32 = HD3<Dual32, f32>;
pub type HD3Dual64 = HD3<Dual64, f64>;

impl<T: DualNum<F>, F: Float> HD3<T, F> {
    #[inline]
    pub fn new(x: [T; 4]) -> Self {
        Self(x, PhantomData)
    }

    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new([re, T::zero(), T::zero(), T::zero()])
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
        Self::new([
            f0,
            f1 * self.0[1],
            f2 * self.0[1] * self.0[1] + f1 * self.0[2],
            f3 * self.0[1] * self.0[1] * self.0[1]
                + three * f2 * self.0[1] * self.0[2]
                + f1 * self.0[3],
        ])
    }
}

impl<T: DualNum<F>, F: Float> From<F> for HD3<T, F> {
    fn from(float: F) -> Self {
        let mut res = [T::zero(); 4];
        res[0] = T::from(float);
        Self::new(res)
    }
}

impl<T: DualNum<F>, F: Float> Zero for HD3<T, F> {
    #[inline]
    fn zero() -> Self {
        Self::new([T::zero(); 4])
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| x.is_zero())
    }
}

impl<T: DualNum<F>, F: Float> One for HD3<T, F> {
    #[inline]
    fn one() -> Self {
        let mut res = [T::zero(); 4];
        res[0] = T::one();
        Self::new(res)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.iter().skip(1).all(|x| x.is_zero()) && self.0[0].is_one()
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a HD3<T, F>> for &'b HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn mul(self, rhs: &HD3<T, F>) -> HD3<T, F> {
        let two = T::one() + T::one();
        let three = two + T::one();
        HD3::new([
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

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a HD3<T, F>> for &'b HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn div(self, rhs: &HD3<T, F>) -> HD3<T, F> {
        let rec = T::one() / rhs.0[0];
        let f0 = rec;
        let f1 = -f0 * rec;
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
        self * rhs.chain_rule(f0, f1, f2, f3)
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Add<&'a HD3<T, F>> for &'b HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn add(self, rhs: &HD3<T, F>) -> HD3<T, F> {
        HD3::new([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Sub<&'a HD3<T, F>> for &'b HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn sub(self, rhs: &HD3<T, F>) -> HD3<T, F> {
        HD3::new([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Rem<&'a HD3<T, F>> for &'b HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn rem(self, _rhs: &HD3<T, F>) -> HD3<T, F> {
        unimplemented!()
    }
}

forward_binop!(HD3, Mul, *, mul);
forward_binop!(HD3, Div, /, div);
forward_binop!(HD3, Add, +, add);
forward_binop!(HD3, Sub, -, sub);
forward_binop!(HD3, Rem, %, rem);

/* Neg impl */
impl<T: DualNum<F>, F: Float> Neg for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.neg());
        self
    }
}

impl<T: DualNum<F>, F: Float> Neg for &HD3<T, F> {
    type Output = HD3<T, F>;
    #[inline]
    fn neg(self) -> Self::Output {
        HD3::new([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

/* scalar operations */
impl<T: DualNum<F>, F: Float> Mul<F> for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn mul(self, other: F) -> Self {
        HD3::new([
            self.0[0] * other,
            self.0[1] * other,
            self.0[2] * other,
            self.0[3] * other,
        ])
    }
}

impl<T: DualNum<F>, F: Float> Div<F> for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn div(self, other: F) -> Self {
        self * other.recip()
    }
}

impl<T: DualNum<F>, F: Float> Add<F> for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn add(self, other: F) -> Self {
        HD3::new([self.0[0] + other, self.0[1], self.0[2], self.0[3]])
    }
}

impl<T: DualNum<F>, F: Float> Sub<F> for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn sub(self, other: F) -> Self {
        HD3::new([self.0[0] - other, self.0[1], self.0[2], self.0[3]])
    }
}

impl<T: DualNum<F>, F: Float> Rem<F> for HD3<T, F> {
    type Output = Self;
    #[inline]
    fn rem(self, _other: F) -> Self {
        unimplemented!()
    }
}

/* assign operations */
impl<T: DualNum<F>, F: Float> MulAssign for HD3<T, F> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<T: DualNum<F>, F: Float> MulAssign<F> for HD3<T, F> {
    #[inline]
    fn mul_assign(&mut self, other: F) {
        self.0[0] *= other;
        self.0[1] *= other;
        self.0[2] *= other;
        self.0[3] *= other;
    }
}

impl<T: DualNum<F>, F: Float> DivAssign for HD3<T, F> {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl<T: DualNum<F>, F: Float> DivAssign<F> for HD3<T, F> {
    #[inline]
    fn div_assign(&mut self, other: F) {
        self.0[0] /= other;
        self.0[1] /= other;
        self.0[2] /= other;
        self.0[3] /= other;
    }
}

impl<T: DualNum<F>, F: Float> AddAssign for HD3<T, F> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
    }
}

impl<T: DualNum<F>, F: Float> AddAssign<F> for HD3<T, F> {
    #[inline]
    fn add_assign(&mut self, other: F) {
        self.0[0] += other;
    }
}

impl<T: DualNum<F>, F: Float> SubAssign for HD3<T, F> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
    }
}

impl<T: DualNum<F>, F: Float> SubAssign<F> for HD3<T, F> {
    #[inline]
    fn sub_assign(&mut self, other: F) {
        self.0[0] -= other;
    }
}

impl<T: DualNum<F>, F: Float> RemAssign for HD3<T, F> {
    #[inline]
    fn rem_assign(&mut self, _other: Self) {
        unimplemented!()
    }
}

impl<T: DualNum<F>, F: Float> RemAssign<F> for HD3<T, F> {
    #[inline]
    fn rem_assign(&mut self, _other: F) {
        unimplemented!()
    }
}

impl<T: DualNum<F>, F: Float> DualNumMethods<F> for HD3<T, F> {
    const NDERIV: usize = T::NDERIV + 3;

    #[inline]
    fn re(&self) -> F {
        self.0[0].re()
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
        let f2 = f1 * rec * F::from(-2.0).unwrap();
        let f3 = f2 * rec * F::from(-3.0).unwrap();
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
    /// let res0 = HD3_64::from(0.0).derive().powi(0);
    /// assert!((res0.0[0] - 1.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res1 = HD3_64::from(0.0).derive().powi(1);
    /// assert!((res1.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[1] - 1.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res2 = HD3_64::from(0.0).derive().powi(2);
    /// assert!((res2.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[2] - 2.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res3 = HD3_64::from(0.0).derive().powi(3);
    /// assert!((res3.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[3] - 6.00000000000000).abs() < 1e-10);
    /// let res4 = HD3_64::from(0.0).derive().powi(4);
    /// assert!((res4.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[3] - 0.00000000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powi(&self, exp: i32) -> Self {
        match exp {
            0 => HD3::one(),
            1 => *self,
            2 => self * self,
            _ => {
                let f3 = self.0[0].powi(exp - 3);
                let f2 = f3 * self.0[0];
                let f1 = f2 * self.0[0];
                let f0 = f1 * self.0[0];
                self.chain_rule(
                    f0,
                    f1 * F::from(exp).unwrap(),
                    f2 * F::from(exp * (exp - 1)).unwrap(),
                    f3 * F::from(exp * (exp - 1) * (exp - 2)).unwrap(),
                )
            }
        }
    }

    /// ```
    /// # use num_hyperdual::hd3::HD3_64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HD3_64::from(1.2).derive().powf(4.2);
    /// assert!((res.0[0] - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.0[1] - 7.52712759108966).abs() < 1e-10);
    /// assert!((res.0[2] - 20.0723402429058).abs() < 1e-10);
    /// assert!((res.0[3] - 36.7992904453272).abs() < 1e-10);
    /// let res0 = HD3_64::from(0.0).derive().powf(0.0);
    /// assert!((res0.0[0] - 1.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res0.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res1 = HD3_64::from(0.0).derive().powf(1.0);
    /// assert!((res1.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[1] - 1.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res1.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res2 = HD3_64::from(0.0).derive().powf(2.0);
    /// assert!((res2.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[2] - 2.00000000000000).abs() < 1e-10);
    /// assert!((res2.0[3] - 0.00000000000000).abs() < 1e-10);
    /// let res3 = HD3_64::from(0.0).derive().powf(3.0);
    /// assert!((res3.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res3.0[3] - 6.00000000000000).abs() < 1e-10);
    /// let res4 = HD3_64::from(0.0).derive().powf(4.0);
    /// assert!((res4.0[0] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[1] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[2] - 0.00000000000000).abs() < 1e-10);
    /// assert!((res4.0[3] - 0.00000000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powf(&self, n: F) -> Self {
        if n.is_zero() {
            Self::one()
        } else if n.is_one() {
            *self
        } else if n - F::one() - F::one() < F::epsilon() {
            self * self
        } else {
            let n1 = n - F::one();
            let n2 = n1 - F::one();
            let n3 = n2 - F::one();
            let pow3 = self.0[0].powf(n3);
            let f0 = pow3 * self.0[0] * self.0[0] * self.0[0];
            let f1 = pow3 * self.0[0] * self.0[0] * n;
            let f2 = pow3 * self.0[0] * n * n1;
            let f3 = pow3 * n * n1 * n2;
            self.chain_rule(f0, f1, f2, f3)
        }
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
        let half = F::from(0.5).unwrap();
        let f0 = self.0[0].sqrt();
        let f1 = f0 * rec * half;
        let f2 = -f1 * rec * half;
        let f3 = f2 * rec * (-F::one() - half);
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
        let third = F::from(1.0 / 3.0).unwrap();
        let f0 = self.0[0].cbrt();
        let f1 = f0 * rec * third;
        let f2 = f1 * rec * (third - F::one());
        let f3 = f2 * rec * (third - F::one() - F::one());
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
        let ln2 = F::from(2.0).unwrap().ln();
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
        let f3 = f2 * rec * F::from(-2.0).unwrap();
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
    fn log(&self, base: F) -> Self {
        let rec = self.0[0].recip();
        let f0 = self.0[0].log(base);
        let f1 = rec / base.ln();
        let f2 = -f1 * rec;
        let f3 = f2 * rec * F::from(-2.0).unwrap();
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
        let f1 = rec / (F::one() + F::one()).ln();
        let f2 = -f1 * rec;
        let f3 = f2 * rec * F::from(-2.0).unwrap();
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
        let f1 = rec / F::from(10.0).unwrap().ln();
        let f2 = -f1 * rec;
        let f3 = f2 * rec * F::from(-2.0).unwrap();
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
        let rec = (self.0[0] + F::one()).recip();
        let f0 = self.0[0].ln_1p();
        let f1 = rec;
        let f2 = -f1 * rec;
        let f3 = f2 * rec * F::from(-2.0).unwrap();
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
        let f3 = (self.0[0] * self.0[0] * (F::one() + F::one()) + F::one()) * f1 * rec * rec;
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
        let f3 = (self.0[0] * self.0[0] * (F::one() + F::one()) + F::one()) * f1 * rec * rec;
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
        let two = F::one() + F::one();
        let rec = (T::one() + self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].atan();
        let f1 = rec;
        let f2 = -self.0[0] * f1 * rec * two;
        let f3 = (self.0[0] * self.0[0] * F::from(6.0).unwrap() - two) * f1 * rec * rec;
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
        let f3 = (self.0[0] * self.0[0] * (F::one() + F::one()) - F::one()) * f1 * rec * rec;
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
        let rec = (self.0[0] * self.0[0] - F::one()).recip();
        let f0 = self.0[0].acosh();
        let f1 = rec.sqrt();
        let f2 = -self.0[0] * f1 * rec;
        let f3 = (self.0[0] * self.0[0] * (F::one() + F::one()) + F::one()) * f1 * rec * rec;
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
        let two = F::one() + F::one();
        let rec = (T::one() - self.0[0] * self.0[0]).recip();
        let f0 = self.0[0].atanh();
        let f1 = rec;
        let f2 = self.0[0] * f1 * rec * two;
        let f3 = (self.0[0] * self.0[0] * F::from(6.0).unwrap() + two) * f1 * rec * rec;
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
        if self.re() < F::epsilon() {
            Self::one() - self * self / F::from(6.0).unwrap()
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
        if self.re() < F::epsilon() {
            *self / F::from(3.0).unwrap()
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
        if self.re() < F::epsilon() {
            self * self / F::from(15.0).unwrap()
        } else {
            let (s, c) = self.sin_cos();
            let s2 = self * self;
            ((&s - self * c) * F::from(3.0).unwrap() - &s2 * s) / (s2 * self)
        }
    }
}

impl<T: DualNum<F>, F: Float> Inv for HD3<T, F> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

impl<T: DualNum<F>, F: Float> Inv for &HD3<T, F> {
    type Output = HD3<T, F>;
    fn inv(self) -> HD3<T, F> {
        self.recip()
    }
}

/* iterator methods */
impl<T: DualNum<F>, F: Float> Sum for HD3<T, F> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: DualNum<F>, F: 'a + Float> Sum<&'a HD3<T, F>> for HD3<T, F> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HD3<T, F>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: DualNum<F>, F: Float> Product for HD3<T, F> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: DualNum<F>, F: 'a + Float> Product<&'a HD3<T, F>> for HD3<T, F> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HD3<T, F>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for HD3<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "re: {}, dx: {}, dx2: {}, dx3: {}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

impl<T: DualNum<F>, F: Float> Num for HD3<T, F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!()
    }
}

impl_from_primitive!(HD3);
