use crate::DualNumMethods;
use num_traits::{Float, Inv, Num, One, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A hyper dual number.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug, Default)]
#[repr(C)]
pub struct HyperDual<T> {
    /// Real part of the hyper dual number
    pub re: T,
    /// Eps1 part
    pub eps1: T,
    /// Eps2 part
    pub eps2: T,
    /// Eps1eps2 part
    pub eps1eps2: T,
}

pub type HyperDual32 = HyperDual<f32>;
pub type HyperDual64 = HyperDual<f64>;

impl<T: Clone + Num> HyperDual<T> {
    /// Create a new HyperDual
    #[inline]
    pub fn new(re: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        HyperDual {
            re,
            eps1,
            eps2,
            eps1eps2,
        }
    }
}

impl<T: Float> DualNumMethods<T> for HyperDual<T> {
    fn re(&self) -> T {
        self.re
    }

    /// Returns `1/self`
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).recip();
    /// assert!((res.re - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.eps1 - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.eps2 - -0.694444444444445).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 1.15740740740741).abs() < 1e-10);
    /// ```
    #[inline]
    fn recip(&self) -> Self {
        if self.re == T::zero() {
            panic!("Cannot take reciprocal value of zero-valued `real`!");
        }
        let recip_re = self.re.recip();
        let recip_re2 = recip_re * recip_re;
        let two = T::one() + T::one();
        HyperDual::new(
            recip_re,
            -self.eps1 * recip_re2,
            -self.eps2 * recip_re2,
            (two * self.eps1 * self.eps2 * recip_re - self.eps1eps2) * recip_re2,
        )
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).exp();
    /// assert!((res.re - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps1 - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps2 - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp(&self) -> Self {
        let fx = self.re.exp();
        HyperDual::new(
            fx,
            self.eps1 * fx,
            self.eps2 * fx,
            self.eps1eps2 * fx + self.eps1 * self.eps2 * fx,
        )
    }

    /// Computes `e^(self)-1` in a way that is accurate even if the number is close to zero.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).exp_m1();
    /// assert!((res.re - 2.32011692273655).abs() < 1e-10);
    /// assert!((res.eps1 - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps2 - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp_m1(&self) -> Self {
        let fx = self.re.exp_m1();
        let dx = self.re.exp();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * dx,
        )
    }

    /// Computes `2^(self)`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).exp2();
    /// assert!((res.re - 2.29739670999407).abs() < 1e-10);
    /// assert!((res.eps1 - 1.59243405216008).abs() < 1e-10);
    /// assert!((res.eps2 - 1.59243405216008).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 1.10379117348241).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp2(&self) -> Self {
        let fx = self.re.exp2();
        let ln_two = (T::one() + T::one()).ln();
        let dx = fx * ln_two;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * ln_two) * dx,
        )
    }

    /// Computes the principal value of natural logarithm of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).ln();
    /// assert!((res.re - 0.182321556793955).abs() < 1e-10);
    /// assert!((res.eps1 - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.eps2 - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.694444444444445).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln(&self) -> Self {
        let fx = self.re.ln();
        let dx = self.re.recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * dx) * dx,
        )
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).log(4.2);
    /// assert!((res.re - 0.127045866345188).abs() < 1e-10);
    /// assert!((res.eps1 - 0.580685888982970).abs() < 1e-10);
    /// assert!((res.eps2 - 0.580685888982970).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.483904907485808).abs() < 1e-10);
    /// ```
    #[inline]
    fn log(&self, base: T) -> Self {
        let fx = self.re.log(base);
        let lnb = base.ln();
        let dx = (self.re * lnb).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * dx * lnb) * dx,
        )
    }

    /// Computes `ln(1+n)` more accurately than if the operations were performed separately.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).ln_1p();
    /// assert!((res.re - 0.788457360364270).abs() < 1e-10);
    /// assert!((res.eps1 - 0.454545454545455).abs() < 1e-10);
    /// assert!((res.eps2 - 0.454545454545455).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.206611570247934).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln_1p(&self) -> Self {
        let fx = self.re.ln_1p();
        let dx = (T::one() + self.re).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * dx) * dx,
        )
    }

    /// Computes the principal value of logarithm of `self` with basis 2.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).log2();
    /// assert!((res.re - 0.263034405833794).abs() < 1e-10);
    /// assert!((res.eps1 - 1.20224586740747).abs() < 1e-10);
    /// assert!((res.eps2 - 1.20224586740747).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -1.00187155617289).abs() < 1e-10);
    /// ```
    #[inline]
    fn log2(&self) -> Self {
        let fx = self.re.log2();
        let ln2 = (T::one() + T::one()).ln();
        let dx = (self.re * ln2).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * dx * ln2) * dx,
        )
    }

    /// Computes the principal value of logarithm of `self` with basis 10.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).log10();
    /// assert!((res.re - 0.0791812460476248).abs() < 1e-10);
    /// assert!((res.eps1 - 0.361912068252710).abs() < 1e-10);
    /// assert!((res.eps2 - 0.361912068252710).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.301593390210592).abs() < 1e-10);
    /// ```
    #[inline]
    fn log10(&self) -> Self {
        let fx = self.re.log10();
        let ln10 = T::from(10).unwrap().ln();
        let dx = (self.re * ln10).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * dx * ln10) * dx,
        )
    }

    /// Computes the principal value of the square root of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sqrt();
    /// assert!((res.re - 1.09544511501033).abs() < 1e-10);
    /// assert!((res.eps1 - 0.456435464587638).abs() < 1e-10);
    /// assert!((res.eps2 - 0.456435464587638).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.190181443578183).abs() < 1e-10);
    /// ```
    #[inline]
    fn sqrt(&self) -> Self {
        let fx = self.re.sqrt();
        let one = T::one();
        let half = (one + one).recip();
        let dx = fx.recip() * half;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * half / self.re) * dx,
        )
    }

    /// Computes the principal value of the cubic root of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).cbrt();
    /// assert!((res.re - 1.06265856918261).abs() < 1e-10);
    /// assert!((res.eps1 - 0.295182935884059).abs() < 1e-10);
    /// assert!((res.eps2 - 0.295182935884059).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.163990519935588).abs() < 1e-10);
    /// ```
    #[inline]
    fn cbrt(&self) -> Self {
        let fx = self.re.cbrt();
        let one = T::one();
        let third = (one + one + one).recip();
        let dx = fx / self.re * third;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * (one + one) * third / self.re) * dx,
        )
    }

    /// Raises `self` to a floating point power.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).powf(4.2);
    /// assert!((res.re - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.eps1 - 7.52712759108966).abs() < 1e-10);
    /// assert!((res.eps2 - 7.52712759108966).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 20.0723402429058).abs() < 1e-10);
    /// ```
    #[inline]
    fn powf(&self, exp: T) -> Self {
        let rec = self.re.recip();
        let fx = self.re.powf(exp);
        let dx = exp * fx * rec;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * (exp - T::one()) * rec) * dx,
        )
    }

    /// Raises `self` to an integer power.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).powi(4);
    /// assert!((res.re - 2.07360000000000).abs() < 1e-10);
    /// assert!((res.eps1 - 6.91200000000000).abs() < 1e-10);
    /// assert!((res.eps2 - 6.91200000000000).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 17.2800000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powi(&self, exp: i32) -> Self {
        let e = T::from(exp).unwrap();
        let e1 = T::from(exp - 1).unwrap();

        let pow = self.re.powi(exp - 2);
        let fx = pow * self.re * self.re;
        let dx = e * pow * self.re;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 * self.re + self.eps1 * self.eps2 * e1) * e * pow,
        )
    }

    /// Computes the sine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sin();
    /// assert!((res.re - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.eps1 - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.eps2 - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin(&self) -> Self {
        let (fx, dx) = self.re.sin_cos();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx - self.eps1 * self.eps2 * fx,
        )
    }

    /// Computes the cosine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).cos();
    /// assert!((res.re - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.eps1 - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.eps2 - -0.932039085967226).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.362357754476674).abs() < 1e-10);
    /// ```
    #[inline]
    fn cos(&self) -> Self {
        let fx = self.re.cos();
        let dx = -self.re.sin();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx - self.eps1 * self.eps2 * fx,
        )
    }

    /// Computes the sine and the cosine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let (res_sin, res_cos) = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sin_cos();
    /// assert!((res_sin.re - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.eps1 - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_sin.eps2 - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_sin.eps1eps2 - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.re - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.eps1 - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.eps2 - -0.932039085967226).abs() < 1e-10);
    /// assert!((res_cos.eps1eps2 - -0.362357754476674).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin_cos(&self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (
            HyperDual::new(
                s,
                self.eps1 * c,
                self.eps2 * c,
                self.eps1eps2 * c - self.eps1 * self.eps2 * s,
            ),
            HyperDual::new(
                c,
                -self.eps1 * s,
                -self.eps2 * s,
                -self.eps1eps2 * s - self.eps1 * self.eps2 * c,
            ),
        )
    }

    /// Computes the tangent of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).tan();
    /// assert!((res.re - 2.57215162212632).abs() < 1e-10);
    /// assert!((res.eps1 - 7.61596396720705).abs() < 1e-10);
    /// assert!((res.eps2 - 7.61596396720705).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 39.1788281446144).abs() < 1e-10);
    /// ```
    #[inline]
    fn tan(&self) -> Self {
        let fx = self.re.tan();
        let dx = fx * fx + T::one();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * (T::one() + T::one()) * fx) * dx,
        )
    }

    /// Computes the principal value of the inverse sine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(0.2, 1.0, 1.0, 0.0).asin();
    /// assert!((res.re - 0.201357920790331).abs() < 1e-10);
    /// assert!((res.eps1 - 1.02062072615966).abs() < 1e-10);
    /// assert!((res.eps2 - 1.02062072615966).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 0.212629317949929).abs() < 1e-10);
    /// ```
    #[inline]
    fn asin(&self) -> Self {
        let fx = self.re.asin();
        let dx = (T::one() - self.re * self.re).sqrt().recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * self.re * dx * dx) * dx,
        )
    }

    /// Computes the principal value of the inverse cosine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(0.2, 1.0, 1.0, 0.0).acos();
    /// assert!((res.re - 1.36943840600457).abs() < 1e-10);
    /// assert!((res.eps1 - -1.02062072615966).abs() < 1e-10);
    /// assert!((res.eps2 - -1.02062072615966).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.212629317949929).abs() < 1e-10);
    /// ```
    #[inline]
    fn acos(&self) -> Self {
        let fx = self.re.acos();
        let dx = -(T::one() - self.re * self.re).sqrt().recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * self.re * dx * dx) * dx,
        )
    }

    /// Computes the principal value of the inverse tangent of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(0.2, 1.0, 1.0, 0.0).atan();
    /// assert!((res.re - 0.197395559849881).abs() < 1e-10);
    /// assert!((res.eps1 - 0.961538461538461).abs() < 1e-10);
    /// assert!((res.eps2 - 0.961538461538461).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.369822485207101).abs() < 1e-10);
    /// ```
    #[inline]
    fn atan(&self) -> Self {
        let fx = self.re.atan();
        let dx = (T::one() + self.re * self.re).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * (T::one() + T::one()) * self.re * dx) * dx,
        )
    }

    /// Computes the hyperbolic sine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sinh();
    /// assert!((res.re - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.eps1 - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.eps2 - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 1.50946135541217).abs() < 1e-10);
    /// ```
    #[inline]
    fn sinh(&self) -> Self {
        let fx = self.re.sinh();
        let dx = self.re.cosh();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * fx,
        )
    }

    /// Computes the hyperbolic cosine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).cosh();
    /// assert!((res.re - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.eps1 - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.eps2 - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 1.81065556732437).abs() < 1e-10);
    /// ```
    #[inline]
    fn cosh(&self) -> Self {
        let fx = self.re.cosh();
        let dx = self.re.sinh();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * fx,
        )
    }

    /// Computes the hyperbolic tangent of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).tanh();
    /// assert!((res.re - 0.833654607012155).abs() < 1e-10);
    /// assert!((res.eps1 - 0.305019996207409).abs() < 1e-10);
    /// assert!((res.eps2 - 0.305019996207409).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.508562650138273).abs() < 1e-10);
    /// ```
    #[inline]
    fn tanh(&self) -> Self {
        let fx = self.re.tanh();
        let dx = T::one() - fx * fx;
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * (T::one() + T::one()) * fx) * dx,
        )
    }

    /// Computes the principal value of inverse hyperbolic sine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).asinh();
    /// assert!((res.re - 1.01597313417969).abs() < 1e-10);
    /// assert!((res.eps1 - 0.640184399664480).abs() < 1e-10);
    /// assert!((res.eps2 - 0.640184399664480).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.314844786720236).abs() < 1e-10);
    /// ```
    #[inline]
    fn asinh(&self) -> Self {
        let fx = self.re.asinh();
        let dx = (self.re * self.re + T::one()).sqrt().recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * self.re * dx * dx) * dx,
        )
    }

    /// Computes the principal value of inverse hyperbolic cosine of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).acosh();
    /// assert!((res.re - 0.622362503714779).abs() < 1e-10);
    /// assert!((res.eps1 - 1.50755672288882).abs() < 1e-10);
    /// assert!((res.eps2 - 1.50755672288882).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -4.11151833515132).abs() < 1e-10);
    /// ```
    #[inline]
    fn acosh(&self) -> Self {
        let fx = self.re.acosh();
        let dx = (self.re * self.re - T::one()).sqrt().recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 - self.eps1 * self.eps2 * self.re * dx * dx) * dx,
        )
    }

    /// Computes the principal value of inverse hyperbolic tangent of `self`.
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(0.2, 1.0, 1.0, 0.0).atanh();
    /// assert!((res.re - 0.202732554054082).abs() < 1e-10);
    /// assert!((res.eps1 - 1.04166666666667).abs() < 1e-10);
    /// assert!((res.eps2 - 1.04166666666667).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 0.434027777777778).abs() < 1e-10);
    /// ```
    #[inline]
    fn atanh(&self) -> Self {
        let fx = self.re.atanh();
        let dx = (T::one() - self.re * self.re).recip();
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            (self.eps1eps2 + self.eps1 * self.eps2 * (T::one() + T::one()) * self.re * dx) * dx,
        )
    }

    /// Computes the zeroth order spherical bessel function `j0(x)`
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sph_j0();
    /// assert!((res.re - 0.776699238306022).abs() < 1e-10);
    /// assert!((res.eps1 - -0.345284569857790).abs() < 1e-10);
    /// assert!((res.eps2 - -0.345284569857790).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.201224955209705).abs() < 1e-10);
    /// ```
    fn sph_j0(&self) -> Self {
        let (fx, dx, dx2) = if self.re.abs() < T::epsilon() {
            (
                T::one() - self.re * self.re / T::from(6.0).unwrap(),
                -self.re / T::from(3.0).unwrap(),
                -T::from(1.0 / 3.0).unwrap(),
            )
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let two = T::one() + T::one();
            (
                s * rec,
                (-s * rec + c) * rec,
                (two * (s * rec - c) * rec - s) * rec,
            )
        };
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * dx2,
        )
    }

    /// Computes the first order spherical bessel function `j1(x)`
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sph_j1();
    /// assert!((res.re - 0.345284569857790).abs() < 1e-10);
    /// assert!((res.eps1 - 0.201224955209705).abs() < 1e-10);
    /// assert!((res.eps2 - 0.201224955209705).abs() < 1e-10);
    /// assert!((res.eps1eps2 - -0.201097592627034).abs() < 1e-10);
    /// ```
    fn sph_j1(&self) -> Self {
        let (fx, dx, dx2) = if self.re.abs() < T::epsilon() {
            let one = T::one();
            let third = one / (one + one + one);
            (self.re * third, third, T::zero())
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let two = T::one() + T::one();
            (
                (s * rec - c) * rec,
                (two * (c - s * rec) * rec + s) * rec,
                ((two * (s * rec - c) * rec - s) * T::from(3.0).unwrap() * rec + c) * rec,
            )
        };
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * dx2,
        )
    }

    /// Computes the second order spherical bessel function `j2(x)`
    /// ```
    /// # use num_hyperdual::hyperdual::HyperDual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = HyperDual64::new(1.2, 1.0, 1.0, 0.0).sph_j2();
    /// assert!((res.re - 0.0865121863384538).abs() < 1e-10);
    /// assert!((res.eps1 - 0.129004104011656).abs() < 1e-10);
    /// assert!((res.eps2 - 0.129004104011656).abs() < 1e-10);
    /// assert!((res.eps1eps2 - 0.0589484167190109).abs() < 1e-10);
    /// ```
    fn sph_j2(&self) -> Self {
        let (fx, dx, dx2) = if self.re.abs() < T::epsilon() {
            (
                self.re * self.re / T::from(15.0).unwrap(),
                self.re / T::from(7.5).unwrap(),
                T::from(1.0 / 7.5).unwrap(),
            )
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let three = T::one() + T::one() + T::one();
            let nine = three * three;
            (
                (three * (s * rec - c) * rec - s) * rec,
                ((nine * (-s * rec + c) * rec + (three + T::one()) * s) * rec - c) * rec,
                (((T::from(36.0).unwrap() * (s * rec - c) * rec - T::from(17.0).unwrap() * s)
                    * rec
                    + T::from(5.0).unwrap() * c)
                    * rec
                    + s)
                    * rec,
            )
        };
        HyperDual::new(
            fx,
            self.eps1 * dx,
            self.eps2 * dx,
            self.eps1eps2 * dx + self.eps1 * self.eps2 * dx2,
        )
    }
}

impl<T: Float> Inv for HyperDual<T> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

impl<T: Float> Inv for &HyperDual<T> {
    type Output = HyperDual<T>;
    fn inv(self) -> HyperDual<T> {
        self.recip()
    }
}

impl<T: Zero> From<T> for HyperDual<T> {
    #[inline]
    fn from(re: T) -> Self {
        HyperDual {
            re: re,
            eps1: T::zero(),
            eps2: T::zero(),
            eps1eps2: T::zero(),
        }
    }
}

impl<'a, T: Clone + Zero> From<&'a T> for HyperDual<T> {
    #[inline]
    fn from(re: &T) -> Self {
        From::from(re.clone())
    }
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, T: Clone + Num> $imp<&'b HyperDual<T>> for &'a HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: &HyperDual<T>) -> HyperDual<T> {
                self.clone().$method(other.clone())
            }
        }
    };
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T: Clone + Num> $imp<HyperDual<T>> for &'a HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: HyperDual<T>) -> HyperDual<T> {
                self.clone().$method(other)
            }
        }
    };
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T: Clone + Num> $imp<&'a HyperDual<T>> for HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: &HyperDual<T>) -> Self {
                self.$method(other.clone())
            }
        }
    };
}

macro_rules! forward_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_ref_ref_binop!(impl $imp, $method);
        forward_ref_val_binop!(impl $imp, $method);
        forward_val_ref_binop!(impl $imp, $method);
    };
}

/* arithmetic */
forward_all_binop!(impl Add, add);

impl<T: Clone + Num> Add<HyperDual<T>> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn add(self, other: HyperDual<T>) -> Self {
        HyperDual::new(
            self.re.clone() + other.re.clone(),
            self.eps1.clone() + other.eps1.clone(),
            self.eps2.clone() + other.eps2.clone(),
            self.eps1eps2.clone() + other.eps1eps2.clone(),
        )
    }
}

forward_all_binop!(impl Sub, sub);

impl<T: Clone + Num> Sub<HyperDual<T>> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn sub(self, other: HyperDual<T>) -> Self {
        HyperDual::new(
            self.re.clone() - other.re.clone(),
            self.eps1.clone() - other.eps1.clone(),
            self.eps2.clone() - other.eps2.clone(),
            self.eps1eps2.clone() - other.eps1eps2.clone(),
        )
    }
}

forward_all_binop!(impl Mul, mul);

impl<T: Clone + Num> Mul<HyperDual<T>> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn mul(self, other: HyperDual<T>) -> Self {
        HyperDual::new(
            self.re.clone() * other.re.clone(),
            self.re.clone() * other.eps1.clone() + other.re.clone() * self.eps1.clone(),
            self.re.clone() * other.eps2.clone() + other.re.clone() * self.eps2.clone(),
            self.re.clone() * other.eps1eps2.clone()
                + other.re.clone() * self.eps1eps2.clone()
                + self.eps1.clone() * other.eps2.clone()
                + self.eps2.clone() * other.eps1.clone(),
        )
    }
}

forward_all_binop!(impl Div, div);

impl<T: Clone + Num> Div<HyperDual<T>> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn div(self, other: HyperDual<T>) -> Self {
        let inv = T::one() / other.re.clone();
        let inv2 = inv.clone() * inv.clone();
        HyperDual::new(
            self.re.clone() * inv.clone(),
            (self.eps1.clone() * other.re.clone() - self.re.clone() * other.eps1.clone())
                * inv2.clone(),
            (self.eps2.clone() * other.re.clone() - self.re.clone() * other.eps2.clone())
                * inv2.clone(),
            self.eps1eps2.clone() * inv.clone()
                - (self.re.clone() * other.eps1eps2.clone()
                    + self.eps1.clone() * other.eps2.clone()
                    + self.eps2.clone() * other.eps1.clone())
                    * inv2.clone()
                + (T::one() + T::one())
                    * self.re.clone()
                    * other.eps1.clone()
                    * other.eps2.clone()
                    * inv2.clone()
                    * inv.clone(),
        )
    }
}

/* Neg impl */
impl<T: Clone + Num + Neg<Output = T>> Neg for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn neg(self) -> Self {
        HyperDual::new(-self.re, -self.eps1, -self.eps2, -self.eps1eps2)
    }
}

impl<'a, T: Clone + Num + Neg<Output = T>> Neg for &'a HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn neg(self) -> HyperDual<T> {
        -self.clone()
    }
}

macro_rules! real_arithmetic {
    (@forward $imp:ident::$method:ident for $($real:ident),*) => (
        impl<'a, T: Clone + Num> $imp<&'a T> for HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: &T) -> Self {
                self.$method(other.clone())
            }
        }
        impl<'a, T: Clone + Num> $imp<T> for &'a HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: T) -> HyperDual<T> {
                self.clone().$method(other)
            }
        }
        impl<'a, 'b, T: Clone + Num> $imp<&'a T> for &'b HyperDual<T> {
            type Output = HyperDual<T>;

            #[inline]
            fn $method(self, other: &T) -> HyperDual<T> {
                self.clone().$method(other.clone())
            }
        }
        $(
            impl<'a> $imp<&'a HyperDual<$real>> for $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn $method(self, other: &HyperDual<$real>) -> HyperDual<$real> {
                    self.$method(other.clone())
                }
            }
            impl<'a> $imp<HyperDual<$real>> for &'a $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn $method(self, other: HyperDual<$real>) -> HyperDual<$real> {
                    self.clone().$method(other)
                }
            }
            impl<'a, 'b> $imp<&'a HyperDual<$real>> for &'b $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn $method(self, other: &HyperDual<$real>) -> HyperDual<$real> {
                    self.clone().$method(other.clone())
                }
            }
        )*
    );
    ($($real:ident),*) => (
        real_arithmetic!(@forward Add::add for $($real),*);
        real_arithmetic!(@forward Sub::sub for $($real),*);
        real_arithmetic!(@forward Mul::mul for $($real),*);
        real_arithmetic!(@forward Div::div for $($real),*);
        // real_arithmetic!(@forward Rem::rem for $($real),*);

        $(
            impl Add<HyperDual<$real>> for $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn add(self, other: HyperDual<$real>) -> HyperDual<$real> {
                    HyperDual::new(self + other.re, other.eps1, other.eps2, other.eps1eps2)
                }
            }

            impl Sub<HyperDual<$real>> for $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn sub(self, other: HyperDual<$real>) -> HyperDual<$real> {
                    HyperDual::new(self - other.re, Self::zero() - other.eps1, Self::zero() - other.eps2, Self::zero() - other.eps1eps2)
                }
            }

            impl Mul<HyperDual<$real>> for $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn mul(self, other: HyperDual<$real>) -> HyperDual<$real> {
                    HyperDual::new(self * other.re, self * other.eps1, self * other.eps2, self * other.eps1eps2)
                }
            }

            impl Div<HyperDual<$real>> for $real {
                type Output = HyperDual<$real>;

                #[inline]
                fn div(self, other: HyperDual<$real>) -> HyperDual<$real> {
                    self * other.recip()
                }
            }
        )*
    );
}

real_arithmetic!(f32, f64);

impl<T: Clone + Num> Add<T> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn add(self, other: T) -> Self {
        HyperDual::new(self.re + other, self.eps1, self.eps2, self.eps1eps2)
    }
}

impl<T: Clone + Num> Sub<T> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn sub(self, other: T) -> Self {
        HyperDual::new(self.re - other, self.eps1, self.eps2, self.eps1eps2)
    }
}

impl<T: Clone + Num> Mul<T> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn mul(self, other: T) -> Self {
        HyperDual::new(
            self.re * other.clone(),
            self.eps1 * other.clone(),
            self.eps2 * other.clone(),
            self.eps1eps2 * other,
        )
    }
}

impl<T: Clone + Num> Div<T> for HyperDual<T> {
    type Output = HyperDual<T>;

    #[inline]
    fn div(self, other: T) -> Self {
        let inv = T::one() / other;
        HyperDual::new(
            self.re * inv.clone(),
            self.eps1 * inv.clone(),
            self.eps2 * inv.clone(),
            self.eps1eps2 * inv,
        )
    }
}

/* constants */
impl<T: Clone + Num> Zero for HyperDual<T> {
    #[inline]
    fn zero() -> Self {
        HyperDual::new(Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps1.is_zero() && self.eps2.is_zero() && self.eps1eps2.is_zero()
    }
}

impl<T: Clone + Num> One for HyperDual<T> {
    #[inline]
    fn one() -> Self {
        HyperDual::new(One::one(), Zero::zero(), Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.eps1.is_zero() && self.eps2.is_zero() && self.eps1eps2.is_zero()
    }
}

/* string conversions */
impl<T> fmt::Display for HyperDual<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} + {}ε1 + {}ε2 + {}ε1ε2",
            self.re, self.eps1, self.eps2, self.eps1eps2
        )
    }
}

/* iterator methods */
impl<T: Num + Clone> Sum for HyperDual<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: 'a + Num + Clone> Sum<&'a HyperDual<T>> for HyperDual<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HyperDual<T>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: Num + Clone> Product for HyperDual<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: 'a + Num + Clone> Product<&'a HyperDual<T>> for HyperDual<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a HyperDual<T>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}
