use crate::{DualNum, DualNumMethods};
use num_traits::{Float, FloatConst, FromPrimitive, Inv, Num, One, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A dual number.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[repr(C)]
pub struct Dual<T, F = T> {
    /// Real part of the dual number
    pub re: T,
    /// Eps part
    pub eps: T,
    f: PhantomData<F>,
}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

impl<T, F> Dual<T, F> {
    /// Create a new dual number
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Dual {
            re,
            eps,
            f: PhantomData,
        }
    }
}

impl<T: Zero, F> Dual<T, F> {
    /// Create a new dual number from the real part
    #[inline]
    pub fn from_re(re: T) -> Self {
        Dual::new(re, T::zero())
    }
}

impl<T: DualNum<F>, F> From<F> for Dual<T, F> {
    fn from(float: F) -> Self {
        Dual::new(T::from(float), T::zero())
    }
}

impl<F: Float, T: DualNum<F>> DualNumMethods<F> for Dual<T, F> {
    const NDERIV: usize = T::NDERIV + 1;

    #[inline]
    fn re(&self) -> F {
        self.re.re()
    }

    /// Returns `1/self`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).recip();
    /// assert!((res.re - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.eps - -0.694444444444445).abs() < 1e-10);
    /// ```
    #[inline]
    fn recip(&self) -> Self {
        let recip_re = self.re.recip();
        Dual::new(recip_re, -self.eps * recip_re * recip_re)
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).exp();
    /// assert!((res.re - 3.32011692273655).abs() < 1e-10);
    /// assert!((res.eps - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp(&self) -> Self {
        let fx = self.re.exp();
        Dual::new(fx, self.eps * fx)
    }

    /// Computes `e^(self)-1` in a way that is accurate even if the number is close to zero.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).exp_m1();
    /// assert!((res.re - 2.32011692273655).abs() < 1e-10);
    /// assert!((res.eps - 3.32011692273655).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp_m1(&self) -> Self {
        let fx = self.re.exp_m1();
        let dx = self.re.exp();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes `2^(self)`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).exp2();
    /// assert!((res.re - 2.29739670999407).abs() < 1e-10);
    /// assert!((res.eps - 1.59243405216008).abs() < 1e-10);
    /// ```
    #[inline]
    fn exp2(&self) -> Self {
        let fx = self.re.exp2();
        let ln_two = (T::one() + T::one()).ln();
        Dual::new(fx, self.eps * fx * ln_two)
    }

    /// Computes the principal value of natural logarithm of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).ln();
    /// assert!((res.re - 0.182321556793955).abs() < 1e-10);
    /// assert!((res.eps - 0.833333333333333).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln(&self) -> Self {
        let fx = self.re.ln();
        let dx = self.re.recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).log(4.2);
    /// assert!((res.re - 0.127045866345188).abs() < 1e-10);
    /// assert!((res.eps - 0.580685888982970).abs() < 1e-10);
    /// ```
    #[inline]
    fn log(&self, base: F) -> Self {
        let fx = self.re.log(base);
        let dx = (self.re * base.ln()).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes `ln(1+n)` more accurately than if the operations were performed separately.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).ln_1p();
    /// assert!((res.re - 0.788457360364270).abs() < 1e-10);
    /// assert!((res.eps - 0.454545454545455).abs() < 1e-10);
    /// ```
    #[inline]
    fn ln_1p(&self) -> Self {
        let fx = self.re.ln_1p();
        let dx = (T::one() + self.re).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of logarithm of `self` with basis 2.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).log2();
    /// assert!((res.re - 0.263034405833794).abs() < 1e-10);
    /// assert!((res.eps - 1.20224586740747).abs() < 1e-10);
    /// ```
    #[inline]
    fn log2(&self) -> Self {
        let fx = self.re.log2();
        let dx = ((T::one() + T::one()).ln() * self.re).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of logarithm of `self` with basis 10.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).log10();
    /// assert!((res.re - 0.0791812460476248).abs() < 1e-10);
    /// assert!((res.eps - 0.361912068252710).abs() < 1e-10);
    /// ```
    #[inline]
    fn log10(&self) -> Self {
        let fx = self.re.log10();
        let dx = (self.re * F::from(10).unwrap().ln()).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the square root of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sqrt();
    /// assert!((res.re - 1.09544511501033).abs() < 1e-10);
    /// assert!((res.eps - 0.456435464587638).abs() < 1e-10);
    /// ```
    #[inline]
    fn sqrt(&self) -> Self {
        let fx = self.re.sqrt();
        let one = T::one();
        let half = (one + one).recip();
        let dx = fx.recip() * half;
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the cubic root of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).cbrt();
    /// assert!((res.re - 1.06265856918261).abs() < 1e-10);
    /// assert!((res.eps - 0.295182935884059).abs() < 1e-10);
    /// ```
    #[inline]
    fn cbrt(&self) -> Self {
        let fx = self.re.cbrt();
        let one = T::one();
        let third = (one + one + one).recip();
        let dx = fx / self.re * third;
        Dual::new(fx, self.eps * dx)
    }

    /// Raises `self` to a floating point power.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).powf(4.2);
    /// assert!((res.re - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.eps - 7.52712759108966).abs() < 1e-10);
    ///
    /// assert!(Dual64::new(1.2, 1.0).powf(0.0) == Dual64::new(1.0, 0.0));
    /// assert!(Dual64::new(1.2, 1.0).powf(1.0) == Dual64::new(1.2, 1.0));
    /// assert!(Dual64::new(0.0, 1.0).powf(0.0) == Dual64::new(1.0, 0.0));
    /// assert!(Dual64::new(0.0, 1.0).powf(1.0) == Dual64::new(0.0, 1.0));
    /// assert!(Dual64::new(0.0, 1.0).powf(4.2) == Dual64::new(0.0, 0.0));
    /// ```
    #[inline]
    fn powf(&self, exp: F) -> Self {
        if exp.is_zero() {
            Self::one()
        } else if exp.is_one() {
            *self
        } else {
            let pow = self.re.powf(exp - F::one());
            let fx = pow * self.re;
            let dx = pow * exp;
            Dual::new(fx, self.eps * dx)
        }
    }

    /// Raises `self` to an integer power.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).powi(4);
    /// assert!((res.re - 2.07360000000000).abs() < 1e-10);
    /// assert!((res.eps - 6.91200000000000).abs() < 1e-10);
    ///
    /// assert!(Dual64::new(1.2,1.0).powi(0) == Dual64::new(1.0, 0.0));
    /// assert!(Dual64::new(1.2,1.0).powi(1) == Dual64::new(1.2, 1.0));
    /// assert!(Dual64::new(0.0,1.0).powi(0) == Dual64::new(1.0, 0.0));
    /// assert!(Dual64::new(0.0,1.0).powi(1) == Dual64::new(0.0, 1.0));
    /// assert!(Dual64::new(0.0,1.0).powi(4) == Dual64::new(0.0, 0.0));
    /// ```
    #[inline]
    fn powi(&self, exp: i32) -> Self {
        match exp {
            0 => Dual::one(),
            1 => *self,
            _ => {
                let pow = self.re.powi(exp - 1);
                let fx = pow * self.re;
                let dx = T::from(F::from(exp).unwrap()) * pow;
                Dual::new(fx, self.eps * dx)
            }
        }
    }

    /// Computes the sine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sin();
    /// assert!((res.re - 0.932039085967226).abs() < 1e-10);
    /// assert!((res.eps - 0.362357754476674).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin(&self) -> Self {
        let (fx, dx) = self.re.sin_cos();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the cosine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).cos();
    /// assert!((res.re - 0.362357754476674).abs() < 1e-10);
    /// assert!((res.eps - -0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn cos(&self) -> Self {
        let fx = self.re.cos();
        let dx = -self.re.sin();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the sine and the cosine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let (res_sin, res_cos) = Dual64::new(1.2, 1.0).sin_cos();
    /// assert!((res_sin.re - 0.932039085967226).abs() < 1e-10);
    /// assert!((res_sin.eps - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.re - 0.362357754476674).abs() < 1e-10);
    /// assert!((res_cos.eps - -0.932039085967226).abs() < 1e-10);
    /// ```
    #[inline]
    fn sin_cos(&self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (Dual::new(s, self.eps * c), Dual::new(c, -self.eps * s))
    }

    /// Computes the tangent of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).tan();
    /// assert!((res.re - 2.57215162212632).abs() < 1e-10);
    /// assert!((res.eps - 7.61596396720705).abs() < 1e-10);
    /// ```
    #[inline]
    fn tan(&self) -> Self {
        let fx = self.re.tan();
        let dx = fx * fx + T::one();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the inverse sine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(0.2, 1.0).asin();
    /// assert!((res.re - 0.201357920790331).abs() < 1e-10);
    /// assert!((res.eps - 1.02062072615966).abs() < 1e-10);
    /// ```
    #[inline]
    fn asin(&self) -> Self {
        let fx = self.re.asin();
        let dx = (T::one() - self.re * self.re).sqrt().recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the inverse cosine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(0.2, 1.0).acos();
    /// assert!((res.re - 1.36943840600457).abs() < 1e-10);
    /// assert!((res.eps - -1.02062072615966).abs() < 1e-10);
    /// ```
    #[inline]
    fn acos(&self) -> Self {
        let fx = self.re.acos();
        let dx = -(T::one() - self.re * self.re).sqrt().recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the inverse tangent of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(0.2, 1.0).atan();
    /// assert!((res.re - 0.197395559849881).abs() < 1e-10);
    /// assert!((res.eps - 0.961538461538461).abs() < 1e-10);
    /// ```
    #[inline]
    fn atan(&self) -> Self {
        let fx = self.re.atan();
        let dx = (T::one() + self.re * self.re).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the hyperbolic sine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sinh();
    /// assert!((res.re - 1.50946135541217).abs() < 1e-10);
    /// assert!((res.eps - 1.81065556732437).abs() < 1e-10);
    /// ```
    #[inline]
    fn sinh(&self) -> Self {
        let fx = self.re.sinh();
        let dx = self.re.cosh();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the hyperbolic cosine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).cosh();
    /// assert!((res.re - 1.81065556732437).abs() < 1e-10);
    /// assert!((res.eps - 1.50946135541217).abs() < 1e-10);
    /// ```
    #[inline]
    fn cosh(&self) -> Self {
        let fx = self.re.cosh();
        let dx = self.re.sinh();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the hyperbolic tangent of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).tanh();
    /// assert!((res.re - 0.833654607012155).abs() < 1e-10);
    /// assert!((res.eps - 0.305019996207409).abs() < 1e-10);
    /// ```
    #[inline]
    fn tanh(&self) -> Self {
        let fx = self.re.tanh();
        let dx = T::one() - fx * fx;
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of inverse hyperbolic sine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).asinh();
    /// assert!((res.re - 1.01597313417969).abs() < 1e-10);
    /// assert!((res.eps - 0.640184399664480).abs() < 1e-10);
    /// ```
    #[inline]
    fn asinh(&self) -> Self {
        let fx = self.re.asinh();
        let dx = (self.re * self.re + T::one()).sqrt().recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of inverse hyperbolic cosine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).acosh();
    /// assert!((res.re - 0.622362503714779).abs() < 1e-10);
    /// assert!((res.eps - 1.50755672288882).abs() < 1e-10);
    /// ```
    #[inline]
    fn acosh(&self) -> Self {
        let fx = self.re.acosh();
        let dx = (self.re * self.re - T::one()).sqrt().recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of inverse hyperbolic tangent of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(0.2, 1.0).atanh();
    /// assert!((res.re - 0.202732554054082).abs() < 1e-10);
    /// assert!((res.eps - 1.04166666666667).abs() < 1e-10);
    /// ```
    #[inline]
    fn atanh(&self) -> Self {
        let fx = self.re.atanh();
        let dx = (T::one() - self.re * self.re).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the zeroth order spherical bessel function `j0(x)`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sph_j0();
    /// assert!((res.re - 0.776699238306022).abs() < 1e-10);
    /// assert!((res.eps - -0.345284569857790).abs() < 1e-10);
    /// ```
    fn sph_j0(&self) -> Self {
        let (fx, dx) = if self.re().abs() < F::epsilon() {
            (
                T::one() - self.re * self.re / F::from(6.0).unwrap(),
                -self.re / F::from(3.0).unwrap(),
            )
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            (s * rec, (-s * rec + c) * rec)
        };
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the first order spherical bessel function `j1(x)`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sph_j1();
    /// assert!((res.re - 0.345284569857790).abs() < 1e-10);
    /// assert!((res.eps - 0.201224955209705).abs() < 1e-10);
    /// ```
    fn sph_j1(&self) -> Self {
        let (fx, dx) = if self.re().abs() < F::epsilon() {
            let one = T::one();
            let third = one / (one + one + one);
            (self.re * third, third)
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let two = F::one() + F::one();
            ((s * rec - c) * rec, ((c - s * rec) * two * rec + s) * rec)
        };
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the second order spherical bessel function `j2(x)`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNumMethods;
    /// let res = Dual64::new(1.2, 1.0).sph_j2();
    /// assert!((res.re - 0.0865121863384538).abs() < 1e-10);
    /// assert!((res.eps - 0.129004104011656).abs() < 1e-10);
    /// ```
    fn sph_j2(&self) -> Self {
        let (fx, dx) = if self.re().abs() < F::epsilon() {
            (
                self.re * self.re / F::from(15.0).unwrap(),
                self.re / F::from(7.5).unwrap(),
            )
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let three = F::one() + F::one() + F::one();
            let nine = three * three;
            (
                ((s * rec - c) * three * rec - s) * rec,
                (((-s * rec + c) * nine * rec + s * (three + F::one())) * rec - c) * rec,
            )
        };
        Dual::new(fx, self.eps * dx)
    }
}

impl<T: DualNum<F>, F: Float> Inv for Dual<T, F> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

/* arithmetic */

impl<'a, 'b, T: DualNum<F>, F: Float> Mul<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn mul(self, other: &Dual<T, F>) -> Dual<T, F> {
        Dual::new(
            self.re * other.re,
            self.re * other.eps + other.re * self.eps,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Div<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn div(self, other: &Dual<T, F>) -> Dual<T, F> {
        let inv = T::one() / other.re;
        let inv2 = inv * inv;
        Dual::new(
            self.re * inv,
            (self.eps * other.re - self.re * other.eps) * inv2,
        )
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Add<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn add(self, other: &Dual<T, F>) -> Dual<T, F> {
        Dual::new(self.re + other.re, self.eps + other.eps)
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Sub<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn sub(self, other: &Dual<T, F>) -> Dual<T, F> {
        Dual::new(self.re - other.re, self.eps - other.eps)
    }
}

impl<'a, 'b, T: DualNum<F>, F: Float> Rem<&'a Dual<T, F>> for &'b Dual<T, F> {
    type Output = Dual<T, F>;
    #[inline]
    fn rem(self, _other: &Dual<T, F>) -> Dual<T, F> {
        unimplemented!()
    }
}

forward_binop!(Dual, Mul, *, mul);
forward_binop!(Dual, Div, /, div);
forward_binop!(Dual, Add, +, add);
forward_binop!(Dual, Sub, -, sub);
forward_binop!(Dual, Rem, %, rem);

/* Neg impl */
impl<T: DualNum<F>, F: Float> Neg for Dual<T, F> {
    type Output = Dual<T, F>;

    #[inline]
    fn neg(self) -> Self {
        Dual::new(-self.re, -self.eps)
    }
}

impl<'a, T: DualNum<F>, F: Float> Neg for &'a Dual<T, F> {
    type Output = Dual<T, F>;

    #[inline]
    fn neg(self) -> Dual<T, F> {
        -*self
    }
}

/* scalar operations */
impl<T: DualNum<F>, F: Float> Mul<F> for Dual<T, F> {
    type Output = Self;
    #[inline]
    fn mul(self, other: F) -> Self {
        Dual::new(self.re * other, self.eps * other)
    }
}

impl<T: DualNum<F>, F: Float> Div<F> for Dual<T, F> {
    type Output = Self;
    #[inline]
    fn div(self, other: F) -> Self {
        self * other.recip()
    }
}

impl<T: DualNum<F>, F: Float> Add<F> for Dual<T, F> {
    type Output = Self;
    #[inline]
    fn add(self, other: F) -> Self {
        Dual::new(self.re + other, self.eps)
    }
}

impl<T: DualNum<F>, F: Float> Sub<F> for Dual<T, F> {
    type Output = Self;
    #[inline]
    fn sub(self, other: F) -> Self {
        Dual::new(self.re - other, self.eps)
    }
}

impl<T: DualNum<F>, F: Float> Rem<F> for Dual<T, F> {
    type Output = Self;
    #[inline]
    fn rem(self, _other: F) -> Self {
        unimplemented!()
    }
}

/* assign operations */
impl<T: DualNum<F>, F: Float> MulAssign for Dual<T, F> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.eps = self.eps * other.re + self.re * other.eps;
        self.re *= other.re;
    }
}

impl<T: DualNum<F>, F: Float> MulAssign<F> for Dual<T, F> {
    #[inline]
    fn mul_assign(&mut self, other: F) {
        self.re *= other;
        self.eps *= other;
    }
}

impl<T: DualNum<F>, F: Float> DivAssign for Dual<T, F> {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.eps = (self.eps * other.re - self.re * other.eps) / (other.re * other.re);
        self.re /= other.re;
    }
}

impl<T: DualNum<F>, F: Float> DivAssign<F> for Dual<T, F> {
    #[inline]
    fn div_assign(&mut self, other: F) {
        self.re /= other;
        self.eps /= other;
    }
}

impl<T: DualNum<F>, F: Float> AddAssign for Dual<T, F> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.re += other.re;
        self.eps += other.eps;
    }
}

impl<T: DualNum<F>, F: Float> AddAssign<F> for Dual<T, F> {
    #[inline]
    fn add_assign(&mut self, other: F) {
        self.re += other;
    }
}

impl<T: DualNum<F>, F: Float> SubAssign for Dual<T, F> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.re -= other.re;
        self.eps -= other.eps;
    }
}

impl<T: DualNum<F>, F: Float> SubAssign<F> for Dual<T, F> {
    #[inline]
    fn sub_assign(&mut self, other: F) {
        self.re -= other;
    }
}

impl<T: DualNum<F>, F: Float> RemAssign for Dual<T, F> {
    #[inline]
    fn rem_assign(&mut self, _other: Self) {
        unimplemented!()
    }
}

impl<T: DualNum<F>, F: Float> RemAssign<F> for Dual<T, F> {
    #[inline]
    fn rem_assign(&mut self, _other: F) {
        unimplemented!()
    }
}

/* constants */
impl<T: DualNum<F>, F: Float> Zero for Dual<T, F> {
    #[inline]
    fn zero() -> Self {
        Dual::new(Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps.is_zero()
    }
}

impl<T: DualNum<F>, F: Float> One for Dual<T, F> {
    #[inline]
    fn one() -> Self {
        Dual::new(One::one(), Zero::zero())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.eps.is_zero()
    }
}

/* string conversions */
impl<T: fmt::Display, F> fmt::Display for Dual<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

/* iterator methods */
impl<T: DualNum<F>, F: Float> Sum for Dual<T, F> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: DualNum<F>, F: Float> Sum<&'a Dual<T, F>> for Dual<T, F> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Dual<T, F>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: DualNum<F>, F: Float> Product for Dual<T, F> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: DualNum<F>, F: Float> Product<&'a Dual<T, F>> for Dual<T, F> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Dual<T, F>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<T: DualNum<F>, F: Float> Num for Dual<T, F> {
    type FromStrRadixErr = F::FromStrRadixErr;
    fn from_str_radix(_str: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!()
    }
}

impl_from_primitive!(Dual);
impl_signed!(Dual);
impl_float_const!(Dual);
