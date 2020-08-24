use crate::DualNum;
use num_traits::{Float, Inv, Num, One, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dual number.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug, Default)]
#[repr(C)]
pub struct Dual<T> {
    /// Real part of the dual number
    pub re: T,
    /// Eps part
    pub eps: T,
}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

impl<T: Clone + Num> Dual<T> {
    /// Create a new Dual
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Dual { re, eps }
    }
}

impl<T: Float> DualNum<T> for Dual<T> {
    fn re(&self) -> T {
        self.re
    }

    /// Returns `1/self`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).recip();
    /// assert!((res.re - 0.833333333333333).abs() < 1e-10);
    /// assert!((res.eps - -0.694444444444445).abs() < 1e-10);
    /// ```
    #[inline]
    fn recip(&self) -> Self {
        if self.re == T::zero() {
            panic!("Cannot take reciprocal value of zero-valued `real`!");
        }
        let recip_re = self.re.recip();
        Dual::new(recip_re, -self.eps * recip_re * recip_re)
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).log(4.2);
    /// assert!((res.re - 0.127045866345188).abs() < 1e-10);
    /// assert!((res.eps - 0.580685888982970).abs() < 1e-10);
    /// ```
    #[inline]
    fn log(&self, base: T) -> Self {
        let fx = self.re.log(base);
        let dx = (self.re * base.ln()).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes `ln(1+n)` more accurately than if the operations were performed separately.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).log10();
    /// assert!((res.re - 0.0791812460476248).abs() < 1e-10);
    /// assert!((res.eps - 0.361912068252710).abs() < 1e-10);
    /// ```
    #[inline]
    fn log10(&self) -> Self {
        let fx = self.re.log10();
        let dx = (T::from(10).unwrap().ln() * self.re).recip();
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the principal value of the square root of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).powf(4.2);
    /// assert!((res.re - 2.15060788316847).abs() < 1e-10);
    /// assert!((res.eps - 7.52712759108966).abs() < 1e-10);
    /// ```
    #[inline]
    fn powf(&self, exp: T) -> Self {
        let fx = self.re.powf(exp);
        let dx = exp * fx / self.re;
        Dual::new(fx, self.eps * dx)
    }

    /// Raises `self` to an integer power.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).powi(4);
    /// assert!((res.re - 2.07360000000000).abs() < 1e-10);
    /// assert!((res.eps - 6.91200000000000).abs() < 1e-10);
    /// ```
    #[inline]
    fn powi(&self, exp: i32) -> Self {
        let e = T::from(exp).unwrap();

        let pow = self.re.powi(exp - 1);
        let fx = pow * self.re;
        let dx = e * pow;
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the sine of `self`.
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
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
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).sph_j0();
    /// assert!((res.re - 0.776699238306022).abs() < 1e-10);
    /// assert!((res.eps - -0.345284569857790).abs() < 1e-10);
    /// ```
    fn sph_j0(&self) -> Self {
        let (fx, dx) = if self.re.abs() < T::epsilon() {
            (
                T::one() - self.re * self.re / T::from(6.0).unwrap(),
                -self.re / T::from(3.0).unwrap(),
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
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).sph_j1();
    /// assert!((res.re - 0.345284569857790).abs() < 1e-10);
    /// assert!((res.eps - 0.201224955209705).abs() < 1e-10);
    /// ```
    fn sph_j1(&self) -> Self {
        let (fx, dx) = if self.re.abs() < T::epsilon() {
            let one = T::one();
            let third = one / (one + one + one);
            (self.re * third, third)
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let two = T::one() + T::one();
            ((s * rec - c) * rec, (two * (c - s * rec) * rec + s) * rec)
        };
        Dual::new(fx, self.eps * dx)
    }

    /// Computes the second order spherical bessel function `j2(x)`
    /// ```
    /// # use num_hyperdual::dual::Dual64;
    /// # use num_hyperdual::DualNum;
    /// let res = Dual64::new(1.2, 1.0).sph_j2();
    /// assert!((res.re - 0.0865121863384538).abs() < 1e-10);
    /// assert!((res.eps - 0.129004104011656).abs() < 1e-10);
    /// ```
    fn sph_j2(&self) -> Self {
        let (fx, dx) = if self.re.abs() < T::epsilon() {
            (
                self.re * self.re / T::from(15.0).unwrap(),
                self.re / T::from(7.5).unwrap(),
            )
        } else {
            let (s, c) = self.re.sin_cos();
            let rec = self.re.recip();
            let three = T::one() + T::one() + T::one();
            let nine = three * three;
            (
                (three * (s * rec - c) * rec - s) * rec,
                ((nine * (-s * rec + c) * rec + (three + T::one()) * s) * rec - c) * rec,
            )
        };
        Dual::new(fx, self.eps * dx)
    }
}

impl<T: Float> Inv for Dual<T> {
    type Output = Self;
    fn inv(self) -> Self {
        self.recip()
    }
}

impl<T: Float> Inv for &Dual<T> {
    type Output = Dual<T>;
    fn inv(self) -> Dual<T> {
        self.recip()
    }
}

impl<T: Zero> From<T> for Dual<T> {
    #[inline]
    fn from(re: T) -> Self {
        Dual {
            re: re,
            eps: T::zero(),
        }
    }
}

impl<'a, T: Clone + Zero> From<&'a T> for Dual<T> {
    #[inline]
    fn from(re: &T) -> Self {
        From::from(re.clone())
    }
}

macro_rules! forward_ref_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, 'b, T: Clone + Num> $imp<&'b Dual<T>> for &'a Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: &Dual<T>) -> Dual<T> {
                self.clone().$method(other.clone())
            }
        }
    };
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T: Clone + Num> $imp<Dual<T>> for &'a Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: Dual<T>) -> Dual<T> {
                self.clone().$method(other)
            }
        }
    };
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, T: Clone + Num> $imp<&'a Dual<T>> for Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: &Dual<T>) -> Self {
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

impl<T: Clone + Num> Add<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn add(self, other: Dual<T>) -> Self {
        Dual::new(
            self.re.clone() + other.re.clone(),
            self.eps.clone() + other.eps.clone(),
        )
    }
}

forward_all_binop!(impl Sub, sub);

impl<T: Clone + Num> Sub<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn sub(self, other: Dual<T>) -> Self {
        Dual::new(
            self.re.clone() - other.re.clone(),
            self.eps.clone() - other.eps.clone(),
        )
    }
}

forward_all_binop!(impl Mul, mul);

impl<T: Clone + Num> Mul<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn mul(self, other: Dual<T>) -> Self {
        Dual::new(
            self.re.clone() * other.re.clone(),
            self.re.clone() * other.eps.clone() + other.re.clone() * self.eps.clone(),
        )
    }
}

forward_all_binop!(impl Div, div);

impl<T: Clone + Num> Div<Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn div(self, other: Dual<T>) -> Self {
        let inv = T::one() / other.re.clone();
        let inv2 = inv.clone() * inv.clone();
        Dual::new(
            self.re.clone() * inv.clone(),
            (self.eps.clone() * other.re.clone() - self.re.clone() * other.eps.clone())
                * inv2.clone(),
        )
    }
}

/* Neg impl */
impl<T: Clone + Num + Neg<Output = T>> Neg for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn neg(self) -> Self {
        Dual::new(-self.re, -self.eps)
    }
}

impl<'a, T: Clone + Num + Neg<Output = T>> Neg for &'a Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn neg(self) -> Dual<T> {
        -self.clone()
    }
}

macro_rules! real_arithmetic {
    (@forward $imp:ident::$method:ident for $($real:ident),*) => (
        impl<'a, T: Clone + Num> $imp<&'a T> for Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: &T) -> Self {
                self.$method(other.clone())
            }
        }
        impl<'a, T: Clone + Num> $imp<T> for &'a Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: T) -> Dual<T> {
                self.clone().$method(other)
            }
        }
        impl<'a, 'b, T: Clone + Num> $imp<&'a T> for &'b Dual<T> {
            type Output = Dual<T>;

            #[inline]
            fn $method(self, other: &T) -> Dual<T> {
                self.clone().$method(other.clone())
            }
        }
        $(
            impl<'a> $imp<&'a Dual<$real>> for $real {
                type Output = Dual<$real>;

                #[inline]
                fn $method(self, other: &Dual<$real>) -> Dual<$real> {
                    self.$method(other.clone())
                }
            }
            impl<'a> $imp<Dual<$real>> for &'a $real {
                type Output = Dual<$real>;

                #[inline]
                fn $method(self, other: Dual<$real>) -> Dual<$real> {
                    self.clone().$method(other)
                }
            }
            impl<'a, 'b> $imp<&'a Dual<$real>> for &'b $real {
                type Output = Dual<$real>;

                #[inline]
                fn $method(self, other: &Dual<$real>) -> Dual<$real> {
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
            impl Add<Dual<$real>> for $real {
                type Output = Dual<$real>;

                #[inline]
                fn add(self, other: Dual<$real>) -> Dual<$real> {
                    Dual::new(self + other.re, other.eps)
                }
            }

            impl Sub<Dual<$real>> for $real {
                type Output = Dual<$real>;

                #[inline]
                fn sub(self, other: Dual<$real>) -> Dual<$real> {
                    Dual::new(self - other.re, Self::zero() - other.eps)
                }
            }

            impl Mul<Dual<$real>> for $real {
                type Output = Dual<$real>;

                #[inline]
                fn mul(self, other: Dual<$real>) -> Dual<$real> {
                    Dual::new(self * other.re, self * other.eps)
                }
            }

            impl Div<Dual<$real>> for $real {
                type Output = Dual<$real>;

                #[inline]
                fn div(self, other: Dual<$real>) -> Dual<$real> {
                    self * other.recip()
                }
            }
        )*
    );
}

real_arithmetic!(f32, f64);

impl<T: Clone + Num> Add<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn add(self, other: T) -> Self {
        Dual::new(self.re + other, self.eps)
    }
}

impl<T: Clone + Num> Sub<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn sub(self, other: T) -> Self {
        Dual::new(self.re - other, self.eps)
    }
}

impl<T: Clone + Num> Mul<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn mul(self, other: T) -> Self {
        Dual::new(self.re * other.clone(), self.eps * other.clone())
    }
}

impl<T: Clone + Num> Div<T> for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn div(self, other: T) -> Self {
        let inv = T::one() / other;
        Dual::new(self.re * inv.clone(), self.eps * inv.clone())
    }
}

/* constants */
impl<T: Clone + Num> Zero for Dual<T> {
    #[inline]
    fn zero() -> Self {
        Dual::new(Zero::zero(), Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps.is_zero()
    }
}

impl<T: Clone + Num> One for Dual<T> {
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
impl<T> fmt::Display for Dual<T>
where
    T: fmt::Display + Num + PartialOrd + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

/* iterator methods */
impl<T: Num + Clone> Sum for Dual<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<'a, T: 'a + Num + Clone> Sum<&'a Dual<T>> for Dual<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Dual<T>>,
    {
        iter.fold(Self::zero(), |acc, c| acc + c)
    }
}

impl<T: Num + Clone> Product for Dual<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}

impl<'a, T: 'a + Num + Clone> Product<&'a Dual<T>> for Dual<T> {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Dual<T>>,
    {
        iter.fold(Self::one(), |acc, c| acc * c)
    }
}
