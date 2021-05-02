macro_rules! impl_derivatives {
    ($deriv:ident, $nderiv:expr, $struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl<T: DualNum<F>, F: Float, $(const $const: usize,)*> DualNumMethods<F> for $struct<T, F$(, $const)*> {
            const NDERIV: usize = T::NDERIV + $nderiv;

            #[inline]
            fn re(&self) -> F {
                self.re.re()
            }

            #[inline]
            fn scale(&mut self, f: F) {
                self.re.scale(f);
                $(self.$im.scale(f);)*
            }

            #[inline]
            fn recip(&self) -> Self {
                let rec = self.re.recip();
                let f0 = rec;
                let f1 = -f0 * rec;
                second!($deriv, let f2 = f1 * rec * F::from(-2.0).unwrap(););
                third!($deriv, let f3 = f2 * rec * F::from(-3.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn powi(&self, exp: i32) -> Self {
                match exp {
                    0 => Self::one(),
                    1 => *self,
                    2 => self * self,
                    _ => {
                        let pow3 = self.re.powi(exp - 3);
                        let f0 = pow3 * self.re * self.re * self.re;
                        let f1 = pow3 * self.re * self.re * F::from(exp).unwrap();
                        second!($deriv, let f2 = pow3 * self.re * F::from(exp * (exp - 1)).unwrap(););
                        third!($deriv, let f3 = pow3 * F::from(exp * (exp - 1) * (exp - 2)).unwrap(););
                        chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
                    }
                }
            }

            #[inline]
            fn powf(&self, n: F) -> Self {
                if n.is_zero() {
                    Self::one()
                } else if n.is_one() {
                    *self
                } else if (n - F::one() - F::one()).abs() < F::epsilon() {
                    self * self
                } else {
                    let n1 = n - F::one();
                    let n2 = n1 - F::one();
                    let n3 = n2 - F::one();
                    let pow3 = self.re.powf(n3);
                    let f0 = pow3 * self.re * self.re * self.re;
                    let f1 = pow3 * self.re * self.re * n;
                    second!($deriv, let f2 = pow3 * self.re * n * n1;);
                    third!($deriv, let f3 = pow3 * n * n1 * n2;);
                    chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
                }
            }

            #[inline]
            fn sqrt(&self) -> Self {
                let rec = self.re.recip();
                let half = F::from(0.5).unwrap();
                let f0 = self.re.sqrt();
                let f1 = f0 * rec * half;
                second!($deriv, let f2 = -f1 * rec * half;);
                third!($deriv, let f3 = f2 * rec * (-F::one() - half););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn cbrt(&self) -> Self {
                let rec = self.re.recip();
                let third = F::from(1.0 / 3.0).unwrap();
                let f0 = self.re.cbrt();
                let f1 = f0 * rec * third;
                second!($deriv, let f2 = f1 * rec * (third - F::one()););
                third!($deriv, let f3 = f2 * rec * (third - F::one() - F::one()););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }


            #[inline]
            fn exp(&self) -> Self {
                let f = self.re.exp();
                chain_rule!($deriv, Self::chain_rule(self, f, f, f, f))
            }

            #[inline]
            fn exp2(&self) -> Self {
                let ln2 = F::from(2.0).unwrap().ln();
                let f0 = self.re.exp2();
                let f1 = f0 * ln2;
                second!($deriv, let f2 = f1 * ln2;);
                third!($deriv, let f3 = f2 * ln2;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn exp_m1(&self) -> Self {
                let f0 = self.re.exp_m1();
                let f1 = self.re.exp();
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f1, f1))
            }

            #[inline]
            fn ln(&self) -> Self {
                let rec = self.re.recip();
                let f0 = self.re.ln();
                let f1 = rec;
                second!($deriv, let f2 = -f1 * rec;);
                third!($deriv, let f3 = f2 * rec * F::from(-2.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn log(&self, base: F) -> Self {
                let rec = self.re.recip();
                let f0 = self.re.log(base);
                let f1 = rec / base.ln();
                second!($deriv, let f2 = -f1 * rec;);
                third!($deriv, let f3 = f2 * rec * F::from(-2.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn log2(&self) -> Self {
                let rec = self.re.recip();
                let f0 = self.re.log2();
                let f1 = rec / (F::one() + F::one()).ln();
                second!($deriv, let f2 = -f1 * rec;);
                third!($deriv, let f3 = f2 * rec * F::from(-2.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn log10(&self) -> Self {
                let rec = self.re.recip();
                let f0 = self.re.log10();
                let f1 = rec / F::from(10.0).unwrap().ln();
                second!($deriv, let f2 = -f1 * rec;);
                third!($deriv, let f3 = f2 * rec * F::from(-2.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn ln_1p(&self) -> Self {
                let rec = (self.re + F::one()).recip();
                let f0 = self.re.ln_1p();
                let f1 = rec;
                second!($deriv, let f2 = -f1 * rec;);
                third!($deriv, let f3 = f2 * rec * F::from(-2.0).unwrap(););
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn sin(&self) -> Self {
                let (s, c) = self.re.sin_cos();
                chain_rule!($deriv, Self::chain_rule(self, s, c, -s, -c))
            }

            #[inline]
            fn cos(&self) -> Self {
                let (s, c) = self.re.sin_cos();
                chain_rule!($deriv, Self::chain_rule(self, c, -s, -c, s))
            }

            #[inline]
            fn sin_cos(&self) -> (Self, Self) {
                let (s, c) = self.re.sin_cos();
                (chain_rule!($deriv, Self::chain_rule(self, s, c, -s, -c)), chain_rule!($deriv, Self::chain_rule(self, c, -s, -c, s)))
            }

            #[inline]
            fn tan(&self) -> Self {
                let (sin, cos) = self.sin_cos();
                sin / cos
            }

            #[inline]
            fn asin(&self) -> Self {
                let rec = (T::one() - self.re * self.re).recip();
                let f0 = self.re.asin();
                let f1 = rec.sqrt();
                second!($deriv, let f2 = self.re * f1 * rec;);
                third!($deriv, let f3 = (self.re * self.re * (F::one() + F::one()) + F::one()) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn acos(&self) -> Self {
                let rec = (T::one() - self.re * self.re).recip();
                let f0 = self.re.acos();
                let f1 = -rec.sqrt();
                second!($deriv, let f2 = self.re * f1 * rec;);
                third!($deriv, let f3 = (self.re * self.re * (F::one() + F::one()) + F::one()) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn atan(&self) -> Self {
                let rec = (T::one() + self.re * self.re).recip();
                let f0 = self.re.atan();
                let f1 = rec;
                second!($deriv, let two = F::one() + F::one(););
                second!($deriv, let f2 = -self.re * f1 * rec * two;);
                third!($deriv, let f3 = (self.re * self.re * F::from(6.0).unwrap() - two) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn sinh(&self) -> Self {
                let s = self.re.sinh();
                let c = self.re.cosh();
                chain_rule!($deriv, Self::chain_rule(self, s, c, s, c))
            }

            #[inline]
            fn cosh(&self) -> Self {
                let s = self.re.sinh();
                let c = self.re.cosh();
                chain_rule!($deriv, Self::chain_rule(self, c, s, c, s))
            }

            #[inline]
            fn tanh(&self) -> Self {
                self.sinh() / self.cosh()
            }

            #[inline]
            fn asinh(&self) -> Self {
                let rec = (T::one() + self.re * self.re).recip();
                let f0 = self.re.asinh();
                let f1 = rec.sqrt();
                second!($deriv, let f2 = -self.re * f1 * rec;);
                third!($deriv, let f3 = (self.re * self.re * (F::one() + F::one()) - F::one()) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn acosh(&self) -> Self {
                let rec = (self.re * self.re - F::one()).recip();
                let f0 = self.re.acosh();
                let f1 = rec.sqrt();
                second!($deriv, let f2 = -self.re * f1 * rec;);
                third!($deriv, let f3 = (self.re * self.re * (F::one() + F::one()) + F::one()) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn atanh(&self) -> Self {
                let rec = (T::one() - self.re * self.re).recip();
                let f0 = self.re.atanh();
                let f1 = rec;
                second!($deriv, let two = F::one() + F::one(););
                second!($deriv, let f2 = self.re * f1 * rec * two;);
                third!($deriv, let f3 = (self.re * self.re * F::from(6.0).unwrap() + two) * f1 * rec * rec;);
                chain_rule!($deriv, Self::chain_rule(self, f0, f1, f2, f3))
            }

            #[inline]
            fn sph_j0(&self) -> Self {
                if self.re() < F::epsilon() {
                    Self::one() - self * self / F::from(6.0).unwrap()
                } else {
                    self.sin() / self
                }
            }

            #[inline]
            fn sph_j1(&self) -> Self {
                if self.re() < F::epsilon() {
                    *self / F::from(3.0).unwrap()
                } else {
                    let (s, c) = self.sin_cos();
                    (s - self * c) / (self * self)
                }
            }

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
    };
}

macro_rules! second {
    (first, $($code:tt)*) => {};
    (second, $($code:tt)*) => {
        $($code)*
    };
    (third, $($code:tt)*) => {
        $($code)*
    };
}

macro_rules! third {
    (first, $($code:tt)*) => {};
    (second, $($code:tt)*) => {};
    (third, $($code:tt)*) => {
        $($code)*
    };
}

macro_rules! chain_rule {
    (first, Self::chain_rule($self:ident, $f0:expr, $f1:expr, $f2:expr, $f3:expr)) => {
        Self::chain_rule($self, $f0, $f1)
    };
    (second, Self::chain_rule($self:ident, $f0:expr, $f1:expr, $f2:expr, $f3:expr)) => {
        Self::chain_rule($self, $f0, $f1, $f2)
    };
    (third, Self::chain_rule($self:ident, $f0:expr, $f1:expr, $f2:expr, $f3:expr)) => {
        Self::chain_rule($self, $f0, $f1, $f2, $f3)
    };
}

macro_rules! impl_first_derivatives {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl_derivatives!(first, 1, $struct, [$($const),*], [$($im),*]);
    };
}

macro_rules! impl_second_derivatives {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl_derivatives!(second, 2, $struct, [$($const),*], [$($im),*]);
    };
}

macro_rules! impl_third_derivatives {
    ($struct:ident, [$($const:tt),*], [$($im:ident),*]) => {
        impl_derivatives!(third, 3, $struct, [$($const),*], [$($im),*]);
    };
}
