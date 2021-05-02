//! Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.

use num_traits::{FromPrimitive, Inv, NumAssignOps, NumOps, Signed};
use std::fmt;
use std::iter::{Product, Sum};

#[macro_use]
mod macros;
#[macro_use]
mod derivatives;

mod dual;
mod dual_n;
mod hd2;
mod hd3;
mod hyperdual;
mod hyperdual_n;
mod linalg;
mod static_mat;
pub use dual::{Dual, Dual32, Dual64};
pub use dual_n::{DualN, DualN32, DualN64};
pub use hd2::{HD2Dual32, HD2Dual64, HD2DualN32, HD2DualN64, HD2, HD2_32, HD2_64};
pub use hd3::{HD3Dual32, HD3Dual64, HD3DualN32, HD3DualN64, HD3, HD3_32, HD3_64};
pub use hyperdual::{
    HyperDual, HyperDual32, HyperDual64, HyperDualDual32, HyperDualDual64, HyperDualDualN32,
    HyperDualDualN64,
};
pub use hyperdual_n::{
    HyperDualN, HyperDualN32, HyperDualN64, HyperDualNDual32, HyperDualNDual64, HyperDualNDualN32,
    HyperDualNDualN64,
};
pub use linalg::Scale;
pub use static_mat::{StaticMat, StaticVec};

#[cfg(feature = "linalg")]
pub use linalg::*;

pub trait DualNum<F>:
    DualNumMethods<F>
    + Signed
    + NumOps<F>
    + NumAssignOps
    + NumAssignOps<F>
    + Scale<F>
    + Copy
    + Inv<Output = Self>
    + Sum
    + Product
    + FromPrimitive
    + From<F>
    + fmt::Display
    + Sync
    + Send
    + 'static
{
}
impl<D, F> DualNum<F> for D where
    D: DualNumMethods<F>
        + Signed
        + NumOps<F>
        + NumAssignOps
        + NumAssignOps<F>
        + Scale<F>
        + Copy
        + Inv<Output = Self>
        + Sum
        + Product
        + FromPrimitive
        + From<F>
        + fmt::Display
        + Sync
        + Send
        + 'static
{
}

pub trait DualNumMethods<F>: Clone + NumOps {
    /// indicates the highest derivative that can be calculated with this struct
    const NDERIV: usize;

    /// returns the real part (the 0th derivative) of the number
    fn re(&self) -> F;

    fn recip(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, n: F) -> Self;
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn exp_m1(&self) -> Self;
    fn ln(&self) -> Self;
    fn log(&self, base: F) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
    fn ln_1p(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn sph_j0(&self) -> Self;
    fn sph_j1(&self) -> Self;
    fn sph_j2(&self) -> Self;

    #[inline]
    fn mul_add(&self, a: Self, b: Self) -> Self {
        self.clone() * a + b
    }

    #[inline]
    fn powd(&self, exp: &Self) -> Self {
        (self.ln() * exp.clone()).exp()
    }
}

macro_rules! impl_dual_num_float {
    ($float:ty) => {
        impl DualNumMethods<$float> for $float {
            const NDERIV: usize = 0;

            fn re(&self) -> $float {
                *self
            }

            fn mul_add(&self, a: Self, b: Self) -> Self {
                <$float>::mul_add(*self, a, b)
            }
            fn recip(&self) -> Self {
                <$float>::recip(*self)
            }
            fn powi(&self, n: i32) -> Self {
                <$float>::powi(*self, n)
            }
            fn powf(&self, n: Self) -> Self {
                <$float>::powf(*self, n)
            }
            fn powd(&self, n: &Self) -> Self {
                <$float>::powf(*self, *n)
            }
            fn sqrt(&self) -> Self {
                <$float>::sqrt(*self)
            }
            fn exp(&self) -> Self {
                <$float>::exp(*self)
            }
            fn exp2(&self) -> Self {
                <$float>::exp2(*self)
            }
            fn ln(&self) -> Self {
                <$float>::ln(*self)
            }
            fn log(&self, base: Self) -> Self {
                <$float>::log(*self, base)
            }
            fn log2(&self) -> Self {
                <$float>::log2(*self)
            }
            fn log10(&self) -> Self {
                <$float>::log10(*self)
            }
            fn cbrt(&self) -> Self {
                <$float>::cbrt(*self)
            }
            fn sin(&self) -> Self {
                <$float>::sin(*self)
            }
            fn cos(&self) -> Self {
                <$float>::cos(*self)
            }
            fn tan(&self) -> Self {
                <$float>::tan(*self)
            }
            fn asin(&self) -> Self {
                <$float>::asin(*self)
            }
            fn acos(&self) -> Self {
                <$float>::acos(*self)
            }
            fn atan(&self) -> Self {
                <$float>::atan(*self)
            }
            fn sin_cos(&self) -> (Self, Self) {
                <$float>::sin_cos(*self)
            }
            fn exp_m1(&self) -> Self {
                <$float>::exp_m1(*self)
            }
            fn ln_1p(&self) -> Self {
                <$float>::ln_1p(*self)
            }
            fn sinh(&self) -> Self {
                <$float>::sinh(*self)
            }
            fn cosh(&self) -> Self {
                <$float>::cosh(*self)
            }
            fn tanh(&self) -> Self {
                <$float>::tanh(*self)
            }
            fn asinh(&self) -> Self {
                <$float>::asinh(*self)
            }
            fn acosh(&self) -> Self {
                <$float>::acosh(*self)
            }
            fn atanh(&self) -> Self {
                <$float>::atanh(*self)
            }
            fn sph_j0(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    1.0 - self * self / 6.0
                } else {
                    self.sin() / self
                }
            }
            fn sph_j1(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    self / 3.0
                } else {
                    let sc = self.sin_cos();
                    let rec = self.recip();
                    (sc.0 * rec - sc.1) * rec
                }
            }
            fn sph_j2(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    self * self / 15.0
                } else {
                    let sc = self.sin_cos();
                    let s2 = self * self;
                    ((3.0 - s2) * sc.0 - 3.0 * self * sc.1) / (self * s2)
                }
            }
        }

        impl Scale<$float> for $float {
            fn scale(&mut self, f: $float) {
                *self *= f;
            }
        }
    };
}

impl_dual_num_float!(f32);
impl_dual_num_float!(f64);
