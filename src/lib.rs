//! Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.
//!
//! ## Example
//! This example defines a generic scalar and a generic vector function that can be called using any (hyper-) dual number and automatically calculates derivatives.
//! ```
//! use num_dual::*;
//! use nalgebra::SVector;
//!
//! fn foo<D: DualNum<f64>>(x: D) -> D {
//!     x.powi(3)
//! }
//!
//! fn bar<D: DualNum<f64>, const N: usize>(x: SVector<D, N>) -> D {
//!     x.dot(&x).sqrt()
//! }
//!
//! fn main() {
//!     // Calculate a simple derivative
//!     let (f, df) = first_derivative(foo, 5.0);
//!     assert_eq!(f, 125.0);
//!     assert_eq!(df, 75.0);
//!
//!     // Manually construct the dual number
//!     let x = Dual64::new_scalar(5.0, 1.0);
//!     println!("{}", foo(x));                     // 125 + [75]ε
//!
//!     // Calculate a gradient
//!     let (f, g) = gradient(bar, SVector::from([4.0, 3.0]));
//!     assert_eq!(f, 5.0);
//!     assert_eq!(g[0], 0.8);
//!
//!     // Calculate a Hessian
//!     let (f, g, h) = hessian(bar, SVector::from([4.0, 3.0]));
//!     println!("{h}");                            // [[0.072, -0.096], [-0.096, 0.128]]
//!
//!     // for x=cos(t) calculate the third derivative of foo w.r.t. t
//!     let (f0, f1, f2, f3) = third_derivative(|t| foo(t.cos()), 1.0);
//!     println!("{f3}");                           // 1.5836632930100278
//! }
//! ```

#![warn(clippy::all)]
#![allow(clippy::needless_range_loop)]

use num_traits::{Float, FloatConst, FromPrimitive, Inv, NumAssignOps, NumOps, Signed};
use std::fmt;
use std::iter::{Product, Sum};

#[macro_use]
mod macros;
#[macro_use]
mod derivatives;

mod bessel;
mod derivative;
mod dual;
mod dual2;
mod dual3;
mod hyperdual;
mod hyperhyperdual;
pub use bessel::BesselDual;
pub use derivative::Derivative;
pub use dual::{
    first_derivative, gradient, jacobian, try_first_derivative, try_gradient, try_jacobian, Dual,
    Dual32, Dual64, DualDVec32, DualDVec64, DualSVec32, DualSVec64, DualVec, DualVec32, DualVec64,
};
pub use dual2::{
    hessian, second_derivative, try_hessian, try_second_derivative, Dual2, Dual2DVec32,
    Dual2DVec64, Dual2SVec32, Dual2SVec64, Dual2Vec, Dual2Vec32, Dual2Vec64, Dual2_32, Dual2_64,
};
pub use dual3::{third_derivative, try_third_derivative, Dual3, Dual3_32, Dual3_64};
pub use hyperdual::{
    partial_hessian, second_partial_derivative, try_partial_hessian, try_second_partial_derivative,
    HyperDual, HyperDual32, HyperDual64, HyperDualDVec32, HyperDualDVec64, HyperDualSVec32,
    HyperDualSVec64, HyperDualVec, HyperDualVec32, HyperDualVec64,
};
pub use hyperhyperdual::{
    third_partial_derivative, third_partial_derivative_vec, try_third_partial_derivative,
    try_third_partial_derivative_vec, HyperHyperDual, HyperHyperDual32, HyperHyperDual64,
};

#[cfg(feature = "linalg")]
pub mod linalg;

#[cfg(feature = "python")]
pub mod python;

/// A generalized (hyper) dual number.
pub trait DualNum<F>:
    NumOps
    + for<'r> NumOps<&'r Self>
    + Signed
    + NumOps<F>
    + NumAssignOps
    + NumAssignOps<F>
    + Clone
    + Inv<Output = Self>
    + Sum
    + Product
    + FromPrimitive
    + From<F>
    + fmt::Display
    + PartialEq
    + fmt::Debug
    + 'static
{
    /// Highest derivative that can be calculated with this struct
    const NDERIV: usize;

    /// Real part (0th derivative) of the number
    fn re(&self) -> F;

    /// Reciprocal (inverse) of a number `1/x`.
    fn recip(&self) -> Self;

    /// Power with integer exponent `x^n`
    fn powi(&self, n: i32) -> Self;

    /// Power with real exponent `x^n`
    fn powf(&self, n: F) -> Self;

    /// Square root
    fn sqrt(&self) -> Self;

    /// Cubic root
    fn cbrt(&self) -> Self;

    /// Exponential `e^x`
    fn exp(&self) -> Self;

    /// Exponential with base 2 `2^x`
    fn exp2(&self) -> Self;

    /// Exponential minus 1 `e^x-1`
    fn exp_m1(&self) -> Self;

    /// Natural logarithm
    fn ln(&self) -> Self;

    /// Logarithm with arbitrary base
    fn log(&self, base: F) -> Self;

    /// Logarithm with base 2
    fn log2(&self) -> Self;

    /// Logarithm with base 10
    fn log10(&self) -> Self;

    /// Logarithm on x plus one `ln(1+x)`
    fn ln_1p(&self) -> Self;

    /// Sine
    fn sin(&self) -> Self;

    /// Cosine
    fn cos(&self) -> Self;

    /// Tangent
    fn tan(&self) -> Self;

    /// Calculate sine and cosine simultaneously
    fn sin_cos(&self) -> (Self, Self);

    /// Arcsine
    fn asin(&self) -> Self;

    /// Arccosine
    fn acos(&self) -> Self;

    /// Arctangent
    fn atan(&self) -> Self;

    /// Hyperbolic sine
    fn sinh(&self) -> Self;

    /// Hyperbolic cosine
    fn cosh(&self) -> Self;

    /// Hyperbolic tangent
    fn tanh(&self) -> Self;

    /// Area hyperbolic sine
    fn asinh(&self) -> Self;

    /// Area hyperbolic cosine
    fn acosh(&self) -> Self;

    /// Area hyperbolic tangent
    fn atanh(&self) -> Self;

    /// 0th order spherical Bessel function of the first kind
    fn sph_j0(&self) -> Self;

    /// 1st order spherical Bessel function of the first kind
    fn sph_j1(&self) -> Self;

    /// 2nd order spherical Bessel function of the first kind
    fn sph_j2(&self) -> Self;

    /// Fused multiply-add
    #[inline]
    fn mul_add(&self, a: Self, b: Self) -> Self {
        self.clone() * a + b
    }

    /// Power with dual exponent `x^n`
    #[inline]
    fn powd(&self, exp: Self) -> Self {
        (self.ln() * exp).exp()
    }
}

/// The underlying data type of individual derivatives. Usually f32 or f64.
pub trait DualNumFloat:
    Float + FloatConst + FromPrimitive + Signed + fmt::Display + fmt::Debug + Sync + Send + 'static
{
}
impl<T> DualNumFloat for T where
    T: Float + FloatConst + FromPrimitive + Signed + fmt::Display + fmt::Debug + Sync + Send + 'static
{
}

macro_rules! impl_dual_num_float {
    ($float:ty) => {
        impl DualNum<$float> for $float {
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
            fn powd(&self, n: Self) -> Self {
                <$float>::powf(*self, n)
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
    };
}

impl_dual_num_float!(f32);
impl_dual_num_float!(f64);
