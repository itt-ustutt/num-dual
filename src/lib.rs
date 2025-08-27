//! Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.
//!
//! # Example
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
//!     let x = Dual64::new(5.0, 1.0);
//!     println!("{}", foo(x));                     // 125 + [75]ε
//!
//!     // Calculate a gradient
//!     let (f, g) = gradient(bar, &SVector::from([4.0, 3.0]));
//!     assert_eq!(f, 5.0);
//!     assert_eq!(g[0], 0.8);
//!
//!     // Calculate a Hessian
//!     let (f, g, h) = hessian(bar, &SVector::from([4.0, 3.0]));
//!     println!("{h}");                            // [[0.072, -0.096], [-0.096, 0.128]]
//!
//!     // for x=cos(t) calculate the third derivative of foo w.r.t. t
//!     let (f0, f1, f2, f3) = third_derivative(|t| foo(t.cos()), 1.0);
//!     println!("{f3}");                           // 1.5836632930100278
//! }
//! ```
//!
//! # Usage
//! There are two ways to use the data structures and functions provided in this crate:
//! 1. (recommended) Using the provided functions for explicit ([`first_derivative`], [`gradient`], ...) and
//!    implicit ([`implicit_derivative`], [`implicit_derivative_binary`], [`implicit_derivative_vec`]) functions.
//! 2. (for experienced users) Using the different dual number types ([`Dual`], [`HyperDual`], [`DualVec`], ...) directly.
//!
//! The following examples and explanations focus on the first way.
//!
//! # Derivatives of explicit functions
//! To be able to calculate the derivative of a function, it needs to be generic over the type of dual number used.
//! Most commonly this would look like this:
//! ```compile_fail
//! fn foo<D: DualNum<f64> + Copy>(x: X) -> O {...}
//! ```
//! Of course, the function could also use single precision ([`f32`]) or be generic over the precision (`F:` [`DualNumFloat`]).
//! For now, [`Copy`] is not a supertrait of [`DualNum`] to enable the calculation of derivatives with respect
//! to a dynamic number of variables. However, in practice, using the [`Copy`] trait bound leads to an
//! implementation that is more similar to one not using AD and there could be severe performance ramifications
//! when using dynamically allocated dual numbers.
//!
//! The type `X` above is `D` for univariate functions, [`&OVector`](nalgebra::OVector) for multivariate
//! functions, and `(D, D)` or `(&OVector, &OVector)` for partial derivatives. In the simplest case, the output
//! `O` is a scalar `D`. However, it is generalized using the [`Mappable`] trait to also include types like
//! [`Option<D>`] or [`Result<D, E>`], collections like [`Vec<D>`] or [`HashMap<K, D>`], or custom structs that
//! implement the [`Mappable`] trait. Therefore, it is, e.g., possible to calculate the derivative of a fallible
//! function:
//!
//! ```no_run
//! # use num_dual::{DualNum, first_derivative};
//! # type E = ();
//! fn foo<D: DualNum<f64> + Copy>(x: D) -> Result<D, E> { todo!() }
//!
//! fn main() -> Result<(), E> {
//!     let (val, deriv) = first_derivative(foo, 2.0)?;
//!     // ...
//!     Ok(())
//! }
//! ```
//! All dual number types can contain other dual numbers as inner types. Therefore, it is also possible to
//! use the different derivative functions inside of each other.
//!
//! ## extra arguments
//! The [`partial`] and [`partial2`] functions are used to pass additional arguments to the function, e.g.:
//! ```no_run
//! # use num_dual::{DualNum, first_derivative, partial};
//! fn foo<D: DualNum<f64> + Copy>(x: D, args: &(D, D)) -> D { todo!() }
//!
//! fn main() {
//!     let (val, deriv) = first_derivative(partial(foo, &(3.0, 4.0)), 5.0);
//! }
//! ```
//! All types that implement the [`DualStruct`] trait can be used as additional function arguments. The
//! only difference between using the [`partial`] and [`partial2`] functions compared to passing the extra
//! arguments via a closure, is that the type of the extra arguments is automatically adjusted to the correct
//! dual number type used for the automatic differentiation. Note that the following code would not compile:
//! ```compile_fail
//! # use num_dual::{DualNum, first_derivative};
//! # fn foo<D: DualNum<f64> + Copy>(x: D, args: &(D, D)) -> D { todo!() }
//! fn main() {
//!     let (val, deriv) = first_derivative(|x| foo(x, &(3.0, 4.0)), 5.0);
//! }
//! ```
//! The code created by [`partial`] essentially translates to:
//! ```no_run
//! # use num_dual::{DualNum, first_derivative, Dual, DualStruct};
//! # fn foo<D: DualNum<f64> + Copy>(x: D, args: &(D, D)) -> D { todo!() }
//! fn main() {
//!     let (val, deriv) = first_derivative(|x| foo(x, &(Dual::from_inner(&3.0), Dual::from_inner(&4.0))), 5.0);
//! }
//! ```
//!
//! ## the [`Gradients`] trait
//! The functions [`gradient`], [`hessian`], and [`partial_hessian`] are generic over the dimensionality of the
//! variable vector. However, to use the functions in a generic context requires not using the [`Copy`] trait
//! bound on the dual number type, because the dynamically sized dual numbers can by construction not implement
//! [`Copy`]. Also, due to frequent heap allocations, the performance of the automatic differentiation could
//! suffer significantly for dynamically sized dual numbers compared to statically sized dual numbers. The
//! [`Gradients`] trait is introduced to overcome these limitations.
//! ```
//! # use num_dual::{DualNum, Gradients};
//! # use nalgebra::{OVector, DefaultAllocator, allocator::Allocator, vector, dvector};
//! # use approx::assert_relative_eq;
//! fn foo<D: DualNum<f64> + Copy, N: Gradients>(x: OVector<D, N>, n: &D) -> D where DefaultAllocator: Allocator<N> {
//!     x.dot(&x).sqrt() - n
//! }
//!
//! fn main() {
//!     let x = vector![1.0, 5.0, 5.0, 7.0];
//!     let (f, grad) = Gradients::gradient(foo, &x, &10.0);
//!     assert_eq!(f, 0.0);
//!     assert_relative_eq!(grad, vector![0.1, 0.5, 0.5, 0.7]);
//!
//!     let x = dvector![1.0, 5.0, 5.0, 7.0];
//!     let (f, grad) = Gradients::gradient(foo, &x, &10.0);
//!     assert_eq!(f, 0.0);
//!     assert_relative_eq!(grad, dvector![0.1, 0.5, 0.5, 0.7]);
//! }
//! ```
//! For dynamically sized input arrays, the [`Gradients`] trait evaluates gradients or higher-order derivatives
//! by iteratively evaluating scalar derivatives. For functions that do not rely on the [`Copy`] trait bound,
//! only benchmarking can reveal Whether the increased performance through the avoidance of heap allocations
//! can overcome the overhead of repeated function evaluations, i.e., if [`Gradients`] outperforms directly
//! calling [`gradient`], [`hessian`], or [`partial_hessian`].
//!
//! # Derivatives of implicit functions
//! Implicit differentiation is used to determine the derivative `dy/dx` where the output `y` is only related
//! implicitly to the input `x` via the equation `f(x,y)=0`. Automatic implicit differentiation generalizes the
//! idea to determining the output `y` with full derivative information. Note that the first step in calculating
//! an implicit derivative is always determining the "real" part (i.e., neglecting all derivatives) of the equation
//! `f(x,y)=0`. The `num-dual` library is focused on automatic differentiation and not nonlinear equation
//! solving. Therefore, this first step needs to be done with your own custom solutions, or Rust crates for
//! nonlinear equation solving and optimization like, e.g., [argmin](https://argmin-rs.org/).
//!
//! The following example implements a square root for generic dual numbers using implicit differentiation. Of
//! course, the derivatives of the square root can also be determined explicitly using the chain rule, so the
//! example serves mostly as illustration. `x.re()` provides the "real" part of the dual number which is a [`f64`]
//! and therefore, we can use all the functionalities from the std library (including the square root).
//! ```
//! # use num_dual::{DualNum, implicit_derivative, first_derivative};
//! fn implicit_sqrt<D: DualNum<f64> + Copy>(x: D) -> D {
//!     implicit_derivative(|s, x| s * s - x, x.re().sqrt(), &x)
//! }
//!
//! fn main() {
//!     // sanity check, not actually calculating any derivative
//!     assert_eq!(implicit_sqrt(25.0), 5.0);
//!     
//!     let (sq, deriv) = first_derivative(implicit_sqrt, 25.0);
//!     assert_eq!(sq, 5.0);
//!     // The derivative of sqrt(x) is 1/(2*sqrt(x)) which should evaluate to 0.1
//!     assert_eq!(deriv, 0.1);
//! }
//! ```
//! The `implicit_sqrt` or any likewise defined function is generic over the dual type `D`
//! and can, therefore, be used anywhere as a part of an arbitrary complex computation. The functions [`implicit_derivative_binary`] and [`implicit_derivative_vec`] can be used for implicit functions
//! with more than one variable.
//!
//! For implicit functions that contain complex models any a large number of parameters, the [`ImplicitDerivative`]
//! interface might come in handy. The idea is to define the implicit function using the [`ImplicitFunction`] trait
//! and feeding it into the [`ImplicitDerivative`] struct, which internally stores the parameters as dual numbers
//! and their real parts. The [`ImplicitDerivative`] then provides methods for the evaluation of the real part
//! of the residual (which can be passed to a nonlinear solver) and the implicit derivative which can be called
//! after solving for the real part of the solution to reconstruct all the derivatives.
//! ```
//! # use num_dual::{ImplicitFunction, DualNum, Dual, ImplicitDerivative};
//! struct ImplicitSqrt;
//! impl ImplicitFunction<f64> for ImplicitSqrt {
//!     type Parameters<D> = D;
//!     type Variable<D> = D;
//!     fn residual<D: DualNum<f64> + Copy>(x: D, square: &D) -> D {
//!         *square - x * x
//!     }
//! }
//!
//! fn main() {
//!     let x = Dual::from_re(25.0).derivative();
//!     let func = ImplicitDerivative::new(ImplicitSqrt, x);
//!     assert_eq!(func.residual(5.0), 0.0);
//!     assert_eq!(x.sqrt(), func.implicit_derivative(5.0));
//! }
//! ```
//!
//! ## Combination with nonlinear solver libraries
//! As mentioned previously, this crate does not contain any algorithms for nonlinear optimization or root finding.
//! However, combining the capabilities of automatic differentiation with nonlinear solving can be very fruitful.
//! Most importantly, the calculation of Jacobians or Hessians can be completely automated, if the model can be
//! expressed within the functionalities of the [`DualNum`] trait. On top of that implicit derivatives can be of
//! interest, if derivatives of the result of the optimization itself are relevant (e.g., in a bilevel
//! optimization). The synergy is exploited in the [`ipopt-ad`](https://github.com/prehner/ipopt-ad) crate that
//! turns the NLP solver [IPOPT](https://github.com/coin-or/Ipopt) into a black-box optimization algorithm (i.e.,
//! it only requires a function that returns the values of the optimization variable and constraints), without
//! any repercussions regarding the robustness or speed of convergence of the solver.
//!
//! If you are developing nonlinear optimization algorithms in Rust, feel free to reach out to us. We are happy to
//! discuss how to enhance your algorithms with the automatic differentiation capabilities of this crate.

#![warn(clippy::all)]
#![warn(clippy::allow_attributes)]

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, Scalar};
#[cfg(feature = "ndarray")]
use ndarray::ScalarOperand;
use num_traits::{Float, FloatConst, FromPrimitive, Inv, NumAssignOps, NumOps, Signed};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::{Product, Sum};

#[macro_use]
mod macros;
#[macro_use]
mod impl_derivatives;

mod bessel;
mod datatypes;
mod explicit;
mod implicit;
pub use bessel::BesselDual;
pub use datatypes::derivative::Derivative;
pub use datatypes::dual::{Dual, Dual32, Dual64};
pub use datatypes::dual2::{Dual2, Dual2_32, Dual2_64};
pub use datatypes::dual2_vec::{
    Dual2DVec32, Dual2DVec64, Dual2SVec32, Dual2SVec64, Dual2Vec, Dual2Vec32, Dual2Vec64,
};
pub use datatypes::dual3::{Dual3, Dual3_32, Dual3_64};
pub use datatypes::dual_vec::{
    DualDVec32, DualDVec64, DualSVec, DualSVec32, DualSVec64, DualVec, DualVec32, DualVec64,
};
pub use datatypes::hyperdual::{HyperDual, HyperDual32, HyperDual64};
pub use datatypes::hyperdual_vec::{
    HyperDualDVec32, HyperDualDVec64, HyperDualSVec32, HyperDualSVec64, HyperDualVec,
    HyperDualVec32, HyperDualVec64,
};
pub use datatypes::hyperhyperdual::{HyperHyperDual, HyperHyperDual32, HyperHyperDual64};
pub use explicit::{
    first_derivative, gradient, hessian, jacobian, partial, partial2, partial_hessian,
    second_derivative, second_partial_derivative, third_derivative, third_partial_derivative,
    third_partial_derivative_vec, Gradients,
};
pub use implicit::{
    implicit_derivative, implicit_derivative_binary, implicit_derivative_vec, ImplicitDerivative,
    ImplicitFunction,
};

pub mod linalg;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python_macro")]
mod python_macro;

/// A generalized (hyper) dual number.
#[cfg(feature = "ndarray")]
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
    + DualStruct<Self, F, Real = F>
    + Mappable<Self>
    + fmt::Display
    + PartialEq
    + fmt::Debug
    + ScalarOperand
    + 'static
{
    /// Highest derivative that can be calculated with this struct
    const NDERIV: usize;

    /// Reciprocal (inverse) of a number `1/x`
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

    /// Arctangent
    fn atan2(&self, other: Self) -> Self;

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

/// A generalized (hyper) dual number.
#[cfg(not(feature = "ndarray"))]
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
    + DualStruct<Self, F, Real = F>
    + Mappable<Self>
    + fmt::Display
    + PartialEq
    + fmt::Debug
    + 'static
{
    /// Highest derivative that can be calculated with this struct
    const NDERIV: usize;

    /// Reciprocal (inverse) of a number `1/x`
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

    /// Arctangent
    fn atan2(&self, other: Self) -> Self;

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
    T: Float
        + FloatConst
        + FromPrimitive
        + Signed
        + fmt::Display
        + fmt::Debug
        + Sync
        + Send
        + 'static
{
}

macro_rules! impl_dual_num_float {
    ($float:ty) => {
        impl DualNum<$float> for $float {
            const NDERIV: usize = 0;

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
            fn atan2(&self, other: $float) -> Self {
                <$float>::atan2(*self, other)
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

/// A struct that contains dual numbers. Needed for arbitrary arguments in [ImplicitFunction].
///
/// The trait is implemented for all dual types themselves, and common data types (tuple, vec,
/// array, ...) and can be implemented for custom data types to achieve full flexibility.
pub trait DualStruct<D, F> {
    type Real;
    type Inner;
    fn re(&self) -> Self::Real;
    fn from_inner(inner: &Self::Inner) -> Self;
}

/// Trait for structs used as an output of functions for which derivatives are calculated.
///
/// The main intention is to generalize the calculation of derivatives to fallible functions, but
/// other use cases might also appear in the future.
pub trait Mappable<D> {
    type Output<O>;
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O>;
}

impl<D, F> DualStruct<D, F> for () {
    type Real = ();
    type Inner = ();
    fn re(&self) {}
    fn from_inner(_: &Self::Inner) -> Self {}
}

impl<D> Mappable<D> for () {
    type Output<O> = ();
    fn map_dual<M: FnOnce(D) -> O, O>(self, _: M) {}
}

impl DualStruct<f32, f32> for f32 {
    type Real = f32;
    type Inner = f32;
    fn re(&self) -> f32 {
        *self
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        *inner
    }
}

impl Mappable<f32> for f32 {
    type Output<O> = O;
    fn map_dual<M: FnOnce(f32) -> O, O>(self, f: M) -> Self::Output<O> {
        f(self)
    }
}

impl DualStruct<f64, f64> for f64 {
    type Real = f64;
    type Inner = f64;
    fn re(&self) -> f64 {
        *self
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        *inner
    }
}

impl Mappable<f64> for f64 {
    type Output<O> = O;
    fn map_dual<M: FnOnce(f64) -> O, O>(self, f: M) -> Self::Output<O> {
        f(self)
    }
}

impl<D, F, T1: DualStruct<D, F>, T2: DualStruct<D, F>> DualStruct<D, F> for (T1, T2) {
    type Real = (T1::Real, T2::Real);
    type Inner = (T1::Inner, T2::Inner);
    fn re(&self) -> Self::Real {
        let (s1, s2) = self;
        (s1.re(), s2.re())
    }
    fn from_inner(re: &Self::Inner) -> Self {
        let (r1, r2) = re;
        (T1::from_inner(r1), T2::from_inner(r2))
    }
}

impl<D, T1: Mappable<D>, T2: Mappable<D>> Mappable<D> for (T1, T2) {
    type Output<O> = (T1::Output<O>, T2::Output<O>);
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        let (s1, s2) = self;
        (s1.map_dual(&f), s2.map_dual(&f))
    }
}

impl<D, F, T1: DualStruct<D, F>, T2: DualStruct<D, F>, T3: DualStruct<D, F>> DualStruct<D, F>
    for (T1, T2, T3)
{
    type Real = (T1::Real, T2::Real, T3::Real);
    type Inner = (T1::Inner, T2::Inner, T3::Inner);
    fn re(&self) -> Self::Real {
        let (s1, s2, s3) = self;
        (s1.re(), s2.re(), s3.re())
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        let (r1, r2, r3) = inner;
        (T1::from_inner(r1), T2::from_inner(r2), T3::from_inner(r3))
    }
}

impl<D, T1: Mappable<D>, T2: Mappable<D>, T3: Mappable<D>> Mappable<D> for (T1, T2, T3) {
    type Output<O> = (T1::Output<O>, T2::Output<O>, T3::Output<O>);
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        let (s1, s2, s3) = self;
        (s1.map_dual(&f), s2.map_dual(&f), s3.map_dual(&f))
    }
}

impl<
        D,
        F,
        T1: DualStruct<D, F>,
        T2: DualStruct<D, F>,
        T3: DualStruct<D, F>,
        T4: DualStruct<D, F>,
    > DualStruct<D, F> for (T1, T2, T3, T4)
{
    type Real = (T1::Real, T2::Real, T3::Real, T4::Real);
    type Inner = (T1::Inner, T2::Inner, T3::Inner, T4::Inner);
    fn re(&self) -> Self::Real {
        let (s1, s2, s3, s4) = self;
        (s1.re(), s2.re(), s3.re(), s4.re())
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        let (r1, r2, r3, r4) = inner;
        (
            T1::from_inner(r1),
            T2::from_inner(r2),
            T3::from_inner(r3),
            T4::from_inner(r4),
        )
    }
}

impl<D, T1: Mappable<D>, T2: Mappable<D>, T3: Mappable<D>, T4: Mappable<D>> Mappable<D>
    for (T1, T2, T3, T4)
{
    type Output<O> = (T1::Output<O>, T2::Output<O>, T3::Output<O>, T4::Output<O>);
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        let (s1, s2, s3, s4) = self;
        (
            s1.map_dual(&f),
            s2.map_dual(&f),
            s3.map_dual(&f),
            s4.map_dual(&f),
        )
    }
}

impl<
        D,
        F,
        T1: DualStruct<D, F>,
        T2: DualStruct<D, F>,
        T3: DualStruct<D, F>,
        T4: DualStruct<D, F>,
        T5: DualStruct<D, F>,
    > DualStruct<D, F> for (T1, T2, T3, T4, T5)
{
    type Real = (T1::Real, T2::Real, T3::Real, T4::Real, T5::Real);
    type Inner = (T1::Inner, T2::Inner, T3::Inner, T4::Inner, T5::Inner);
    fn re(&self) -> Self::Real {
        let (s1, s2, s3, s4, s5) = self;
        (s1.re(), s2.re(), s3.re(), s4.re(), s5.re())
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        let (r1, r2, r3, r4, r5) = inner;
        (
            T1::from_inner(r1),
            T2::from_inner(r2),
            T3::from_inner(r3),
            T4::from_inner(r4),
            T5::from_inner(r5),
        )
    }
}

impl<D, T1: Mappable<D>, T2: Mappable<D>, T3: Mappable<D>, T4: Mappable<D>, T5: Mappable<D>>
    Mappable<D> for (T1, T2, T3, T4, T5)
{
    type Output<O> = (
        T1::Output<O>,
        T2::Output<O>,
        T3::Output<O>,
        T4::Output<O>,
        T5::Output<O>,
    );
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        let (s1, s2, s3, s4, s5) = self;
        (
            s1.map_dual(&f),
            s2.map_dual(&f),
            s3.map_dual(&f),
            s4.map_dual(&f),
            s5.map_dual(&f),
        )
    }
}

impl<D, F, T: DualStruct<D, F>, const N: usize> DualStruct<D, F> for [T; N] {
    type Real = [T::Real; N];
    type Inner = [T::Inner; N];
    fn re(&self) -> Self::Real {
        self.each_ref().map(|x| x.re())
    }
    fn from_inner(re: &Self::Inner) -> Self {
        re.each_ref().map(T::from_inner)
    }
}

impl<D, T: Mappable<D>, const N: usize> Mappable<D> for [T; N] {
    type Output<O> = [T::Output<O>; N];
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        self.map(|x| x.map_dual(&f))
    }
}

impl<D, F, T: DualStruct<D, F>> DualStruct<D, F> for Option<T> {
    type Real = Option<T::Real>;
    type Inner = Option<T::Inner>;
    fn re(&self) -> Self::Real {
        self.as_ref().map(|x| x.re())
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        inner.as_ref().map(|x| T::from_inner(x))
    }
}

impl<D, T: Mappable<D>> Mappable<D> for Option<T> {
    type Output<O> = Option<T::Output<O>>;
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        self.map(|x| x.map_dual(f))
    }
}

impl<D, T: Mappable<D>, E> Mappable<D> for Result<T, E> {
    type Output<O> = Result<T::Output<O>, E>;
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        self.map(|x| x.map_dual(f))
    }
}

impl<D, F, T: DualStruct<D, F>> DualStruct<D, F> for Vec<T> {
    type Real = Vec<T::Real>;
    type Inner = Vec<T::Inner>;
    fn re(&self) -> Self::Real {
        self.iter().map(|x| x.re()).collect()
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        inner.iter().map(|x| T::from_inner(x)).collect()
    }
}

impl<D, T: Mappable<D>> Mappable<D> for Vec<T> {
    type Output<O> = Vec<T::Output<O>>;
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        self.into_iter().map(|x| x.map_dual(&f)).collect()
    }
}

impl<D, F, T: DualStruct<D, F>, K: Clone + Eq + Hash> DualStruct<D, F> for HashMap<K, T> {
    type Real = HashMap<K, T::Real>;
    type Inner = HashMap<K, T::Inner>;
    fn re(&self) -> Self::Real {
        self.iter().map(|(k, x)| (k.clone(), x.re())).collect()
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        inner
            .iter()
            .map(|(k, x)| (k.clone(), T::from_inner(x)))
            .collect()
    }
}

impl<D, T: Mappable<D>, K: Eq + Hash> Mappable<D> for HashMap<K, T> {
    type Output<O> = HashMap<K, T::Output<O>>;
    fn map_dual<M: Fn(D) -> O, O>(self, f: M) -> Self::Output<O> {
        self.into_iter().map(|(k, x)| (k, x.map_dual(&f))).collect()
    }
}

impl<D: DualNum<F>, F: DualNumFloat, R: Dim, C: Dim> DualStruct<D, F> for OMatrix<D, R, C>
where
    DefaultAllocator: Allocator<R, C>,
    D::Inner: DualNum<F>,
{
    type Real = OMatrix<F, R, C>;
    type Inner = OMatrix<D::Inner, R, C>;
    fn re(&self) -> Self::Real {
        self.map(|x| x.re())
    }
    fn from_inner(inner: &Self::Inner) -> Self {
        inner.map(|x| D::from_inner(&x))
    }
}

impl<D: Scalar, R: Dim, C: Dim> Mappable<Self> for OMatrix<D, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output<O> = O;
    fn map_dual<M: Fn(Self) -> O, O>(self, f: M) -> O {
        f(self)
    }
}
