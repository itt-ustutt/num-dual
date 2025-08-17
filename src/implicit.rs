use crate::linalg::LU;
use crate::{
    first_derivative, jacobian, partial, Dual, DualNum, DualNumFloat, DualSVec, DualStruct, DualVec,
};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OVector, SVector, U1, U2};

/// Calculate the derivative of the unary implicit function
///         g(x, args) = 0
/// ```
/// # use num_dual::{implicit_derivative, DualNum, Dual2_64};
/// # use approx::assert_relative_eq;
/// let y = Dual2_64::from(25.0).derivative();
/// let x = implicit_derivative(|x,y| x.powi(2)-y, 5.0f64, &y);
/// assert_relative_eq!(x.re, y.sqrt().re, max_relative=1e-16);
/// assert_relative_eq!(x.v1, y.sqrt().v1, max_relative=1e-16);
/// assert_relative_eq!(x.v2, y.sqrt().v2, max_relative=1e-16);
/// ```
pub fn implicit_derivative<G, D: DualNum<F>, F: DualNumFloat, A: DualStruct<Dual<D, F>, F>>(
    g: G,
    x: F,
    args: &A::Inner,
) -> D
where
    G: Fn(Dual<D, F>, &A) -> Dual<D, F>,
{
    let mut x = D::from(x);
    for _ in 0..D::NDERIV {
        let (f, df) = first_derivative(partial(&g, args), x.clone());
        x -= f / df;
    }
    x
}

/// Calculate the derivative of the binary implicit function
///         g(x, y, args) = 0
/// ```
/// # use num_dual::{implicit_derivative_binary, Dual64};
/// # use approx::assert_relative_eq;
/// let a = Dual64::from(4.0).derivative();
/// let [x, y] =
///     implicit_derivative_binary(|x, y, a| [x * y - a, x + y - a - 1.0], 1.0f64, 4.0f64, &a);
/// assert_relative_eq!(x.re, 1.0, max_relative = 1e-16);
/// assert_relative_eq!(x.eps, 0.0, max_relative = 1e-16);
/// assert_relative_eq!(y.re, a.re, max_relative = 1e-16);
/// assert_relative_eq!(y.eps, a.eps, max_relative = 1e-16);
/// ```
pub fn implicit_derivative_binary<
    G,
    D: DualNum<F>,
    F: DualNumFloat,
    A: DualStruct<DualVec<D, F, U2>, F>,
>(
    g: G,
    x: F,
    y: F,
    args: &A::Inner,
) -> [D; 2]
where
    G: Fn(DualVec<D, F, U2>, DualVec<D, F, U2>, &A) -> [DualVec<D, F, U2>; 2],
{
    let mut x = D::from(x);
    let mut y = D::from(y);
    let args = A::from_inner(args);
    for _ in 0..D::NDERIV {
        let (f, jac) = jacobian(
            |x| {
                let [[x, y]] = x.data.0;
                SVector::from(g(x, y, &args))
            },
            SVector::from([x.clone(), y.clone()]),
        );
        let [[f0, f1]] = f.data.0;
        let [[j00, j10], [j01, j11]] = jac.data.0;
        let det = (j00.clone() * &j11 - j01.clone() * &j10).recip();
        x -= (j11 * &f0 - j01 * &f1) * &det;
        y -= (j00 * &f1 - j10 * &f0) * &det;
    }
    [x, y]
}

/// Calculate the derivative of the multivariate implicit function
///         g(x, args) = 0
/// ```
/// # use num_dual::{implicit_derivative_vec, Dual64};
/// # use approx::assert_relative_eq;
/// # use nalgebra::SVector;
/// let a = Dual64::from(4.0).derivative();
/// let x = implicit_derivative_vec(
///     |x, a| SVector::from([x[0] * x[1] - a, x[0] + x[1] - a - 1.0]),
///     SVector::from([1.0f64, 4.0f64]),
///     &a,
///     );
/// assert_relative_eq!(x[0].re, 1.0, max_relative = 1e-16);
/// assert_relative_eq!(x[0].eps, 0.0, max_relative = 1e-16);
/// assert_relative_eq!(x[1].re, a.re, max_relative = 1e-16);
/// assert_relative_eq!(x[1].eps, a.eps, max_relative = 1e-16);
/// ```
pub fn implicit_derivative_vec<
    G,
    D: DualNum<F> + Copy,
    F: DualNumFloat,
    A: DualStruct<DualVec<D, F, N>, F>,
    N: Dim,
>(
    g: G,
    x: OVector<F, N>,
    args: &A::Inner,
) -> OVector<D, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
    G: Fn(OVector<DualVec<D, F, N>, N>, &A) -> OVector<DualVec<D, F, N>, N>,
{
    let mut x = x.map(D::from);
    let args = A::from_inner(args);
    for _ in 0..D::NDERIV {
        let (f, jac) = jacobian(|x| g(x, &args), x.clone());
        x -= LU::new(jac).unwrap().solve(&f);
    }
    x
}

/// An implicit function g(x, args) = 0 for which derivatives of x can be
/// calculated with the [ImplicitDerivative] struct.
pub trait ImplicitFunction<F, const N: usize> {
    /// data type of the parameter struct, needs to implement [DualStruct].
    type Parameters<D>;

    /// implementation of the residual function g(x, args) = 0.
    fn residual<D: DualNum<F> + Copy>(
        x: SVector<D, N>,
        parameters: &Self::Parameters<D>,
    ) -> SVector<D, N>;
}

/// Helper struct that stores parameters in dual and real form and provides functions
/// for evaluating real residuals (for external solvers) and implicit derivatives for
/// arbitrary dual numbers.
pub struct ImplicitDerivative<
    G: ImplicitFunction<F, N>,
    D: DualNum<F> + Copy,
    F: DualNumFloat,
    const N: usize,
> {
    base: G::Parameters<D::Real>,
    derivative: G::Parameters<D>,
}

impl<
        G: ImplicitFunction<F, N>,
        D: DualNum<F> + Copy,
        F: DualNum<F> + DualNumFloat,
        const N: usize,
    > ImplicitDerivative<G, D, F, N>
where
    G::Parameters<D>: DualStruct<D, F, Real = G::Parameters<F>>,
{
    pub fn new(_: G, parameters: G::Parameters<D>) -> Self {
        Self {
            base: parameters.re(),
            derivative: parameters,
        }
    }

    /// Evaluate the (real) residual for a scalar function.
    pub fn residual<X: Into<SVector<F, N>>>(&self, x: X) -> SVector<F, N> {
        G::residual(x.into(), &self.base)
    }

    /// Evaluate the implicit derivative for a scalar function.
    pub fn implicit_derivative<
        X: Into<SVector<F, N>>,
        A: DualStruct<DualSVec<D, F, N>, F, Inner = G::Parameters<D>>,
    >(
        &self,
        x: X,
    ) -> SVector<D, N>
    where
        G: ImplicitFunction<F, N, Parameters<DualSVec<D, F, N>> = A>,
    {
        implicit_derivative_vec(G::residual::<DualSVec<D, F, N>>, x.into(), &self.derivative)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::SVector;

    struct TestFunction;
    impl ImplicitFunction<f64, 1> for TestFunction {
        type Parameters<D> = D;

        fn residual<D: DualNum<f64> + Copy>(x: SVector<D, 1>, square: &D) -> SVector<D, 1> {
            let [[x]] = x.data.0;
            SVector::from([*square - x * x])
        }
    }

    struct TestFunction2;
    impl ImplicitFunction<f64, 2> for TestFunction2 {
        type Parameters<D> = (D, D);

        fn residual<D: DualNum<f64> + Copy>(
            x: SVector<D, 2>,
            (square_sum, sum): &(D, D),
        ) -> SVector<D, 2> {
            let [[x, y]] = x.data.0;
            SVector::from([*square_sum - x * x - y * y, *sum - x - y])
        }
    }

    struct TestFunction3<const N: usize>;
    impl<const N: usize> ImplicitFunction<f64, N> for TestFunction3<N> {
        type Parameters<D> = D;

        fn residual<D: DualNum<f64> + Copy>(x: SVector<D, N>, &square_sum: &D) -> SVector<D, N> {
            let mut res = x;
            for i in 1..N {
                res[i] = x[i] - x[i - 1] - D::from(1.0);
            }
            res[0] = square_sum - x.dot(&x);
            res
        }
    }

    #[test]
    fn test() {
        let f: crate::Dual64 = Dual::from(25.0).derivative();
        let func = ImplicitDerivative::new(TestFunction, f);
        println!("{}", func.residual([5.0]));
        println!("{}", func.implicit_derivative([5.0]));
        println!("{}", f.sqrt());
        assert_eq!(f.sqrt(), func.implicit_derivative([5.0])[0]);

        let a: crate::Dual64 = Dual::from(25.0).derivative();
        let b: crate::Dual64 = Dual::from(7.0);
        let func = ImplicitDerivative::new(TestFunction2, (a, b));
        println!("\n{:?}", func.residual([4.0, 3.0]));
        let [[x, y]] = func.implicit_derivative([4.0, 3.0]).data.0;
        let xa = (b + (a * 2.0 - b * b).sqrt()) * 0.5;
        let ya = (b - (a * 2.0 - b * b).sqrt()) * 0.5;
        println!("{x}, {y}");
        println!("{xa}, {ya}");
        assert_eq!(x, xa);
        assert_eq!(y, ya);

        let s: crate::Dual64 = Dual::from(30.0).derivative();
        let func = ImplicitDerivative::new(TestFunction3, s);
        println!("\n{:?}", func.residual([1.0, 2.0, 3.0, 4.0]));
        let x = func.implicit_derivative([1.0, 2.0, 3.0, 4.0]);
        let x0 = ((s - 5.0).sqrt() - 5.0) * 0.5;
        println!("{}, {}, {}, {}", x[0], x[1], x[2], x[3]);
        println!("{}, {}, {}, {}", x0 + 1.0, x0 + 2.0, x0 + 3.0, x0 + 4.0);
        assert_eq!(x0 + 1.0, x[0]);
        assert_eq!(x0 + 2.0, x[1]);
        assert_eq!(x0 + 3.0, x[2]);
        assert_eq!(x0 + 4.0, x[3]);
    }
}
