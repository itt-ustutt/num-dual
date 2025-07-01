use crate::{first_derivative, jacobian, Dual, DualNum, DualNumFloat, DualSVec, DualVec};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, SVector, Scalar, U1, U2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

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
pub fn implicit_derivative<G, D: DualNum<F>, F: DualNumFloat, A: DualStruct<D, F>>(
    g: G,
    x: F,
    args: &A,
) -> D
where
    G: Fn(Dual<D, F>, &A::Lifted<Dual<D, F>>) -> Dual<D, F>,
{
    let mut x = D::from(x);
    let args = args.lift();
    for _ in 0..D::NDERIV {
        let (f, df) = first_derivative(|x| g(x, &args), x.clone());
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
pub fn implicit_derivative_binary<G, D: DualNum<F>, F: DualNumFloat, A: DualStruct<D, F>>(
    g: G,
    x: F,
    y: F,
    args: &A,
) -> [D; 2]
where
    G: Fn(
        DualVec<D, F, U2>,
        DualVec<D, F, U2>,
        &A::Lifted<DualVec<D, F, U2>>,
    ) -> [DualVec<D, F, U2>; 2],
{
    let mut x = D::from(x);
    let mut y = D::from(y);
    let args = args.lift();
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
    A: DualStruct<D, F>,
    N: Dim,
>(
    g: G,
    x: OVector<F, N>,
    args: &A,
) -> OVector<D, N>
where
    DefaultAllocator: Allocator<N> + Allocator<N, N> + Allocator<U1, N>,
    G: Fn(
        OVector<DualVec<D, F, N>, N>,
        &A::Lifted<DualVec<D, F, N>>,
    ) -> OVector<DualVec<D, F, N>, N>,
{
    let mut x = x.map(D::from);
    let args = args.lift();
    for _ in 0..D::NDERIV {
        let (f, jac) = jacobian(|x| g(x, &args), x.clone());
        x -= LU::new(jac).unwrap().solve(&f);
    }
    x
}

/// An implicit function g(x, args) = 0 for which derivatives of x can be
/// calculated with the [ImplicitDerivative] struct.
pub trait ImplicitFunction<F> {
    /// data type of the parameter struct, needs to implement [DualStruct].
    type Parameters<D: DualNum<F>>: DualStruct<D, F>;

    /// data type of the variable `x`, needs to be either `D`, `[D; 2]`, or `SVector<D, N>`.
    type Variable<D>;

    /// implementation of the residual function g(x, args) = 0.
    fn residual<D: DualNum<F> + Copy>(
        parameters: &Self::Parameters<D>,
        x: Self::Variable<D>,
    ) -> Self::Variable<D>;
}

/// Helper struct that stores parameters in dual and real form and provides functions
/// for evaluating real residuals (for external solvers) and implicit derivatives for
/// arbitrary dual numbers.
pub struct ImplicitDerivative<G: ImplicitFunction<F>, D: DualNum<F>, F: DualNum<F>> {
    base: G::Parameters<F>,
    derivative: G::Parameters<D>,
}

impl<G: ImplicitFunction<F>, D: DualNum<F> + Copy, F: DualNum<F> + DualNumFloat>
    ImplicitDerivative<G, D, F>
where
    G::Parameters<D>: DualStruct<D, F, Real = G::Parameters<F>>,
{
    pub fn new(_: G, parameters: G::Parameters<D>) -> Self {
        Self {
            base: parameters.real(),
            derivative: parameters,
        }
    }

    /// Evaluate the (real) residual for a scalar function.
    pub fn residual(&self, x: F) -> F
    where
        G: ImplicitFunction<F, Variable<F> = F>,
    {
        G::residual(&self.base, x)
    }

    /// Evaluate the (real) residual for a bivariate function.
    pub fn residual_binary(&self, x: F, y: F) -> [F; 2]
    where
        G: ImplicitFunction<F, Variable<F> = [F; 2]>,
    {
        G::residual(&self.base, [x, y])
    }

    /// Evaluate the (real) residual for a multivariate function.
    pub fn residual_vec<N: Dim>(&self, x: OVector<F, N>) -> OVector<F, N>
    where
        DefaultAllocator: Allocator<N>,
        G: ImplicitFunction<F, Variable<F> = OVector<F, N>>,
    {
        G::residual(&self.base, x)
    }

    /// Evaluate the implicit derivative for a scalar function.
    pub fn implicit_derivative(&self, x: F) -> D
    where
        G: ImplicitFunction<F, Variable<Dual<D, F>> = Dual<D, F>>,
        G::Parameters<D>: DualStruct<D, F, Lifted<Dual<D, F>> = G::Parameters<Dual<D, F>>>,
    {
        implicit_derivative(|x, args| G::residual(args, x), x, &self.derivative)
    }

    /// Evaluate the implicit derivative for a bivariate function.
    pub fn implicit_derivative_binary(&self, x: F, y: F) -> [D; 2]
    where
        G: ImplicitFunction<F, Variable<DualVec<D, F, U2>> = [DualVec<D, F, U2>; 2]>,
        G::Parameters<D>:
            DualStruct<D, F, Lifted<DualVec<D, F, U2>> = G::Parameters<DualVec<D, F, U2>>>,
    {
        implicit_derivative_binary(
            |x, y, args| G::residual(args, [x, y]),
            x,
            y,
            &self.derivative,
        )
    }

    /// Evaluate the implicit derivative for a multivariate function.
    pub fn implicit_derivative_vec<const N: usize>(&self, x: SVector<F, N>) -> SVector<D, N>
    where
        G: ImplicitFunction<F, Variable<DualSVec<D, F, N>> = SVector<DualSVec<D, F, N>, N>>,
        G::Parameters<D>:
            DualStruct<D, F, Lifted<DualSVec<D, F, N>> = G::Parameters<DualSVec<D, F, N>>>,
    {
        implicit_derivative_vec(|x, args| G::residual(args, x), x, &self.derivative)
    }
}

/// A struct that contains dual numbers. Needed for arbitrary arguments in [ImplicitFunction].
///
/// The trait is implemented for all dual types themselves, and common data types (tuple, vec,
/// array, ...) and can be implemented for custom data types to achieve full flexibility.
pub trait DualStruct<D, F> {
    type Real;
    type Lifted<D2: DualNum<F, Inner = D>>;
    fn real(&self) -> Self::Real;
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2>;
}

impl<D, F> DualStruct<D, F> for () {
    type Real = ();
    type Lifted<D2: DualNum<F, Inner = D>> = ();
    fn real(&self) {}
    fn lift<D2: DualNum<F, Inner = D>>(&self) {}
}

impl DualStruct<f32, f32> for f32 {
    type Real = f32;
    type Lifted<D: DualNum<f32, Inner = f32>> = D;
    fn real(&self) -> f32 {
        *self
    }
    fn lift<D: DualNum<f32, Inner = f32>>(&self) -> D {
        D::from_inner(*self)
    }
}

impl DualStruct<f64, f64> for f64 {
    type Real = f64;
    type Lifted<D: DualNum<f64, Inner = f64>> = D;
    fn real(&self) -> f64 {
        *self
    }
    fn lift<D: DualNum<f64, Inner = f64>>(&self) -> D {
        D::from_inner(*self)
    }
}

impl<D, F, T1: DualStruct<D, F>, T2: DualStruct<D, F>> DualStruct<D, F> for (T1, T2) {
    type Real = (T1::Real, T2::Real);
    type Lifted<D2: DualNum<F, Inner = D>> = (T1::Lifted<D2>, T2::Lifted<D2>);
    fn real(&self) -> Self::Real {
        let (s1, s2) = self;
        (s1.real(), s2.real())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2) = self;
        (s1.lift(), s2.lift())
    }
}

impl<D, F, T1: DualStruct<D, F>, T2: DualStruct<D, F>, T3: DualStruct<D, F>> DualStruct<D, F>
    for (T1, T2, T3)
{
    type Real = (T1::Real, T2::Real, T3::Real);
    type Lifted<D2: DualNum<F, Inner = D>> = (T1::Lifted<D2>, T2::Lifted<D2>, T3::Lifted<D2>);
    fn real(&self) -> Self::Real {
        let (s1, s2, s3) = self;
        (s1.real(), s2.real(), s3.real())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3) = self;
        (s1.lift(), s2.lift(), s3.lift())
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
    type Lifted<D2: DualNum<F, Inner = D>> = (
        T1::Lifted<D2>,
        T2::Lifted<D2>,
        T3::Lifted<D2>,
        T4::Lifted<D2>,
    );
    fn real(&self) -> Self::Real {
        let (s1, s2, s3, s4) = self;
        (s1.real(), s2.real(), s3.real(), s4.real())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3, s4) = self;
        (s1.lift(), s2.lift(), s3.lift(), s4.lift())
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
    type Lifted<D2: DualNum<F, Inner = D>> = (
        T1::Lifted<D2>,
        T2::Lifted<D2>,
        T3::Lifted<D2>,
        T4::Lifted<D2>,
        T5::Lifted<D2>,
    );
    fn real(&self) -> Self::Real {
        let (s1, s2, s3, s4, s5) = self;
        (s1.real(), s2.real(), s3.real(), s4.real(), s5.real())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3, s4, s5) = self;
        (s1.lift(), s2.lift(), s3.lift(), s4.lift(), s5.lift())
    }
}

impl<D, F, T: DualStruct<D, F>, const N: usize> DualStruct<D, F> for [T; N] {
    type Real = [T::Real; N];
    type Lifted<D2: DualNum<F, Inner = D>> = [T::Lifted<D2>; N];
    fn real(&self) -> Self::Real {
        self.each_ref().map(|x| x.real())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.each_ref().map(|x| x.lift())
    }
}

impl<D, F, T: DualStruct<D, F>> DualStruct<D, F> for Vec<T> {
    type Real = Vec<T::Real>;
    type Lifted<D2: DualNum<F, Inner = D>> = Vec<T::Lifted<D2>>;
    fn real(&self) -> Self::Real {
        self.iter().map(|x| x.real()).collect()
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.iter().map(|x| x.lift()).collect()
    }
}

impl<D, F, T: DualStruct<D, F>, K: Clone + Eq + Hash> DualStruct<D, F> for HashMap<K, T> {
    type Real = HashMap<K, T::Real>;
    type Lifted<D2: DualNum<F, Inner = D>> = HashMap<K, T::Lifted<D2>>;
    fn real(&self) -> Self::Real {
        self.iter().map(|(k, x)| (k.clone(), x.real())).collect()
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.iter().map(|(k, x)| (k.clone(), x.lift())).collect()
    }
}

impl<D: DualNum<F>, F: Scalar, R: Dim, C: Dim> DualStruct<D, F> for OMatrix<D, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Real = OMatrix<F, R, C>;
    type Lifted<D2: DualNum<F, Inner = D>> = OMatrix<D2, R, C>;
    fn real(&self) -> Self::Real {
        self.map(|x| x.re())
    }
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.map(|x| D2::from_inner(x))
    }
}

// TODO: Later, replace the LU in linalg with this

#[derive(Debug)]
struct LinAlgError();

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

struct LU<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    a: OMatrix<T, D, D>,
    p: OVector<usize, D>,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Float, D: Dim> LU<T, F, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    fn new(mut a: OMatrix<T, D, D>) -> Result<Self, LinAlgError> {
        let (n, _) = a.shape_generic();
        let mut p = OVector::zeros_generic(n, U1);
        let n = n.value();

        for i in 0..n {
            p[i] = i;
        }

        for i in 0..n {
            let mut max_a = F::zero();
            let mut imax = i;

            for k in i..n {
                let abs_a = a[(k, i)].abs();
                if abs_a.re() > max_a {
                    max_a = abs_a.re();
                    imax = k;
                }
            }

            if max_a.is_zero() {
                return Err(LinAlgError());
            }

            if imax != i {
                let j = p[i];
                p[i] = p[imax];
                p[imax] = j;

                for j in 0..n {
                    let ptr = a[(i, j)];
                    a[(i, j)] = a[(imax, j)];
                    a[(imax, j)] = ptr;
                }
            }

            for j in i + 1..n {
                a[(j, i)] = a[(j, i)] / a[(i, i)];

                for k in i + 1..n {
                    a[(j, k)] = a[(j, k)] - a[(j, i)] * a[(i, k)];
                }
            }
        }
        Ok(Self {
            a,
            p,
            f: PhantomData,
        })
    }

    fn solve(&self, b: &OVector<T, D>) -> OVector<T, D> {
        let (n, _) = b.shape_generic();
        let mut x = OVector::zeros_generic(n, U1);
        let n = n.value();

        for i in 0..n {
            x[i] = b[self.p[i]];

            for k in 0..i {
                x[i] = x[i] - self.a[(i, k)] * x[k];
            }
        }

        for i in (0..n).rev() {
            for k in i + 1..n {
                x[i] = x[i] - self.a[(i, k)] * x[k];
            }

            x[i] /= self.a[(i, i)];
        }

        x
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Dual64;
    use nalgebra::{SMatrix, SVector};

    #[test]
    fn test_solve_f64_nalgebra() {
        let a = SMatrix::from([[4.0, 6.0], [3.0, 3.0]]);
        let b = SVector::from([10.0, 12.0]);
        let lu = LU::new(a).unwrap();
        assert_eq!(lu.solve(&b), SVector::from([1.0, 2.0]));
    }

    #[test]
    fn test_solve_dual64_nalgebra() {
        let a = SMatrix::from([
            [Dual64::new(4.0, 3.0), Dual64::new(6.0, 1.0)],
            [Dual64::new(3.0, 3.0), Dual64::new(3.0, 2.0)],
        ]);
        let b = SVector::from([Dual64::new(10.0, 20.0), Dual64::new(12.0, 20.0)]);
        let lu = LU::new(a).unwrap();
        let x = lu.solve(&b);
        assert_eq!((x[0].re, x[0].eps, x[1].re, x[1].eps), (1.0, 2.0, 2.0, 1.0));
    }

    struct TestFunction;
    impl ImplicitFunction<f64> for TestFunction {
        type Parameters<D: DualNum<f64>> = D;
        type Variable<D> = D;

        fn residual<D: DualNum<f64> + Copy>(square: &D, x: D) -> D {
            *square - x * x
        }
    }

    struct TestFunction2;
    impl ImplicitFunction<f64> for TestFunction2 {
        type Parameters<D: DualNum<f64>> = (D, D);
        type Variable<D> = [D; 2];

        fn residual<D: DualNum<f64> + Copy>((square_sum, sum): &(D, D), [x, y]: [D; 2]) -> [D; 2] {
            [*square_sum - x * x - y * y, *sum - x - y]
        }
    }

    struct TestFunction3<const N: usize>;
    impl<const N: usize> ImplicitFunction<f64> for TestFunction3<N> {
        type Parameters<D: DualNum<f64>> = D;
        type Variable<D> = SVector<D, N>;

        fn residual<D: DualNum<f64> + Copy>(&square_sum: &D, x: SVector<D, N>) -> SVector<D, N> {
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
        println!("{}", func.residual(5.0));
        println!("{}", func.implicit_derivative(5.0));
        println!("{}", f.sqrt());
        assert_eq!(f.sqrt(), func.implicit_derivative(5.0));

        let a: crate::Dual64 = Dual::from(25.0).derivative();
        let b: crate::Dual64 = Dual::from(7.0);
        let func = ImplicitDerivative::new(TestFunction2, (a, b));
        println!("\n{:?}", func.residual_binary(4.0, 3.0));
        let [x, y] = func.implicit_derivative_binary(4.0, 3.0);
        let xa = (b + (a * 2.0 - b * b).sqrt()) * 0.5;
        let ya = (b - (a * 2.0 - b * b).sqrt()) * 0.5;
        println!("{x}, {y}");
        println!("{xa}, {ya}");
        assert_eq!(x, xa);
        assert_eq!(y, ya);

        let s: crate::Dual64 = Dual::from(30.0).derivative();
        let func = ImplicitDerivative::new(TestFunction3, s);
        println!(
            "\n{:?}",
            func.residual_vec(SVector::from([1.0, 2.0, 3.0, 4.0]))
        );
        let x = func.implicit_derivative_vec(SVector::from([1.0, 2.0, 3.0, 4.0]));
        let x0 = ((s - 5.0).sqrt() - 5.0) * 0.5;
        println!("{}, {}, {}, {}", x[0], x[1], x[2], x[3]);
        println!("{}, {}, {}, {}", x0 + 1.0, x0 + 2.0, x0 + 3.0, x0 + 4.0);
        assert_eq!(x0 + 1.0, x[0]);
        assert_eq!(x0 + 2.0, x[1]);
        assert_eq!(x0 + 3.0, x[2]);
        assert_eq!(x0 + 4.0, x[3]);
    }
}
