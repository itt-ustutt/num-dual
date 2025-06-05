use crate::{first_derivative, jacobian, Dual, DualNum, DualNumFloat, DualVec};
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, OMatrix, OVector, SVector, U1};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

/// Calculate the derivative of the unary implicit function
///         g(x, args) = 0
/// ```
/// # use num_dual::{implicit_derivative_unary, DualNum, Dual2_64};
/// # use approx::assert_relative_eq;
/// let y = Dual2_64::from(25.0).derivative();
/// let x = implicit_derivative_unary(|x,y| x.powi(2)-y, 5.0f64, &y);
/// assert_relative_eq!(x.re, y.sqrt().re, max_relative=1e-16);
/// assert_relative_eq!(x.v1, y.sqrt().v1, max_relative=1e-16);
/// assert_relative_eq!(x.v2, y.sqrt().v2, max_relative=1e-16);
/// ```
pub fn implicit_derivative_unary<G, D: DualNum<F>, F: DualNumFloat, A: Lift<D, F>>(
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
pub fn implicit_derivative_binary<G, D: DualNum<F>, F: DualNumFloat, A: Lift<D, F>>(
    g: G,
    x: F,
    y: F,
    args: &A,
) -> [D; 2]
where
    G: Fn(
        DualVec<D, F, Const<2>>,
        DualVec<D, F, Const<2>>,
        &A::Lifted<DualVec<D, F, Const<2>>>,
    ) -> [DualVec<D, F, Const<2>>; 2],
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
        let [[j00, j01], [j10, j11]] = jac.data.0;
        let det = (j00.clone() * &j11 - j01.clone() * &j10).recip();
        x -= (j11 * &f0 - j01 * &f1) * &det;
        y -= (j00 * &f1 - j10 * &f0) * &det;
    }
    [x, y]
}

/// Calculate the derivative of the multivariate implicit function
///         g(x, args) = 0
/// ```
/// # use num_dual::{implicit_derivative, Dual64};
/// # use approx::assert_relative_eq;
/// # use nalgebra::SVector;
/// let a = Dual64::from(4.0).derivative();
/// let x = implicit_derivative(
///     |x, a| SVector::from([x[0] * x[1] - a, x[0] + x[1] - a - 1.0]),
///     SVector::from([1.0f64, 4.0f64]),
///     &a,
///     );
/// assert_relative_eq!(x[0].re, 1.0, max_relative = 1e-16);
/// assert_relative_eq!(x[0].eps, 0.0, max_relative = 1e-16);
/// assert_relative_eq!(x[1].re, a.re, max_relative = 1e-16);
/// assert_relative_eq!(x[1].eps, a.eps, max_relative = 1e-16);
/// ```
pub fn implicit_derivative<G, D: DualNum<F> + Copy, F: DualNumFloat, A: Lift<D, F>, N: Dim>(
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

pub trait Lift<D, F> {
    type Lifted<D2: DualNum<F, Inner = D>>;
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2>;
}

impl<D, F> Lift<D, F> for () {
    type Lifted<D2: DualNum<F, Inner = D>> = ();
    fn lift<D2: DualNum<F, Inner = D>>(&self) {}
}

impl Lift<Self, Self> for f32 {
    type Lifted<D: DualNum<Self, Inner = Self>> = D;
    fn lift<D: DualNum<Self, Inner = Self>>(&self) -> D {
        D::from_inner(*self)
    }
}

impl Lift<Self, Self> for f64 {
    type Lifted<D: DualNum<Self, Inner = Self>> = D;
    fn lift<D: DualNum<Self, Inner = Self>>(&self) -> D {
        D::from_inner(*self)
    }
}

impl<D, F, T1: Lift<D, F>, T2: Lift<D, F>> Lift<D, F> for (T1, T2) {
    type Lifted<D2: DualNum<F, Inner = D>> = (T1::Lifted<D2>, T2::Lifted<D2>);
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2) = self;
        (s1.lift(), s2.lift())
    }
}

impl<D, F, T1: Lift<D, F>, T2: Lift<D, F>, T3: Lift<D, F>> Lift<D, F> for (T1, T2, T3) {
    type Lifted<D2: DualNum<F, Inner = D>> = (T1::Lifted<D2>, T2::Lifted<D2>, T3::Lifted<D2>);
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3) = self;
        (s1.lift(), s2.lift(), s3.lift())
    }
}

impl<D, F, T1: Lift<D, F>, T2: Lift<D, F>, T3: Lift<D, F>, T4: Lift<D, F>> Lift<D, F>
    for (T1, T2, T3, T4)
{
    type Lifted<D2: DualNum<F, Inner = D>> = (
        T1::Lifted<D2>,
        T2::Lifted<D2>,
        T3::Lifted<D2>,
        T4::Lifted<D2>,
    );
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3, s4) = self;
        (s1.lift(), s2.lift(), s3.lift(), s4.lift())
    }
}

impl<D, F, T1: Lift<D, F>, T2: Lift<D, F>, T3: Lift<D, F>, T4: Lift<D, F>, T5: Lift<D, F>>
    Lift<D, F> for (T1, T2, T3, T4, T5)
{
    type Lifted<D2: DualNum<F, Inner = D>> = (
        T1::Lifted<D2>,
        T2::Lifted<D2>,
        T3::Lifted<D2>,
        T4::Lifted<D2>,
        T5::Lifted<D2>,
    );
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        let (s1, s2, s3, s4, s5) = self;
        (s1.lift(), s2.lift(), s3.lift(), s4.lift(), s5.lift())
    }
}

impl<D, F, T: Lift<D, F>, const N: usize> Lift<D, F> for [T; N] {
    type Lifted<D2: DualNum<F, Inner = D>> = [T::Lifted<D2>; N];
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.each_ref().map(|x| x.lift())
    }
}

impl<D, F, T: Lift<D, F>> Lift<D, F> for Vec<T> {
    type Lifted<D2: DualNum<F, Inner = D>> = Vec<T::Lifted<D2>>;
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.iter().map(|x| x.lift()).collect()
    }
}

impl<D, F, T: Lift<D, F>, K: Clone + Eq + Hash> Lift<D, F> for HashMap<K, T> {
    type Lifted<D2: DualNum<F, Inner = D>> = HashMap<K, T::Lifted<D2>>;
    fn lift<D2: DualNum<F, Inner = D>>(&self) -> Self::Lifted<D2> {
        self.iter().map(|(k, x)| (k.clone(), x.lift())).collect()
    }
}

impl<D: DualNum<F>, F, R: Dim, C: Dim> Lift<D, F> for OMatrix<D, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Lifted<D2: DualNum<F, Inner = D>> = OMatrix<D2, R, C>;
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
}
