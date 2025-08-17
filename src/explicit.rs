use crate::*;
use nalgebra::{Const, DMatrix, DVector, Dyn, OVector, SVector, U1};

/// Evaluate the function `g` with extra arguments `args` that are automatically adjusted to the correct
/// dual number type.
pub fn partial<G: Fn(X, &A) -> O, T: DualNum<F>, F, X, A: DualStruct<T, F>, O>(
    g: G,
    args: &A::Inner,
) -> impl Fn(X) -> O {
    let args = A::from_inner(args);
    move |x| g(x, &args)
}

/// Evaluate the function `g` with extra arguments `args1` and `args2` that are automatically adjusted to the
/// correct dual number type.
pub fn partial2<
    G: Fn(X, &A1, &A2) -> O,
    T: DualNum<F>,
    F,
    X,
    A1: DualStruct<T, F>,
    A2: DualStruct<T, F>,
    O,
>(
    g: G,
    args1: &A1::Inner,
    args2: &A2::Inner,
) -> impl Fn(X) -> O {
    let args1 = A1::from_inner(args1);
    let args2 = A2::from_inner(args2);
    move |x| g(x, &args1, &args2)
}

/// Calculate the first derivative of a scalar function.
/// ```
/// # use num_dual::{first_derivative, DualNum};
/// let (f, df) = first_derivative(|x| x.powi(2), 5.0);
/// assert_eq!(f, 25.0);
/// assert_eq!(df, 10.0);
/// ```
pub fn first_derivative<G, T: DualNum<F>, F: DualNumFloat, O: Mappable<Dual<T, F>>>(
    g: G,
    x: T,
) -> O::Output<(T, T)>
where
    G: Fn(Dual<T, F>) -> O,
{
    let x = Dual::from_re(x).derivative();
    g(x).map_dual(|r| (r.re, r.eps))
}

/// Calculate the gradient of a scalar function
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualSVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<DualSVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g) = gradient(fun, &v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// ```
///
/// The variable vector can also be dynamically sized
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{gradient, DualNum, DualDVec64};
/// # use nalgebra::DVector;
/// let v = DVector::repeat(4, 2.0);
/// let fun = |v: DVector<DualDVec64>| v.iter().map(|v| v * v).sum::<DualDVec64>().sqrt();
/// let (f, g) = gradient(fun, &v);
/// assert_eq!(f, 4.0);
/// assert_relative_eq!(g[0], 0.5);
/// assert_relative_eq!(g[1], 0.5);
/// assert_relative_eq!(g[2], 0.5);
/// assert_relative_eq!(g[3], 0.5);
/// ```
pub fn gradient<G, T: DualNum<F>, F: DualNumFloat, D: Dim, O: Mappable<DualVec<T, F, D>>>(
    g: G,
    x: &OVector<T, D>,
) -> O::Output<(T, OVector<T, D>)>
where
    G: Fn(OVector<DualVec<T, F, D>, D>) -> O,
    DefaultAllocator: Allocator<D>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative_generic(r, c, i);
    }
    g(x).map_dual(|res| (res.re, res.eps.unwrap_generic(r, c)))
}

/// Calculate the Jacobian of a vector function.
/// ```
/// # use num_dual::{jacobian, DualSVec64, DualNum};
/// # use nalgebra::SVector;
/// let xy = SVector::from([5.0, 3.0, 2.0]);
/// let fun = |xy: SVector<DualSVec64<3>, 3>| SVector::from([
///                      xy[0] * xy[1].powi(3) * xy[2],
///                      xy[0].powi(2) * xy[1] * xy[2].powi(2)
///                     ]);
/// let (f, jac) = jacobian(fun, xy);
/// assert_eq!(f[0], 270.0);          // xy³z
/// assert_eq!(f[1], 300.0);          // x²yz²
/// assert_eq!(jac[(0,0)], 54.0);     // y³z
/// assert_eq!(jac[(0,1)], 270.0);    // 3xy²z
/// assert_eq!(jac[(0,2)], 135.0);    // xy³
/// assert_eq!(jac[(1,0)], 120.0);    // 2xyz²
/// assert_eq!(jac[(1,1)], 100.0);    // x²z²
/// assert_eq!(jac[(1,2)], 300.0);     // 2x²yz
/// ```
#[expect(clippy::type_complexity)]
pub fn jacobian<
    G,
    T: DualNum<F>,
    F: DualNumFloat,
    M: Dim,
    N: Dim,
    O: Mappable<OVector<DualVec<T, F, N>, M>>,
>(
    g: G,
    x: OVector<T, N>,
) -> O::Output<(OVector<T, M>, OMatrix<T, M, N>)>
where
    G: FnOnce(OVector<DualVec<T, F, N>, N>) -> O,
    DefaultAllocator: Allocator<M> + Allocator<N> + Allocator<M, N> + Allocator<U1, N>,
{
    let mut x = x.map(DualVec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps = Derivative::derivative_generic(r, c, i);
    }
    let res = g(x);
    let res = res.map_dual(|res| {
        let eps = OMatrix::from_rows(
            res.map(|res| res.eps.unwrap_generic(r, c).transpose())
                .as_slice(),
        );
        (res.map(|r| r.re), eps)
    });
    res
}

/// Calculate the second derivative of a univariate function.
/// ```
/// # use num_dual::{second_derivative, DualNum};
/// let (f, df, d2f) = second_derivative(|x| x.powi(2), 5.0);
/// assert_eq!(f, 25.0);       // x²
/// assert_eq!(df, 10.0);      // 2x
/// assert_eq!(d2f, 2.0);      // 2
/// ```
///
/// The argument can also be a dual number.
/// ```
/// # use num_dual::{second_derivative, Dual2, Dual64, DualNum};
/// let x = Dual64::new(5.0, 1.0);
/// let (f, df, d2f) = second_derivative(|x| x.powi(3), x);
/// assert_eq!(f.re, 125.0);    // x³
/// assert_eq!(f.eps, 75.0);    // 3x²
/// assert_eq!(df.re, 75.0);    // 3x²
/// assert_eq!(df.eps, 30.0);   // 6x
/// assert_eq!(d2f.re, 30.0);   // 6x
/// assert_eq!(d2f.eps, 6.0);   // 6
/// ```
pub fn second_derivative<G, T: DualNum<F>, F, O: Mappable<Dual2<T, F>>>(
    g: G,
    x: T,
) -> O::Output<(T, T, T)>
where
    G: Fn(Dual2<T, F>) -> O,
{
    let x = Dual2::from_re(x).derivative();
    g(x).map_dual(|r| (r.re, r.v1, r.v2))
}

/// Calculate second partial derivatives with respect to scalars.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{second_partial_derivative, DualNum, HyperDual64};
/// # use nalgebra::SVector;
/// let fun = |(x, y): (HyperDual64, HyperDual64)| (x.powi(2) + y.powi(2)).sqrt();
/// let (f, dfdx, dfdy, d2fdxdy) = second_partial_derivative(fun, (4.0, 3.0));
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(dfdx, 0.8);
/// assert_relative_eq!(dfdy, 0.6);
/// assert_relative_eq!(d2fdxdy, -0.096);
/// ```
pub fn second_partial_derivative<G, T: DualNum<F>, F, O: Mappable<HyperDual<T, F>>>(
    g: G,
    (x, y): (T, T),
) -> O::Output<(T, T, T, T)>
where
    G: Fn((HyperDual<T, F>, HyperDual<T, F>)) -> O,
{
    let x = HyperDual::from_re(x).derivative1();
    let y = HyperDual::from_re(y).derivative2();
    g((x, y)).map_dual(|r| (r.re, r.eps1, r.eps2, r.eps1eps2))
}

/// Calculate the Hessian of a scalar function.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{hessian, DualNum, Dual2SVec64};
/// # use nalgebra::SVector;
/// let v = SVector::from([4.0, 3.0]);
/// let fun = |v: SVector<Dual2SVec64<2>, 2>| (v[0].powi(2) + v[1].powi(2)).sqrt();
/// let (f, g, h) = hessian(fun, &v);
/// assert_eq!(f, 5.0);
/// assert_relative_eq!(g[0], 0.8);
/// assert_relative_eq!(g[1], 0.6);
/// assert_relative_eq!(h[(0,0)], 0.072);
/// assert_relative_eq!(h[(0,1)], -0.096);
/// assert_relative_eq!(h[(1,0)], -0.096);
/// assert_relative_eq!(h[(1,1)], 0.128);
/// ```
#[expect(clippy::type_complexity)]
pub fn hessian<G, T: DualNum<F>, F: DualNumFloat, D: Dim, O: Mappable<Dual2Vec<T, F, D>>>(
    g: G,
    x: &OVector<T, D>,
) -> O::Output<(T, OVector<T, D>, OMatrix<T, D, D>)>
where
    G: Fn(OVector<Dual2Vec<T, F, D>, D>) -> O,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    let mut x = x.map(Dual2Vec::from_re);
    let (r, c) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.v1 = Derivative::derivative_generic(c, r, i)
    }
    g(x).map_dual(|res| {
        (
            res.re,
            res.v1.unwrap_generic(c, r).transpose(),
            res.v2.unwrap_generic(r, r),
        )
    })
}

/// Calculate second partial derivatives with respect to vectors.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{partial_hessian, DualNum, HyperDualSVec64};
/// # use nalgebra::SVector;
/// let x = SVector::from([4.0, 3.0]);
/// let y = SVector::from([5.0]);
/// let fun = |(x, y): (SVector<HyperDualSVec64<2, 1>, 2>, SVector<HyperDualSVec64<2, 1>, 1>)|
///                 y[0] / (x[0].powi(2) + x[1].powi(2)).sqrt();
/// let (f, dfdx, dfdy, d2fdxdy) = partial_hessian(fun, (&x, &y));
/// assert_eq!(f, 1.0);
/// assert_relative_eq!(dfdx[0], -0.16);
/// assert_relative_eq!(dfdx[1], -0.12);
/// assert_relative_eq!(dfdy[0], 0.2);
/// assert_relative_eq!(d2fdxdy[0], -0.032);
/// assert_relative_eq!(d2fdxdy[1], -0.024);
/// ```
#[expect(clippy::type_complexity)]
pub fn partial_hessian<
    G,
    T: DualNum<F>,
    F: DualNumFloat,
    M: Dim,
    N: Dim,
    O: Mappable<HyperDualVec<T, F, M, N>>,
>(
    g: G,
    (x, y): (&OVector<T, M>, &OVector<T, N>),
) -> O::Output<(T, OVector<T, M>, OVector<T, N>, OMatrix<T, M, N>)>
where
    G: Fn(
        (
            OVector<HyperDualVec<T, F, M, N>, M>,
            OVector<HyperDualVec<T, F, M, N>, N>,
        ),
    ) -> O,
    DefaultAllocator: Allocator<N> + Allocator<M> + Allocator<M, N> + Allocator<U1, N>,
{
    let mut x = x.map(HyperDualVec::from_re);
    let mut y = y.map(HyperDualVec::from_re);
    let (m, _) = x.shape_generic();
    for (i, xi) in x.iter_mut().enumerate() {
        xi.eps1 = Derivative::derivative_generic(m, U1, i)
    }
    let (n, _) = y.shape_generic();
    for (i, yi) in y.iter_mut().enumerate() {
        yi.eps2 = Derivative::derivative_generic(U1, n, i)
    }
    g((x, y)).map_dual(|r| {
        (
            r.re,
            r.eps1.unwrap_generic(m, U1),
            r.eps2.unwrap_generic(U1, n).transpose(),
            r.eps1eps2.unwrap_generic(m, n),
        )
    })
}

/// Calculate the third derivative of a univariate function.
/// ```
/// # use num_dual::{third_derivative, DualNum};
/// let (f, df, d2f, d3f) = third_derivative(|x| x.powi(3), 5.0);
/// assert_eq!(f, 125.0);      // x³
/// assert_eq!(df, 75.0);      // 3x²
/// assert_eq!(d2f, 30.0);     // 6x
/// assert_eq!(d3f, 6.0);      // 6
/// ```
pub fn third_derivative<G, T: DualNum<F>, F, O: Mappable<Dual3<T, F>>>(
    g: G,
    x: T,
) -> O::Output<(T, T, T, T)>
where
    G: Fn(Dual3<T, F>) -> O,
{
    let x = Dual3::from_re(x).derivative();
    g(x).map_dual(|r| (r.re, r.v1, r.v2, r.v3))
}

/// Calculate third partial derivatives with respect to scalars.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{third_partial_derivative, DualNum, HyperHyperDual64};
/// # use nalgebra::SVector;
/// let fun = |(x, y, z): (HyperHyperDual64, HyperHyperDual64, HyperHyperDual64)| (x.powi(2) + y.powi(2) + z.powi(2)).powi(3);
/// let (f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz) = third_partial_derivative(fun, (1.0, 2.0, 3.0));
/// println!("{:?}", third_partial_derivative(fun, (1.0, 2.0, 3.0)));
/// assert_eq!(f, 2744.0);
/// assert_relative_eq!(dfdx, 1176.0);
/// assert_relative_eq!(dfdy, 2352.0);
/// assert_relative_eq!(dfdz, 3528.0);
/// assert_relative_eq!(d2fdxdy, 672.0);
/// assert_relative_eq!(d2fdxdz, 1008.0);
/// assert_relative_eq!(d2fdydz, 2016.0);
/// assert_relative_eq!(d3fdxdydz, 288.0);
/// ```
#[expect(clippy::type_complexity)]
pub fn third_partial_derivative<G, T: DualNum<F>, F, O: Mappable<HyperHyperDual<T, F>>>(
    g: G,
    (x, y, z): (T, T, T),
) -> O::Output<(T, T, T, T, T, T, T, T)>
where
    G: Fn(
        (
            HyperHyperDual<T, F>,
            HyperHyperDual<T, F>,
            HyperHyperDual<T, F>,
        ),
    ) -> O,
{
    let x = HyperHyperDual::from_re(x).derivative1();
    let y = HyperHyperDual::from_re(y).derivative2();
    let z = HyperHyperDual::from_re(z).derivative3();
    g((x, y, z)).map_dual(|r| {
        (
            r.re,
            r.eps1,
            r.eps2,
            r.eps3,
            r.eps1eps2,
            r.eps1eps3,
            r.eps2eps3,
            r.eps1eps2eps3,
        )
    })
}

/// Calculate the third partial derivative of a scalar function
/// with arbitrary many variables.
/// ```
/// # use approx::assert_relative_eq;
/// # use num_dual::{third_partial_derivative_vec, DualNum, HyperHyperDual64};
/// # use nalgebra::SVector;
/// let fun = |x: &[HyperHyperDual64]| x[0].powi(3)*x[1].powi(2);
/// let (f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz) = third_partial_derivative_vec(fun, &[1.0, 2.0], 0, 0, 1);
/// # println!("{:?}", third_partial_derivative_vec(fun, &[1.0, 2.0, 3.0], 0, 0, 1));
/// assert_eq!(f, 4.0);
/// assert_relative_eq!(dfdx, 12.0);
/// assert_relative_eq!(dfdy, 12.0);
/// assert_relative_eq!(dfdz, 4.0);
/// assert_relative_eq!(d2fdxdy, 24.0);
/// assert_relative_eq!(d2fdxdz, 12.0);
/// assert_relative_eq!(d2fdydz, 12.0);
/// assert_relative_eq!(d3fdxdydz, 24.0);
/// ```
#[expect(clippy::type_complexity)]
pub fn third_partial_derivative_vec<G, T: DualNum<F>, F, O: Mappable<HyperHyperDual<T, F>>>(
    g: G,
    x: &[T],
    i: usize,
    j: usize,
    k: usize,
) -> O::Output<(T, T, T, T, T, T, T, T)>
where
    G: Fn(&[HyperHyperDual<T, F>]) -> O,
{
    let mut x: Vec<_> = x
        .iter()
        .map(|x| HyperHyperDual::from_re(x.clone()))
        .collect();
    x[i].eps1 = T::one();
    x[j].eps2 = T::one();
    x[k].eps3 = T::one();
    g(&x).map_dual(|r| {
        (
            r.re,
            r.eps1,
            r.eps2,
            r.eps3,
            r.eps1eps2,
            r.eps1eps3,
            r.eps2eps3,
            r.eps1eps2eps3,
        )
    })
}

/// Evaluation of gradients, hessians, and partial (Nx1) hessians that is generic over the dimensionality
/// of the input vector.
pub trait Gradients: Dim
where
    DefaultAllocator: Allocator<Self>,
{
    type Dual<T: DualNum<F> + Copy, F: DualNumFloat>: DualNum<F, Inner = T> + Copy;
    type Dual2<T: DualNum<F> + Copy, F: DualNumFloat>: DualNum<F, Inner = T> + Copy;
    type HyperDual<T: DualNum<F> + Copy, F: DualNumFloat>: DualNum<F, Inner = T> + Copy;

    fn gradient<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<Self::Dual<T, F>, F>>(
        g: G,
        x: &OVector<T, Self>,
        args: &A::Inner,
    ) -> (T, OVector<T, Self>)
    where
        G: Fn(OVector<Self::Dual<T, F>, Self>, &A) -> Self::Dual<T, F>;

    fn hessian<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<Self::Dual2<T, F>, F>>(
        g: G,
        x: &OVector<T, Self>,
        args: &A::Inner,
    ) -> (T, OVector<T, Self>, OMatrix<T, Self, Self>)
    where
        G: Fn(OVector<Self::Dual2<T, F>, Self>, &A) -> Self::Dual2<T, F>,
        DefaultAllocator: Allocator<Self, Self>;

    fn partial_hessian<
        G,
        T: DualNum<F> + Copy,
        F: DualNumFloat,
        A: DualStruct<Self::HyperDual<T, F>, F>,
    >(
        g: G,
        x: &OVector<T, Self>,
        y: T,
        args: &A::Inner,
    ) -> (T, OVector<T, Self>, T, OVector<T, Self>)
    where
        G: Fn(
            OVector<Self::HyperDual<T, F>, Self>,
            Self::HyperDual<T, F>,
            &A,
        ) -> Self::HyperDual<T, F>;
}

impl<const N: usize> Gradients for Const<N> {
    type Dual<T: DualNum<F> + Copy, F: DualNumFloat> = DualSVec<T, F, N>;
    type Dual2<T: DualNum<F> + Copy, F: DualNumFloat> = Dual2Vec<T, F, Const<N>>;
    type HyperDual<T: DualNum<F> + Copy, F: DualNumFloat> = HyperDualVec<T, F, Const<N>, U1>;

    fn gradient<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<DualSVec<T, F, N>, F>>(
        g: G,
        x: &SVector<T, N>,
        args: &A::Inner,
    ) -> (T, SVector<T, N>)
    where
        G: Fn(SVector<DualSVec<T, F, N>, N>, &A) -> DualSVec<T, F, N>,
    {
        gradient(partial(g, args), x)
    }

    fn hessian<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<Self::Dual2<T, F>, F>>(
        g: G,
        x: &OVector<T, Self>,
        args: &A::Inner,
    ) -> (T, OVector<T, Self>, OMatrix<T, Self, Self>)
    where
        G: Fn(OVector<Self::Dual2<T, F>, Self>, &A) -> Self::Dual2<T, F>,
    {
        hessian(partial(g, args), x)
    }

    fn partial_hessian<
        G,
        T: DualNum<F> + Copy,
        F: DualNumFloat,
        A: DualStruct<Self::HyperDual<T, F>, F>,
    >(
        g: G,
        x: &OVector<T, Self>,
        y: T,
        args: &A::Inner,
    ) -> (T, OVector<T, Self>, T, OVector<T, Self>)
    where
        G: Fn(
            OVector<Self::HyperDual<T, F>, Self>,
            Self::HyperDual<T, F>,
            &A,
        ) -> Self::HyperDual<T, F>,
    {
        let (a, b, c, d) = partial_hessian(
            |(x, y)| {
                let [[y]] = y.data.0;
                g(x, y, &A::from_inner(args))
            },
            (x, &SVector::from([y])),
        );
        let [[c]] = c.data.0;
        (a, b, c, d)
    }
}

impl Gradients for Dyn {
    type Dual<T: DualNum<F> + Copy, F: DualNumFloat> = Dual<T, F>;
    type Dual2<T: DualNum<F> + Copy, F: DualNumFloat> = HyperDual<T, F>;
    type HyperDual<T: DualNum<F> + Copy, F: DualNumFloat> = HyperDual<T, F>;

    fn gradient<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<Dual<T, F>, F>>(
        g: G,
        x: &DVector<T>,
        args: &A::Inner,
    ) -> (T, DVector<T>)
    where
        G: Fn(OVector<Dual<T, F>, Dyn>, &A) -> Dual<T, F>,
    {
        let mut re = T::zero();
        let n = x.len();
        let args = A::from_inner(args);
        let grad = DVector::from_fn(n, |i, _| {
            let mut x = x.map(Dual::from_re);
            x[i].eps = T::one();
            let res = g(x, &args);
            re = res.re;
            res.eps
        });
        (re, grad)
    }

    fn hessian<G, T: DualNum<F> + Copy, F: DualNumFloat, A: DualStruct<HyperDual<T, F>, F>>(
        g: G,
        x: &DVector<T>,
        args: &A::Inner,
    ) -> (T, DVector<T>, DMatrix<T>)
    where
        G: Fn(DVector<HyperDual<T, F>>, &A) -> HyperDual<T, F>,
    {
        let mut re = T::zero();
        let n = x.len();
        let args = A::from_inner(args);
        let mut grad = DVector::zeros(n);
        let hessian = DMatrix::from_fn(n, n, |i, j| {
            let mut x = x.map(HyperDual::from_re);
            x[i].eps1 = T::one();
            x[j].eps2 = T::one();
            let res = g(x, &args);
            re = res.re;
            grad[i] = res.eps1;
            grad[j] = res.eps2;
            res.eps1eps2
        });
        (re, grad, hessian)
    }

    fn partial_hessian<
        G,
        T: DualNum<F> + Copy,
        F: DualNumFloat,
        A: DualStruct<HyperDual<T, F>, F>,
    >(
        g: G,
        x: &DVector<T>,
        y: T,
        args: &A::Inner,
    ) -> (T, DVector<T>, T, DVector<T>)
    where
        G: Fn(DVector<HyperDual<T, F>>, HyperDual<T, F>, &A) -> HyperDual<T, F>,
    {
        let mut re = T::zero();
        let n = x.len();
        let args = A::from_inner(args);
        let y = HyperDual::from_re(y).derivative2();
        let mut grad_x = DVector::zeros(n);
        let mut grad_y = T::zero();
        let hessian = DVector::from_fn(n, |i, _| {
            let mut x = x.map(HyperDual::from_re);
            x[i].eps1 = T::one();
            let res = g(x, y, &args);
            re = res.re;
            grad_x[i] = res.eps1;
            grad_y = res.eps2;
            res.eps1eps2
        });
        (re, grad_x, grad_y, hessian)
    }
}
