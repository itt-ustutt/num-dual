use crate::*;
use ndarray::*;
use ndarray_linalg::convert::replicate;
use ndarray_linalg::error::Result;
use ndarray_linalg::*;

impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for HyperDual<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for HD3<T, F> {}

type LU64 = LUFactorized<OwnedRepr<f64>>;

pub trait FactorizeIntoDual {
    fn factorize_into_dual(self) -> Result<LU64>;
}

impl<D: DualNum<f64>> FactorizeIntoDual for Array2<D> {
    fn factorize_into_dual(self) -> Result<LU64> {
        self.map(|s| s.re()).factorize_into()
    }
}

pub trait SolveDual<D: DualNum<f64>>: FactorizeIntoDual + Clone {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve(&self, b: &Array1<D>) -> Result<Array1<D>> {
        let mut b = replicate(b);
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_into(&self, mut b: Array1<D>) -> Result<Array1<D>> {
        self.solve_inplace(&mut b)?;
        Ok(b)
    }
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve_inplace<'a>(&self, b: &'a mut Array1<D>) -> Result<&'a mut Array1<D>> {
        let lu = self.clone().factorize_into_dual()?;
        self.solve_recursive_inplace(&lu, b)
    }

    fn solve_recursive_into(&self, lu: &LU64, mut b: Array1<D>) -> Result<Array1<D>> {
        self.solve_recursive_inplace(lu, &mut b)?;
        Ok(b)
    }

    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<D>,
    ) -> Result<&'a mut Array1<D>>;
}

impl SolveDual<f64> for Array2<f64> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[1.0, 3.0],
    ///                [5.0, 7.0]]);
    /// let b = arr1(&[10.0, 26.0]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-14);
    /// ```
    fn solve_inplace<'a>(&self, b: &'a mut Array1<f64>) -> Result<&'a mut Array1<f64>> {
        <Self as Solve<f64>>::solve_inplace(self, b)
    }

    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<f64>,
    ) -> Result<&'a mut Array1<f64>> {
        lu.solve_inplace(b)
    }
}

impl<D: DualNum<f64> + 'static> SolveDual<Dual<D, f64>> for Array2<Dual<D, f64>>
where
    Array2<D>: SolveDual<D>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::Dual64;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[Dual64::new(1.0, 2.0), Dual64::new(3.0, 4.0)],
    ///                [Dual64::new(5.0, 6.0), Dual64::new(7.0, 8.0)]]);
    /// let b = arr1(&[Dual64::new(10.0, 28.0), Dual64::new(26.0, 68.0)]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps, 4.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<Dual<D, f64>>,
    ) -> Result<&'a mut Array1<Dual<D, f64>>> {
        let f = self.mapv(|s| s.re);
        let dx0 = f.solve_recursive_into(lu, b.mapv(|b| b.re))?;
        let dx1 =
            f.solve_recursive_into(lu, b.mapv(|b| b.eps) - &self.mapv(|s| s.eps).dot(&dx0))?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&mut *b)
            .apply(|&dx0, &dx1, b| *b = Dual::new(dx0, dx1));
        Ok(b)
    }
}

impl<D: DualNum<f64> + 'static> SolveDual<HyperDual<D, f64>> for Array2<HyperDual<D, f64>>
where
    Array2<D>: SolveDual<D>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::HyperDual64;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[HyperDual64::new(1.0, 2.0, 3.0, 4.0), HyperDual64::new(2.0, 3.0, 4.0, 5.0)],
    ///                [HyperDual64::new(3.0, 4.0, 5.0, 6.0), HyperDual64::new(4.0, 5.0, 6.0, 7.0)]]);
    /// let b = arr1(&[HyperDual64::new(5.0, 16.0, 22.0, 64.0), HyperDual64::new(11.0, 32.0, 42.0, 112.0)]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps2, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps2, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1eps2, 5.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<HyperDual<D, f64>>,
    ) -> Result<&'a mut Array1<HyperDual<D, f64>>> {
        let f = self.mapv(|s| s.re);
        let s1 = self.mapv(|s| s.eps1);
        let s2 = self.mapv(|s| s.eps2);
        let s12 = self.mapv(|s| s.eps1eps2);
        let dx0 = f.solve_recursive_into(lu, b.mapv(|b| b.re))?;
        let dx1 = f.solve_recursive_into(lu, b.mapv(|b| b.eps1) - s1.dot(&dx0))?;
        let dx2 = f.solve_recursive_into(lu, b.mapv(|b| b.eps2) - s2.dot(&dx0))?;
        let dx12 =
            f.solve_into(b.mapv(|b| b.eps1eps2) - s1.dot(&dx2) - s2.dot(&dx1) - s12.dot(&dx0))?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&dx2)
            .and(&dx12)
            .and(&mut *b)
            .apply(|&dx0, &dx1, &dx2, &dx12, b| *b = HyperDual::new(dx0, dx1, dx2, dx12));
        Ok(b)
    }
}

impl<D: DualNum<f64> + 'static> SolveDual<HD3<D, f64>> for Array2<HD3<D, f64>>
where
    Array2<D>: SolveDual<D>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::HD3_64;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[HD3_64::new([1.0, 2.0, 3.0, 4.0]), HD3_64::new([2.0, 3.0, 4.0, 5.0])],
    ///                [HD3_64::new([3.0, 4.0, 5.0, 6.0]), HD3_64::new([4.0, 5.0, 6.0, 7.0])]]);
    /// let b = arr1(&[HD3_64::new([5.0, 16.0, 48.0, 136.0]), HD3_64::new([11.0, 32.0, 88.0, 232.0])]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].0[0], 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[1], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[2], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[3], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[1], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[2], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[3], 5.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<HD3<D, f64>>,
    ) -> Result<&'a mut Array1<HD3<D, f64>>> {
        let f = self.mapv(|s| s.0[0]);
        let s1 = self.mapv(|s| s.0[1]);
        let s2 = self.mapv(|s| s.0[2]);
        let s3 = self.mapv(|s| s.0[3]);
        let dx0 = f.solve_recursive_into(lu, b.map(|b| b.0[0]))?;
        let dx1 = f.solve_recursive_into(lu, b.mapv(|b| b.0[1]) - s1.dot(&dx0))?;
        let dx2 =
            f.solve_recursive_into(lu, b.mapv(|b| b.0[2]) - s2.dot(&dx0) - s1.dot(&dx1) * 2.0)?;
        let dx3 = f.solve_recursive_into(
            lu,
            b.mapv(|b| b.0[3]) - s3.dot(&dx0) - s2.dot(&dx1) * 3.0 - s1.dot(&dx2) * 3.0,
        )?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&dx2)
            .and(&dx3)
            .and(&mut *b)
            .apply(|&dx0, &dx1, &dx2, &dx3, b| *b = HD3::new([dx0, dx1, dx2, dx3]));
        Ok(b)
    }
}

pub trait EighDual<A> {
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<A>, Array2<A>)>;
}

impl EighDual<Dual64> for Array2<Dual64> {
    /// Caculates the eigenvalues and eigenvectors of a symmetric matrix
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::Dual64;
    /// # use num_hyperdual::linalg::EighDual;
    /// # use ndarray::{arr1, arr2};
    /// # use ndarray_linalg::UPLO;
    /// let a = arr2(&[[Dual64::new(2.0, 1.0), Dual64::new(2.0, 2.0)],
    ///                [Dual64::new(2.0, 2.0), Dual64::new(5.0, 3.0)]]);
    /// let (l, v) = a.eigh(UPLO::Upper).unwrap();
    /// let av = a.dot(&v);
    /// assert_abs_diff_eq!(av[(0,0)].re, (l[0]*v[(0,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].re, (l[0]*v[(1,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].re, (l[1]*v[(0,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].re, (l[1]*v[(1,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,0)].eps, (l[0]*v[(0,0)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].eps, (l[0]*v[(1,0)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].eps, (l[1]*v[(0,1)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].eps, (l[1]*v[(1,1)]).eps, epsilon = 1e-14);
    /// ```
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<Dual64>, Array2<Dual64>)> {
        let s1 = self.map(|x| x.eps);
        let (l0, v0) = self.map(|x| x.re).eigh(uplo)?;
        let m = v0.t().dot(&s1).dot(&v0);
        let a = Array::from_shape_fn((l0.len(), l0.len()), |(i, j)| {
            if i == j {
                0.0
            } else {
                m[(i, j)] / (l0[i] - l0[j])
            }
        });
        let l = Zip::from(&l0)
            .and(&m.diag())
            .apply_collect(|&l0, &l1| Dual64::new(l0, l1));
        let v = Zip::from(&v0)
            .and(&a.dot(&v0))
            .apply_collect(|&v0, &v1| Dual64::new(v0, v1));
        Ok((l, v))
    }
}
