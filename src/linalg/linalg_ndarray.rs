use crate::*;
use ndarray::*;
use ndarray_linalg::convert::replicate;
use ndarray_linalg::error::Result;
use ndarray_linalg::*;

impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for HyperDual<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual3<T, F> {}

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
    /// # use num_dual::linalg::SolveDual;
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
    /// # use num_dual::Dual64;
    /// # use num_dual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[Dual64::new_scalar(1.0, 2.0), Dual64::new_scalar(3.0, 4.0)],
    ///                [Dual64::new_scalar(5.0, 6.0), Dual64::new_scalar(7.0, 8.0)]]);
    /// let b = arr1(&[Dual64::new_scalar(10.0, 28.0), Dual64::new_scalar(26.0, 68.0)]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps[0], 4.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<Dual<D, f64>>,
    ) -> Result<&'a mut Array1<Dual<D, f64>>> {
        let f = self.mapv(|s| s.re);
        let dx0 = f.solve_recursive_into(lu, b.mapv(|b| b.re))?;
        let dx1 = f.solve_recursive_into(
            lu,
            b.mapv(|b| b.eps[0]) - &self.mapv(|s| s.eps[0]).dot(&dx0),
        )?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&mut *b)
            .for_each(|&dx0, &dx1, b| *b = Dual::new_scalar(dx0, dx1));
        Ok(b)
    }
}

impl<D: DualNum<f64> + 'static> SolveDual<DualVec<D, f64, 2>> for Array2<DualVec<D, f64, 2>>
where
    Array2<D>: SolveDual<D>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_dual::{DualVec64, StaticVec};
    /// # use num_dual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[DualVec64::new(1.0, StaticVec::new_vec([2.0, 1.0])), DualVec64::new(3.0, StaticVec::new_vec([4.0, 1.0]))],
    ///                [DualVec64::new(5.0, StaticVec::new_vec([6.0, 1.0])), DualVec64::new(7.0, StaticVec::new_vec([8.0, 1.0]))]]);
    /// let b = arr1(&[DualVec64::new(10.0, StaticVec::new_vec([28.0, 18.0])), DualVec64::new(26.0, StaticVec::new_vec([68.0, 42.0]))]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps[1], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps[0], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps[1], 4.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<DualVec<D, f64, 2>>,
    ) -> Result<&'a mut Array1<DualVec<D, f64, 2>>> {
        let f = self.mapv(|s| s.re);
        let dx0 = f.solve_recursive_into(lu, b.mapv(|b| b.re))?;
        let dx1_0 = f.solve_recursive_into(
            lu,
            b.mapv(|b| b.eps[0]) - &self.mapv(|s| s.eps[0]).dot(&dx0),
        )?;
        let dx1_1 = f.solve_recursive_into(
            lu,
            b.mapv(|b| b.eps[1]) - &self.mapv(|s| s.eps[1]).dot(&dx0),
        )?;
        Zip::from(&dx0)
            .and(&dx1_0)
            .and(&dx1_1)
            .and(&mut *b)
            .for_each(|&dx0, &dx1_0, &dx1_1, b| {
                *b = DualVec::new(dx0, StaticVec::new_vec([dx1_0, dx1_1]))
            });
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
    /// # use num_dual::HyperDual64;
    /// # use num_dual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[HyperDual64::new_scalar(1.0, 2.0, 3.0, 4.0), HyperDual64::new_scalar(2.0, 3.0, 4.0, 5.0)],
    ///                [HyperDual64::new_scalar(3.0, 4.0, 5.0, 6.0), HyperDual64::new_scalar(4.0, 5.0, 6.0, 7.0)]]);
    /// let b = arr1(&[HyperDual64::new_scalar(5.0, 16.0, 22.0, 64.0), HyperDual64::new_scalar(11.0, 32.0, 42.0, 112.0)]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps2[0], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2[(0,0)], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1[0], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps2[0], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1eps2[(0,0)], 5.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<HyperDual<D, f64>>,
    ) -> Result<&'a mut Array1<HyperDual<D, f64>>> {
        let f = self.mapv(|s| s.re);
        let s1 = self.mapv(|s| s.eps1[0]);
        let s2 = self.mapv(|s| s.eps2[0]);
        let s12 = self.mapv(|s| s.eps1eps2[(0, 0)]);
        let dx0 = f.solve_recursive_into(lu, b.mapv(|b| b.re))?;
        let dx1 = f.solve_recursive_into(lu, b.mapv(|b| b.eps1[0]) - s1.dot(&dx0))?;
        let dx2 = f.solve_recursive_into(lu, b.mapv(|b| b.eps2[0]) - s2.dot(&dx0))?;
        let dx12 = f.solve_into(
            b.mapv(|b| b.eps1eps2[(0, 0)]) - s1.dot(&dx2) - s2.dot(&dx1) - s12.dot(&dx0),
        )?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&dx2)
            .and(&dx12)
            .and(&mut *b)
            .for_each(|&dx0, &dx1, &dx2, &dx12, b| *b = HyperDual::new_scalar(dx0, dx1, dx2, dx12));
        Ok(b)
    }
}

impl<D: DualNum<f64> + 'static> SolveDual<Dual3<D, f64>> for Array2<Dual3<D, f64>>
where
    Array2<D>: SolveDual<D>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_dual::Dual3_64;
    /// # use num_dual::linalg::SolveDual;
    /// # use ndarray::{arr1, arr2};
    /// let a = arr2(&[[Dual3_64::new(1.0, 2.0, 3.0, 4.0), Dual3_64::new(2.0, 3.0, 4.0, 5.0)],
    ///                [Dual3_64::new(3.0, 4.0, 5.0, 6.0), Dual3_64::new(4.0, 5.0, 6.0, 7.0)]]);
    /// let b = arr1(&[Dual3_64::new(5.0, 16.0, 48.0, 136.0), Dual3_64::new(11.0, 32.0, 88.0, 232.0)]);
    /// let x = a.solve_into(b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].v1, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].v2, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].v3, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].v1, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].v2, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].v3, 5.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive_inplace<'a>(
        &self,
        lu: &LU64,
        b: &'a mut Array1<Dual3<D, f64>>,
    ) -> Result<&'a mut Array1<Dual3<D, f64>>> {
        let f = self.mapv(|s| s.re);
        let s1 = self.mapv(|s| s.v1);
        let s2 = self.mapv(|s| s.v2);
        let s3 = self.mapv(|s| s.v3);
        let dx0 = f.solve_recursive_into(lu, b.map(|b| b.re))?;
        let dx1 = f.solve_recursive_into(lu, b.mapv(|b| b.v1) - s1.dot(&dx0))?;
        let dx2 =
            f.solve_recursive_into(lu, b.mapv(|b| b.v2) - s2.dot(&dx0) - s1.dot(&dx1) * 2.0)?;
        let dx3 = f.solve_recursive_into(
            lu,
            b.mapv(|b| b.v3) - s3.dot(&dx0) - s2.dot(&dx1) * 3.0 - s1.dot(&dx2) * 3.0,
        )?;
        Zip::from(&dx0)
            .and(&dx1)
            .and(&dx2)
            .and(&dx3)
            .and(&mut *b)
            .for_each(|&dx0, &dx1, &dx2, &dx3, b| *b = Dual3::new(dx0, dx1, dx2, dx3));
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
    /// # use num_dual::Dual64;
    /// # use num_dual::linalg::EighDual;
    /// # use ndarray::{arr1, arr2};
    /// # use ndarray_linalg::UPLO;
    /// let a = arr2(&[[Dual64::new_scalar(2.0, 1.0), Dual64::new_scalar(2.0, 2.0)],
    ///                [Dual64::new_scalar(2.0, 2.0), Dual64::new_scalar(5.0, 3.0)]]);
    /// let (l, v) = a.eigh(UPLO::Upper).unwrap();
    /// let av = a.dot(&v);
    /// assert_abs_diff_eq!(av[(0,0)].re, (l[0]*v[(0,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].re, (l[0]*v[(1,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].re, (l[1]*v[(0,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].re, (l[1]*v[(1,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,0)].eps[0], (l[0]*v[(0,0)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].eps[0], (l[0]*v[(1,0)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].eps[0], (l[1]*v[(0,1)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].eps[0], (l[1]*v[(1,1)]).eps[0], epsilon = 1e-14);
    /// ```
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<Dual64>, Array2<Dual64>)> {
        let s1 = self.map(|x| x.eps[0]);
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
            .map_collect(|&l0, &l1| Dual64::new_scalar(l0, l1));
        let v = Zip::from(&v0)
            .and(&a.dot(&v0))
            .map_collect(|&v0, &v1| Dual64::new_scalar(v0, v1));
        Ok((l, v))
    }
}

impl EighDual<DualVec64<2>> for Array2<DualVec64<2>> {
    /// Caculates the eigenvalues and eigenvectors of a symmetric matrix
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_dual::{DualVec64, StaticVec};
    /// # use num_dual::linalg::EighDual;
    /// # use ndarray::{arr1, arr2};
    /// # use ndarray_linalg::UPLO;
    /// let a = arr2(&[[DualVec64::new(2.0, StaticVec::new_vec([1.0, 1.0])), DualVec64::new(2.0, StaticVec::new_vec([2.0, 1.0]))],
    ///                [DualVec64::new(2.0, StaticVec::new_vec([2.0, 1.0])), DualVec64::new(5.0, StaticVec::new_vec([3.0, 1.0]))]]);
    /// let (l, v) = a.eigh(UPLO::Upper).unwrap();
    /// let av = a.dot(&v);
    /// assert_abs_diff_eq!(av[(0,0)].re, (l[0]*v[(0,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].re, (l[0]*v[(1,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].re, (l[1]*v[(0,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].re, (l[1]*v[(1,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,0)].eps[0], (l[0]*v[(0,0)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].eps[0], (l[0]*v[(1,0)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].eps[0], (l[1]*v[(0,1)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].eps[0], (l[1]*v[(1,1)]).eps[0], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,0)].eps[1], (l[0]*v[(0,0)]).eps[1], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].eps[1], (l[0]*v[(1,0)]).eps[1], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].eps[1], (l[1]*v[(0,1)]).eps[1], epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].eps[1], (l[1]*v[(1,1)]).eps[1], epsilon = 1e-14);
    /// ```
    fn eigh(&self, uplo: UPLO) -> Result<(Array1<DualVec64<2>>, Array2<DualVec64<2>>)> {
        let s1_0 = self.map(|x| x.eps[0]);
        let s1_1 = self.map(|x| x.eps[1]);
        let (l0, v0) = self.map(|x| x.re).eigh(uplo)?;
        let m_0 = v0.t().dot(&s1_0).dot(&v0);
        let m_1 = v0.t().dot(&s1_1).dot(&v0);
        let a_0 = Array::from_shape_fn((l0.len(), l0.len()), |(i, j)| {
            if i == j {
                0.0
            } else {
                m_0[(i, j)] / (l0[i] - l0[j])
            }
        });
        let a_1 = Array::from_shape_fn((l0.len(), l0.len()), |(i, j)| {
            if i == j {
                0.0
            } else {
                m_1[(i, j)] / (l0[i] - l0[j])
            }
        });
        let l = Zip::from(&l0)
            .and(&m_0.diag())
            .and(&m_1.diag())
            .map_collect(|&l0, &l1_0, &l1_1| DualVec64::new(l0, StaticVec::new_vec([l1_0, l1_1])));
        let v = Zip::from(&v0)
            .and(&a_0.dot(&v0))
            .and(&a_1.dot(&v0))
            .map_collect(|&v0, &v1_0, &v1_1| DualVec64::new(v0, StaticVec::new_vec([v1_0, v1_1])));
        Ok((l, v))
    }
}
