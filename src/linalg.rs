use crate::dual::Dual;
use crate::hd3::HD3;
use crate::hyperdual::HyperDual;
use crate::static_vec::{OuterProduct, StaticMat, StaticVec};
use crate::{DualNum, DualVec};
use ndarray::Array2;
use ndarray_linalg::error::Result;
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::{Eigh, UPLO};
use num_traits::Zero;
use std::ops::AddAssign;

// impl<T0: Clone + 'static, T1: Clone + 'static, F: Clone + 'static> ScalarOperand
//     for Dual<T0, T1, F>
// {
// }
// impl<T0: Clone + 'static, T1: Clone + 'static, T2: Clone + 'static, F: Clone + 'static>
//     ScalarOperand for HyperDual<T0, T1, T2, F>
// {
// }
// impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for HD3<T, F> {}

impl<T: DualNum<f64>, const N: usize> InverseDual<N> for StaticMat<T, N, N> {
    fn re_inv(&self) -> Result<StaticMat<f64, N, N>> {
        Array2::from_shape_fn((N, N), |i| self[i].re())
            .inv()
            .map(StaticMat::from_ndarray)
    }
}

pub trait InverseDual<const N: usize> {
    fn re_inv(&self) -> Result<StaticMat<f64, N, N>>;
}

pub trait SolveDual<T, const N: usize>: InverseDual<N> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    fn solve(&self, b: &StaticVec<T, N>) -> Result<StaticVec<T, N>> {
        Ok(self.solve_recursive(&self.re_inv()?, b))
    }
    // /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    // /// is the argument, and `x` is the successful result.
    // fn solve_into<T1>(&self, mut b: StaticVec<T1, N>) -> Result<StaticVec<T1, N>> {
    //     self.solve_inplace(&mut b)?;
    //     Ok(b)
    // }
    // /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    // /// is the argument, and `x` is the successful result.
    // fn solve_inplace<'a, T1>(
    //     &self,
    //     b: &'a mut StaticVec<T1, N>,
    // ) -> Result<&'a mut StaticVec<T1, N>> {
    //     let inv = self.re_inv()?;
    //     Ok(self.solve_recursive_inplace(&inv, b))
    // }

    fn solve_recursive(&self, inv: &StaticMat<f64, N, N>, b: &StaticVec<T, N>) -> StaticVec<T, N>;

    // fn solve_recursive_inplace<'a, T1>(
    //     &self,
    //     inv: &StaticMat<f64, N, N>,
    //     b: &'a mut StaticVec<T1, N>,
    // ) -> &'a mut StaticVec<T1, N>;
}

impl<T: DualVec<f64, f64>, const N: usize> SolveDual<T, N> for StaticMat<f64, N, N> {
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use num_hyperdual::static_vec::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[1.0, 3.0],
    ///                         [5.0, 7.0]]);
    /// let b = StaticVec::new([10.0, 26.0]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive(&self, inv: &StaticMat<f64, N, N>, b: &StaticVec<T, N>) -> StaticVec<T, N> {
        b.dot(inv)
    }
}

impl<T0: DualNum<f64>, T1: DualVec<T0, f64>, const N: usize> SolveDual<Dual<T0, T1, f64>, N>
    for StaticMat<Dual<T0, T1, f64>, N, N>
where
    StaticMat<T0, N, N>: SolveDual<T0, N> + SolveDual<T1, N>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::{Dual64, DualN64};
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use num_hyperdual::static_vec::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[Dual64::new(1.0, 2.0), Dual64::new(3.0, 4.0)],
    ///                         [Dual64::new(5.0, 6.0), Dual64::new(7.0, 8.0)]]);
    /// let b = StaticVec::new([Dual64::new(10.0, 28.0), Dual64::new(26.0, 68.0)]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps, 4.0, epsilon = 1e-14);
    ///
    /// let a = StaticMat::new([[DualN64::new(1.0, StaticVec::new([2.0, 3.0]))]]);
    /// let b = StaticVec::new([DualN64::new(4.0, StaticVec::new([13.0, 18.0]))]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps[0], 5.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps[1], 6.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive(
        &self,
        inv: &StaticMat<f64, N, N>,
        b: &StaticVec<Dual<T0, T1, f64>, N>,
    ) -> StaticVec<Dual<T0, T1, f64>, N> {
        let f = self.map(|s| s.re);
        let dx0 = f.solve_recursive(inv, &b.map(|b| b.re));
        let dx1 = f.solve_recursive(inv, &(b.map(|b| b.eps) - self.map(|s| s.eps).dot(&dx0)));
        let mut res = [Dual::zero(); N];
        for i in 0..N {
            res[i] = Dual::new(dx0[i], dx1[i]);
        }
        StaticVec::new(res)
    }
}

impl<
        T0: DualNum<f64>,
        T1: DualVec<T0, f64> + OuterProduct<Output = T2>,
        T2: DualVec<T0, f64>,
        const N: usize,
    > SolveDual<HyperDual<T0, T1, T2, f64>, N> for StaticMat<HyperDual<T0, T1, T2, f64>, N, N>
where
    StaticMat<T0, N, N>: SolveDual<T0, N> + SolveDual<T1, N> + SolveDual<T2, N>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::{HyperDual64, HyperDualN64};
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use num_hyperdual::static_vec::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[HyperDual64::new(1.0, 2.0, 3.0, 4.0), HyperDual64::new(2.0, 3.0, 4.0, 5.0)],
    ///                         [HyperDual64::new(3.0, 4.0, 5.0, 6.0), HyperDual64::new(4.0, 5.0, 6.0, 7.0)]]);
    /// let b = StaticVec::new([HyperDual64::new(5.0, 16.0, 22.0, 64.0), HyperDual64::new(11.0, 32.0, 42.0, 112.0)]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps2, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].re, 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps2, 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].eps1eps2, 5.0, epsilon = 1e-14);
    ///
    /// let a = StaticMat::new([[HyperDualN64::new(1.0, StaticVec::new([2.0, 3.0]), StaticVec::new([4.0, 5.0]), StaticMat::new([[4.0, 5.0], [5.0, 6.0]]))]]);
    /// let b = StaticVec::new([HyperDualN64::new(3.0, StaticVec::new([10.0, 14.0]), StaticVec::new([14.0, 18.0]), StaticMat::new([[37.0, 47.0], [47.0, 59.0]]))]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0].re, 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1[0], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1[1], 5.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps2[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps2[1], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2[(0,0)], 5.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2[(0,1)], 6.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2[(1,0)], 6.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].eps1eps2[(1,1)], 7.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive(
        &self,
        inv: &StaticMat<f64, N, N>,
        b: &StaticVec<HyperDual<T0, T1, T2, f64>, N>,
    ) -> StaticVec<HyperDual<T0, T1, T2, f64>, N> {
        let f = self.map(|s| s.re);
        let s1 = self.map(|s| s.eps1);
        let s2 = self.map(|s| s.eps2);
        let s12 = self.map(|s| s.eps1eps2);
        let dx0 = f.solve_recursive(inv, &b.map(|b| b.re));
        let dx1 = f.solve_recursive(inv, &(b.map(|b| b.eps1) - s1.dot(&dx0)));
        let dx2 = f.solve_recursive(inv, &(b.map(|b| b.eps2) - s2.dot(&dx0)));
        let dx12 = f.solve_recursive(
            inv,
            &(b.map(|b| b.eps1eps2) - s1.dot_outer(&dx2) - dx1.dot_outer(&s2) - s12.dot(&dx0)),
        );
        let mut res = [HyperDual::zero(); N];
        for i in 0..N {
            res[i] = HyperDual::new(dx0[i], dx1[i], dx2[i], dx12[i]);
        }
        StaticVec::new(res)
    }
}

impl<D: DualNum<f64>, const N: usize> SolveDual<HD3<D, f64>, N> for StaticMat<HD3<D, f64>, N, N>
where
    StaticMat<D, N, N>: SolveDual<D, N>,
{
    /// Solves a system of linear equations `A * x = b` where `A` is `self`, `b`
    /// is the argument, and `x` is the successful result.
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::HD3_64;
    /// # use num_hyperdual::linalg::SolveDual;
    /// # use num_hyperdual::static_vec::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[HD3_64::new([1.0, 2.0, 3.0, 4.0]), HD3_64::new([2.0, 3.0, 4.0, 5.0])],
    ///                         [HD3_64::new([3.0, 4.0, 5.0, 6.0]), HD3_64::new([4.0, 5.0, 6.0, 7.0])]]);
    /// let b = StaticVec::new([HD3_64::new([5.0, 16.0, 48.0, 136.0]), HD3_64::new([11.0, 32.0, 88.0, 232.0])]);
    /// let x = a.solve(&b).unwrap();
    /// assert_abs_diff_eq!(x[0].0[0], 1.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[1], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[2], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[0].0[3], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[0], 2.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[1], 3.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[2], 4.0, epsilon = 1e-14);
    /// assert_abs_diff_eq!(x[1].0[3], 5.0, epsilon = 1e-14);
    /// ```
    fn solve_recursive(
        &self,
        inv: &StaticMat<f64, N, N>,
        b: &StaticVec<HD3<D, f64>, N>,
    ) -> StaticVec<HD3<D, f64>, N> {
        let f = self.map(|s| s.0[0]);
        let s1 = self.map(|s| s.0[1]);
        let s2 = self.map(|s| s.0[2]);
        let s3 = self.map(|s| s.0[3]);
        let dx0 = f.solve_recursive(inv, &b.map(|b| b.0[0]));
        let dx1 = f.solve_recursive(inv, &(b.map(|b| b.0[1]) - s1.dot(&dx0)));
        let dx2 = f.solve_recursive(
            inv,
            &(b.map(|b| b.0[2]) - s2.dot(&dx0) - s1.dot(&dx1) * D::from(2.0)),
        );
        let dx3 = f.solve_recursive(
            inv,
            &(b.map(|b| b.0[3])
                - s3.dot(&dx0)
                - s2.dot(&dx1) * D::from(3.0)
                - s1.dot(&dx2) * D::from(3.0)),
        );
        let mut res = [HD3::zero(); N];
        for i in 0..N {
            res[i] = HD3::new([dx0[i], dx1[i], dx2[i], dx3[i]]);
        }
        StaticVec::new(res)
    }
}

impl<T1: DualVec<f64, f64>, const N: usize> StaticMat<Dual<f64, T1, f64>, N, N> {
    /// Caculates the eigenvalues and eigenvectors of a symmetric matrix
    /// ```
    /// # use approx::assert_abs_diff_eq;
    /// # use num_hyperdual::Dual;
    /// # use num_hyperdual::static_vec::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[Dual::new(2.0, 1.0), Dual::new(2.0, 2.0)],
    ///                         [Dual::new(2.0, 2.0), Dual::new(5.0, 5.0)]]);
    /// let (l, v) = a.eigh().unwrap();
    /// let av = a.matmul(&v);
    /// assert_abs_diff_eq!(av[(0,0)].re, (l[0]*v[(0,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].re, (l[0]*v[(1,0)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].re, (l[1]*v[(0,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].re, (l[1]*v[(1,1)]).re, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,0)].eps, (l[0]*v[(0,0)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,0)].eps, (l[0]*v[(1,0)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(0,1)].eps, (l[1]*v[(0,1)]).eps, epsilon = 1e-14);
    /// assert_abs_diff_eq!(av[(1,1)].eps, (l[1]*v[(1,1)]).eps, epsilon = 1e-14);
    ///
    /// let a = StaticMat::new([[Dual::new(2.0, StaticVec::new([1.0, 2.0])), Dual::new(2.0, StaticVec::new([2.0, 3.0]))],
    ///                         [Dual::new(2.0, StaticVec::new([2.0, 3.0])), Dual::new(5.0, StaticVec::new([5.0, 1.0]))]]);
    /// let (l, v) = a.eigh().unwrap();
    /// let av = a.matmul(&v);
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
    pub fn eigh(&self) -> Result<(StaticVec<Dual<f64, T1, f64>, N>, Self)> {
        let s1 = self.map(|x| x.eps);
        let arr = Array2::from_shape_fn((N, N), |i| self[i].re);
        let (l0, v0) = arr.eigh(UPLO::Upper)?;
        let v0: StaticMat<_, N, N> = StaticMat::from_ndarray(v0);
        let m = s1.t().matmul(&v0).t().matmul(&v0);
        let mut a: StaticMat<_, N, N> = StaticMat::zero();
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    a[(i, j)] = m[(i, j)] / (l0[i] - l0[j]);
                }
            }
        }
        let mut l = StaticVec::zero();
        for i in 0..N {
            l[i] = Dual::new(l0[i], m[(i, i)]);
        }
        let v1 = a.matmul(&v0);
        let mut v = StaticMat::zero();
        for i in 0..N {
            for j in 0..N {
                v[(i, j)] = Dual::new(v0[(i, j)], v1[(i, j)]);
            }
        }
        Ok((l, v))
    }
}

impl<T: Copy + Zero + AddAssign, const M: usize, const N: usize> StaticMat<T, M, N> {
    fn from_ndarray(arr: Array2<T>) -> Self {
        let mut res = StaticMat::zero();
        for i in 0..M {
            for j in 0..N {
                res[(i, j)] = arr[(i, j)]
            }
        }
        res
    }
}
