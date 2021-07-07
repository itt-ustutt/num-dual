use super::LinAlgErr;
use crate::{DualNum, StaticMat, StaticVec};
use num_traits::{Float, Zero};
use std::iter::Product;
use std::marker::PhantomData;

pub struct LU<T, F, const N: usize> {
    a: StaticMat<T, N, N>,
    p: [usize; N],
    p_count: usize,
    f: PhantomData<F>,
}

impl<T: DualNum<F>, F: Float, const N: usize> LU<T, F, N> {
    pub fn new(mut a: StaticMat<T, N, N>) -> Result<LU<T, F, N>, LinAlgErr> {
        let tol = F::from(1e-10).unwrap();
        let mut p = [0; N];
        let mut p_count = N;

        for i in 0..N {
            p[i] = i;
        }

        for i in 0..N {
            let mut max_a = F::zero();
            let mut imax = i;

            for k in i..N {
                let abs_a = a[(k, i)].abs();
                if abs_a.re() > max_a {
                    max_a = abs_a.re();
                    imax = k;
                }
            }

            if max_a < tol {
                return Err(LinAlgErr());
            }

            if imax != i {
                let j = p[i];
                p[i] = p[imax];
                p[imax] = j;

                for j in 0..N {
                    let ptr = a[(i, j)];
                    a[(i, j)] = a[(imax, j)];
                    a[(imax, j)] = ptr;
                }

                p_count += 1;
            }

            for j in i + 1..N {
                a[(j, i)] = a[(j, i)] / a[(i, i)];

                for k in i + 1..N {
                    a[(j, k)] = a[(j, k)] - a[(j, i)] * a[(i, k)];
                }
            }
        }
        Ok(LU {
            a,
            p,
            p_count,
            f: PhantomData,
        })
    }

    pub fn solve(&self, b: &StaticVec<T, N>) -> StaticVec<T, N> {
        let mut x = StaticVec::zero();

        for i in 0..N {
            x[i] = b[self.p[i]];

            for k in 0..i {
                x[i] = x[i] - self.a[(i, k)] * x[k];
            }
        }

        for i in (0..N).rev() {
            for k in i + 1..N {
                x[i] = x[i] - self.a[(i, k)] * x[k];
            }

            x[i] = x[i] / self.a[(i, i)];
        }

        x
    }

    pub fn determinant(&self) -> T
    where
        T: Product,
    {
        let det = (0..N).into_iter().map(|i| self.a[(i, i)]).product();

        if (self.p_count - N) % 2 == 0 {
            det
        } else {
            -det
        }
    }

    pub fn inverse(&self) -> StaticMat<T, N, N> {
        let mut ia = StaticMat::zero();

        for j in 0..N {
            for i in 0..N {
                ia[(i, j)] = if self.p[i] == j { T::one() } else { T::zero() };

                for k in 0..i {
                    ia[(i, j)] = ia[(i, j)] - self.a[(i, k)] * ia[(k, j)];
                }
            }

            for i in (0..N).rev() {
                for k in i + 1..N {
                    ia[(i, j)] = ia[(i, j)] - self.a[(i, k)] * ia[(k, j)];
                }
                ia[(i, j)] = ia[(i, j)] / self.a[(i, i)];
            }
        }

        ia
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual64;

    #[test]
    fn test_solve_f64() {
        let a = StaticMat::new([[4.0, 3.0], [6.0, 3.0]]);
        let b = StaticVec::new_vec([10.0, 12.0]);
        let lu = LU::new(a).unwrap();
        assert_eq!(lu.determinant(), -6.0);
        assert_eq!(lu.solve(&b), StaticVec::new_vec([1.0, 2.0]));
        assert_eq!(
            lu.inverse() * lu.determinant(),
            StaticMat::new([[3.0, -3.0], [-6.0, 4.0]])
        );
    }

    #[test]
    fn test_solve_dual64() {
        let a = StaticMat::new([
            [Dual64::new_scalar(4.0, 3.0), Dual64::new_scalar(3.0, 3.0)],
            [Dual64::new_scalar(6.0, 1.0), Dual64::new_scalar(3.0, 2.0)],
        ]);
        let b = StaticVec::new_vec([
            Dual64::new_scalar(10.0, 20.0),
            Dual64::new_scalar(12.0, 20.0),
        ]);
        let lu = LU::new(a).unwrap();
        let det = lu.determinant();
        assert_eq!((det.re, det.eps[0]), (-6.0, -4.0));
        let x = lu.solve(&b);
        assert_eq!(
            (x[0].re, x[0].eps[0], x[1].re, x[1].eps[0]),
            (1.0, 2.0, 2.0, 1.0)
        );
    }
}
