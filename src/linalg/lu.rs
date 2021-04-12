use super::static_mat::{StaticMat, StaticVec};
use super::LinAlgErr;
use crate::DualNum;
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
            println!("{}", i);
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
