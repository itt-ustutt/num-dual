#![allow(clippy::assign_op_pattern)]
use crate::DualNum;
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt;
use std::iter::Product;
use std::marker::PhantomData;

#[cfg(feature = "ndarray-linalg")]
mod linalg_ndarray;
#[cfg(feature = "ndarray-linalg")]
pub use linalg_ndarray::*;

#[derive(Debug)]
pub struct LinAlgError();

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

impl std::error::Error for LinAlgError {}

pub struct LU<T, F> {
    a: Array2<T>,
    p: Array1<usize>,
    p_count: usize,
    f: PhantomData<F>,
}

impl<T: DualNum<F>, F: Float> LU<T, F> {
    pub fn new(mut a: Array2<T>) -> Result<Self, LinAlgError> {
        let tol = F::from(1e-10).unwrap();
        let n = a.shape()[0];
        let mut p = Array1::zeros(n);
        let mut p_count = n;

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

            if max_a < tol {
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

                p_count += 1;
            }

            for j in i + 1..n {
                a[(j, i)] = a[(j, i)] / a[(i, i)];

                for k in i + 1..n {
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

    pub fn solve(&self, b: &Array1<T>) -> Array1<T> {
        let n = b.len();
        let mut x = Array1::zeros(n);

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

            x[i] = x[i] / self.a[(i, i)];
        }

        x
    }

    pub fn determinant(&self) -> T
    where
        T: Product,
    {
        let n = self.p.len();
        let det = (0..n).into_iter().map(|i| self.a[(i, i)]).product();

        if (self.p_count - n) % 2 == 0 {
            det
        } else {
            -det
        }
    }

    pub fn inverse(&self) -> Array2<T> {
        let n = self.p.len();
        let mut ia = Array2::zeros((n, n));

        for j in 0..n {
            for i in 0..n {
                ia[(i, j)] = if self.p[i] == j { T::one() } else { T::zero() };

                for k in 0..i {
                    ia[(i, j)] = ia[(i, j)] - self.a[(i, k)] * ia[(k, j)];
                }
            }

            for i in (0..n).rev() {
                for k in i + 1..n {
                    ia[(i, j)] = ia[(i, j)] - self.a[(i, k)] * ia[(k, j)];
                }
                ia[(i, j)] = ia[(i, j)] / self.a[(i, i)];
            }
        }

        ia
    }
}

pub fn norm<T: DualNum<F>, F: Float>(x: &Array1<T>) -> T {
    x.iter().fold(T::zero(), |acc, &x| acc + x)
}

pub fn smallest_ev<T: DualNum<F>, F: Float>(a: &Array2<T>) -> Result<(T, Array1<T>), LinAlgError> {
    unimplemented!()
    // let (e, vecs) = a.eigh(UPLO::Upper)?;
    // let i_min = e.map(Dual64::re).argmin().unwrap();
    // Ok((e[i_min], vecs.row(i_min).to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual64;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_solve_f64() {
        let a = arr2(&[[4.0, 3.0], [6.0, 3.0]]);
        let b = arr1(&[10.0, 12.0]);
        let lu = LU::new(a).unwrap();
        assert_eq!(lu.determinant(), -6.0);
        assert_eq!(lu.solve(&b), arr1(&[1.0, 2.0]));
        assert_eq!(
            lu.inverse() * lu.determinant(),
            arr2(&[[3.0, -3.0], [-6.0, 4.0]])
        );
    }

    #[test]
    fn test_solve_dual64() {
        let a = arr2(&[
            [Dual64::new_scalar(4.0, 3.0), Dual64::new_scalar(3.0, 3.0)],
            [Dual64::new_scalar(6.0, 1.0), Dual64::new_scalar(3.0, 2.0)],
        ]);
        let b = arr1(&[
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
