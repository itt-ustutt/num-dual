#![allow(clippy::assign_op_pattern)]
use crate::{Dual, Dual2, Dual3, DualNum, HyperDual};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::fmt;
use std::iter::Product;
use std::marker::PhantomData;

#[cfg(feature = "ndarray-linalg")]
mod linalg_ndarray;
#[cfg(feature = "ndarray-linalg")]
pub use linalg_ndarray::*;

impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual2<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for Dual3<T, F> {}
impl<T: Clone + 'static, F: Clone + 'static> ScalarOperand for HyperDual<T, F> {}

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
    x.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
}

pub fn smallest_ev<T: DualNum<F>, F: Float>(a: Array2<T>) -> (T, Array1<T>) {
    let (e, vecs) = jacobi_eigenvalue(a, 200);
    (e[0], vecs.column(0).to_owned())
}

pub fn jacobi_eigenvalue<T: DualNum<F>, F: Float>(
    mut a: Array2<T>,
    max_iter: usize,
) -> (Array1<T>, Array2<T>) {
    let n = a.shape()[0];

    let mut v = Array2::eye(n);
    let mut d = a.diag().to_owned();

    let mut bw = d.clone();
    let mut zw = Array1::zeros(n);

    for it_num in 0..max_iter {
        let mut thresh = F::zero();
        for j in 0..n {
            for i in 0..j {
                thresh = thresh + a[(i, j)].re().powi(2);
            }
        }
        thresh = thresh.sqrt() / F::from(n).unwrap();

        if thresh.is_zero() {
            break;
        }

        for p in 0..n {
            for q in p + 1..n {
                let gapq = a[(p, q)].abs() * F::from(10.0).unwrap();
                let termp = gapq + d[p].abs();
                let termq = gapq + d[q].abs();

                if 4 < it_num && termp == d[p].abs() && termq == d[q].abs() {
                    a[(p, q)] = T::zero();
                } else if thresh <= a[(p, q)].re().abs() {
                    let h = d[q] - d[p];
                    let term = h.abs() + gapq;

                    let t = if term == h.abs() {
                        a[(p, q)] / h
                    } else {
                        let theta = h * F::from(0.5).unwrap() / a[(p, q)];
                        let mut t = (theta.abs() + (theta * theta + F::one()).sqrt()).recip();
                        if theta.is_negative() {
                            t = -t;
                        }
                        t
                    };

                    let c = (t * t + F::one()).sqrt().recip();
                    let s = t * c;
                    let tau = s / (c + F::one());
                    let h = t * a[(p, q)];

                    zw[p] -= h;
                    zw[q] += h;
                    d[p] -= h;
                    d[q] += h;

                    a[(p, q)] = T::zero();

                    for j in 0..p {
                        let g = a[(j, p)];
                        let h = a[(j, q)];
                        a[(j, p)] = g - s * (h + g * tau);
                        a[(j, q)] = h + s * (g - h * tau);
                    }

                    for j in p + 1..q {
                        let g = a[(p, j)];
                        let h = a[(j, q)];
                        a[(p, j)] = g - s * (h + g * tau);
                        a[(j, q)] = h + s * (g - h * tau);
                    }

                    for j in q + 1..n {
                        let g = a[(p, j)];
                        let h = a[(q, j)];
                        a[(p, j)] = g - s * (h + g * tau);
                        a[(q, j)] = h + s * (g - h * tau);
                    }

                    for j in 0..n {
                        let g = v[(j, p)];
                        let h = v[(j, q)];
                        v[(j, p)] = g - s * (h + g * tau);
                        v[(j, q)] = h + s * (g - h * tau);
                    }
                }
            }
        }

        bw += &zw;
        d.assign(&bw);
        zw.fill(T::zero());
    }

    for k in 0..n - 1 {
        let mut m = k;

        for l in k + 1..n {
            if d[l].re() < d[m].re() {
                m = l;
            }
        }

        if m != k {
            d.swap(m, k);

            for l in 0..n {
                v.swap((l, m), (l, k));
            }
        }
    }

    (d, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dual64;
    use approx::assert_abs_diff_eq;
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

    #[test]
    fn test_eig_f64_2() {
        let a = arr2(&[[2.0, 2.0], [2.0, 5.0]]);
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let av = a.dot(&v);
        println!("{} {}", l, v);
        assert_abs_diff_eq!(av[(0, 0)], (l[0] * v[(0, 0)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 0)], (l[0] * v[(1, 0)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 1)], (l[1] * v[(0, 1)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 1)], (l[1] * v[(1, 1)]), epsilon = 1e-14);
    }

    #[test]
    fn test_eig_f64_3() {
        let a = arr2(&[[2.0, 2.0, 7.0], [2.0, 5.0, 9.0], [7.0, 9.0, 2.0]]);
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let av = a.dot(&v);
        println!("{} {}", l, v);
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(av[(i, j)], (l[j] * v[(i, j)]), epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_eig_dual64() {
        let a = arr2(&[
            [Dual64::new_scalar(2.0, 1.0), Dual64::new_scalar(2.0, 2.0)],
            [Dual64::new_scalar(2.0, 2.0), Dual64::new_scalar(5.0, 3.0)],
        ]);
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let av = a.dot(&v);
        println!("{} {}", l, v);
        assert_abs_diff_eq!(av[(0, 0)].re, (l[0] * v[(0, 0)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 0)].re, (l[0] * v[(1, 0)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 1)].re, (l[1] * v[(0, 1)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 1)].re, (l[1] * v[(1, 1)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(
            av[(0, 0)].eps[0],
            (l[0] * v[(0, 0)]).eps[0],
            epsilon = 1e-14
        );
        assert_abs_diff_eq!(
            av[(1, 0)].eps[0],
            (l[0] * v[(1, 0)]).eps[0],
            epsilon = 1e-14
        );
        assert_abs_diff_eq!(
            av[(0, 1)].eps[0],
            (l[1] * v[(0, 1)]).eps[0],
            epsilon = 1e-14
        );
        assert_abs_diff_eq!(
            av[(1, 1)].eps[0],
            (l[1] * v[(1, 1)]).eps[0],
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_norm_f64() {
        let v = arr1(&[3.0, 4.0]);
        assert_eq!(norm(&v), 5.0);
    }

    #[test]
    fn test_norm_dual64() {
        let v = arr1(&[Dual64::new_scalar(3.0, 1.0), Dual64::new_scalar(4.0, 3.0)]);
        println!("{}", norm(&v));
        assert_eq!(norm(&v).re, 5.0);
        assert_eq!(norm(&v).eps[0], 3.0);
    }
}
