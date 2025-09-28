//! Basic linear algebra functionalities (linear solve and eigenvalues) for matrices containing dual numbers.
use crate::DualNum;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OMatrix, OVector, U1};
use num_traits::Float;
use std::fmt;
use std::iter::Product;
use std::marker::PhantomData;

/// Error type for fallible linear algebra operations.
#[derive(Debug)]
pub struct LinAlgError();

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The matrix appears to be singular.")
    }
}

impl std::error::Error for LinAlgError {}

/// LU decomposition for symmetric matrices with dual numbers as elements.
pub struct LU<T: DualNum<F>, F, D: Dim>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    a: OMatrix<T, D, D>,
    p: OVector<usize, D>,
    p_count: usize,
    f: PhantomData<F>,
}

impl<T: DualNum<F> + Copy, F: Float, D: Dim> LU<T, F, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    pub fn new(mut a: OMatrix<T, D, D>) -> Result<Self, LinAlgError> {
        let (n, _) = a.shape_generic();
        let mut p = OVector::zeros_generic(n, U1);
        let n = n.value();
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

    pub fn solve(&self, b: &OVector<T, D>) -> OVector<T, D> {
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

    pub fn determinant(&self) -> T
    where
        T: Product,
    {
        let n = self.p.len();
        let det = (0..n).map(|i| self.a[(i, i)]).product();

        if (self.p_count - n).is_multiple_of(2) {
            det
        } else {
            -det
        }
    }

    pub fn inverse(&self) -> OMatrix<T, D, D> {
        let (r, c) = self.a.shape_generic();
        let n = self.p.len();
        let mut ia = OMatrix::zeros_generic(r, c);

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
                ia[(i, j)] /= self.a[(i, i)];
            }
        }

        ia
    }
}

/// Smallest eigenvalue and corresponding eigenvector calculated using the full Jacobi
/// eigenvalue algorithm ([`jacobi_eigenvalue`]).
pub fn smallest_ev<T: DualNum<F> + Copy, F: Float, D: Dim>(
    a: OMatrix<T, D, D>,
) -> (T, OVector<T, D>)
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    let (r, _) = a.shape_generic();
    let n = r.value();
    if n == 1 {
        (a[(0, 0)], OVector::from_element_generic(r, U1, T::one()))
    } else if n == 2 {
        let (a, b, c) = (a[(0, 0)], a[(0, 1)], a[(1, 1)]);
        let l = (a + c - ((a - c).powi(2) + b * b * F::from(4.0).unwrap()).sqrt())
            * F::from(0.5).unwrap();
        let u = OVector::from_fn_generic(r, U1, |i, _| [b, l - a][i]);
        let u = u / (b * b + (l - a) * (l - a)).sqrt();
        (l, u)
    } else {
        let (e, vecs) = jacobi_eigenvalue(a, 200);
        (e[0], vecs.column(0).into_owned())
    }
}

/// Eigenvalues and corresponding eigenvectors of a symmetric matrix.
pub fn jacobi_eigenvalue<T: DualNum<F> + Copy, F: Float, D: Dim>(
    mut a: OMatrix<T, D, D>,
    max_iter: usize,
) -> (OVector<T, D>, OMatrix<T, D, D>)
where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    let (r, c) = a.shape_generic();
    let n = r.value();

    let mut v = OMatrix::identity_generic(r, c);
    let mut d = a.diagonal().to_owned();

    let mut bw = d.clone();
    let mut zw = OVector::zeros_generic(r, U1);

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
        d = bw.clone();
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
            d.swap_rows(m, k);

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
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_solve_f64() {
        let a = dmatrix![4.0, 3.0; 6.0, 3.0];
        let b = dvector![10.0, 12.0];
        let lu = LU::new(a).unwrap();
        assert_eq!(lu.determinant(), -6.0);
        assert_eq!(lu.solve(&b), dvector![1.0, 2.0]);
        assert_eq!(
            lu.inverse() * lu.determinant(),
            dmatrix![3.0, -3.0; -6.0, 4.0]
        );
    }

    #[test]
    fn test_solve_dual64() {
        let a = dmatrix![
            Dual64::new(4.0, 3.0), Dual64::new(3.0, 3.0);
            Dual64::new(6.0, 1.0), Dual64::new(3.0, 2.0)
        ];
        let b = dvector![Dual64::new(10.0, 20.0), Dual64::new(12.0, 20.0)];
        let lu = LU::new(a).unwrap();
        let det = lu.determinant();
        assert_eq!((det.re, det.eps), (-6.0, -4.0));
        let x = lu.solve(&b);
        assert_eq!((x[0].re, x[0].eps, x[1].re, x[1].eps), (1.0, 2.0, 2.0, 1.0));
    }

    #[test]
    fn test_eig_f64_2() {
        let a = dmatrix![2.0, 2.0; 2.0, 5.0];
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let (l1, v1) = smallest_ev(a.clone());
        let av = a * &v;
        println!("{l} {v}");
        println!("{l1} {v1}");
        assert_abs_diff_eq!(av[(0, 0)], (l[0] * v[(0, 0)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 0)], (l[0] * v[(1, 0)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 1)], (l[1] * v[(0, 1)]), epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 1)], (l[1] * v[(1, 1)]), epsilon = 1e-14);
        assert_abs_diff_eq!(l[0], l1, epsilon = 1e-14);
        assert_abs_diff_eq!(v[(0, 0)], v1[0], epsilon = 1e-14);
        assert_abs_diff_eq!(v[(1, 0)], v1[1], epsilon = 1e-14);
    }

    #[test]
    fn test_eig_f64_3() {
        let a = dmatrix![2.0, 2.0, 7.0; 2.0, 5.0, 9.0; 7.0, 9.0, 2.0];
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let av = a * &v;
        println!("{l} {v}");
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(av[(i, j)], (l[j] * v[(i, j)]), epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_eig_dual64() {
        let a = dmatrix![
            Dual64::new(2.0, 1.0), Dual64::new(2.0, 2.0);
            Dual64::new(2.0, 2.0), Dual64::new(5.0, 3.0)
        ];
        let (l, v) = jacobi_eigenvalue(a.clone(), 200);
        let (l1, v1) = smallest_ev(a.clone());
        let av = a * &v;
        println!("{l} {v}");
        println!("{l1} {v1}");
        assert_abs_diff_eq!(av[(0, 0)].re, (l[0] * v[(0, 0)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 0)].re, (l[0] * v[(1, 0)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 1)].re, (l[1] * v[(0, 1)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 1)].re, (l[1] * v[(1, 1)]).re, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 0)].eps, (l[0] * v[(0, 0)]).eps, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 0)].eps, (l[0] * v[(1, 0)]).eps, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(0, 1)].eps, (l[1] * v[(0, 1)]).eps, epsilon = 1e-14);
        assert_abs_diff_eq!(av[(1, 1)].eps, (l[1] * v[(1, 1)]).eps, epsilon = 1e-14);
        assert_abs_diff_eq!(l[0].re, l1.re, epsilon = 1e-14);
        assert_abs_diff_eq!(l[0].eps, l1.eps, epsilon = 1e-14);
        assert_abs_diff_eq!(v[(0, 0)].re, v1[0].re, epsilon = 1e-14);
        assert_abs_diff_eq!(v[(0, 0)].eps, v1[0].eps, epsilon = 1e-14);
        assert_abs_diff_eq!(v[(1, 0)].re, v1[1].re, epsilon = 1e-14);
        assert_abs_diff_eq!(v[(1, 0)].eps, v1[1].eps, epsilon = 1e-14);
    }

    #[test]
    fn test_norm_f64() {
        let v = dvector![3.0, 4.0];
        assert_eq!(v.norm(), 5.0);
    }

    #[test]
    fn test_norm_dual64() {
        let v = dvector![Dual64::new(3.0, 1.0), Dual64::new(4.0, 3.0)];
        println!("{}", v.norm());
        assert_eq!(v.norm().re, 5.0);
        assert_eq!(v.norm().eps, 3.0);
    }
}
