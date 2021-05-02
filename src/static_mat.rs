use super::Scale;
use num_traits::{Float, One, Zero};
use std::fmt;
use std::iter::Flatten;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};
use std::slice::Iter;

/// A statically allocated MxN matrix. The struct is used in the vector (hyper) dual numbers
/// and provides utilities for the calculation of Jacobians.
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct StaticMat<T, const M: usize, const N: usize>([[T; N]; M]);

pub type StaticVec<T, const N: usize> = StaticMat<T, 1, N>;

impl<T, const M: usize, const N: usize> StaticMat<T, M, N> {
    /// Create a new StaticMat from an array of arrays.
    pub fn new(mat: [[T; N]; M]) -> Self {
        Self(mat)
    }
}

impl<T, const N: usize> StaticVec<T, N> {
    /// Create a new StaticVec from an array.
    pub fn new_vec(vec: [T; N]) -> Self {
        Self([vec])
    }
}

impl<T: Copy + Zero + One + AddAssign, const N: usize> StaticMat<T, N, N> {
    /// Create a NxN unity matrix.
    pub fn eye() -> Self {
        let mut res = Self::zero();
        for i in 0..N {
            res[(i, i)] = T::one()
        }
        res
    }
}

macro_rules! impl_op {
    ($trt:ident, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Copy + $trt_assign, const M: usize, const N: usize> $trt for StaticMat<T, M, N> {
            type Output = StaticMat<T, M, N>;
            fn $mth(mut self, other: Self) -> Self {
                for i in 0..M {
                    for j in 0..N {
                        self.0[i][j] $op_assign other.0[i][j];
                    }
                }
                self
            }
        }

        impl<T: Copy + $trt_assign, const M: usize, const N: usize> $trt_assign for StaticMat<T, M, N> {
            fn $mth_assign(&mut self, other: Self) {
                for i in 0..M {
                    for j in 0..N {
                        self.0[i][j] $op_assign other.0[i][j];
                    }
                }
            }
        }

        impl<T: Copy + $trt_assign, const M: usize, const N: usize> $trt<T> for StaticMat<T, M, N> {
            type Output = StaticMat<T, M, N>;
            fn $mth(mut self, other: T) -> Self {
                for i in 0..M {
                    for j in 0..N {
                        self.0[i][j] $op_assign other;
                    }
                }
                self
            }
        }

        impl<T: Copy + $trt_assign, const M: usize, const N: usize> $trt_assign<T> for StaticMat<T, M, N> {
            fn $mth_assign(&mut self, other: T) {
                for i in 0..M {
                    for j in 0..N {
                        self.0[i][j] $op_assign other;
                    }
                }
            }
        }
    };
}

impl_op!(Add, add, AddAssign, +=, add_assign);
impl_op!(Sub, sub, SubAssign, -=, sub_assign);
impl_op!(Mul, mul, MulAssign, *=, mul_assign);
impl_op!(Div, div, DivAssign, /=, div_assign);
impl_op!(Rem, rem, RemAssign, %=, rem_assign);

impl<T: Zero + Copy, const N: usize> StaticVec<T, N> {
    /// sum over all elements in the vector.
    pub fn sum(&self) -> T {
        self.0[0].iter().fold(T::zero(), |acc, &x| acc + x)
    }
}

impl<T: Float, const N: usize> StaticVec<T, N> {
    /// Calculate the Euclidian norm of the vector
    pub fn norm(&self) -> T {
        self.0[0]
            .iter()
            .fold(T::zero(), |acc, &x| acc + x.powi(2))
            .sqrt()
    }
}

impl<T: Copy + Zero + AddAssign, const M: usize, const N: usize> Zero for StaticMat<T, M, N> {
    fn zero() -> Self {
        Self([[T::zero(); N]; M])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|r| r.iter().all(T::is_zero))
    }
}

impl<T: Copy + One + MulAssign, const M: usize, const N: usize> One for StaticMat<T, M, N> {
    fn one() -> Self {
        Self([[T::one(); N]; M])
    }
}

impl<T: Copy + Neg<Output = T>, const M: usize, const N: usize> Neg for StaticMat<T, M, N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] = -self.0[i][j];
            }
        }
        self
    }
}

impl<T: fmt::Display, const M: usize, const N: usize> fmt::Display for StaticMat<T, M, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..M {
            write!(f, "[")?;
            for j in 0..N {
                write!(f, "{}", self.0[i][j])?;
                if j != N - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i != M - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for StaticMat<T, M, N> {
    type Output = T;
    fn index(&self, i: (usize, usize)) -> &Self::Output {
        &self.0[i.0][i.1]
    }
}

impl<T, const N: usize> Index<usize> for StaticVec<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[0][i]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for StaticMat<T, M, N> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut Self::Output {
        &mut self.0[i.0][i.1]
    }
}

impl<T, const N: usize> IndexMut<usize> for StaticVec<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[0][i]
    }
}

impl<T: Copy, const M: usize, const N: usize> StaticMat<T, M, N> {
    /// Apply a function elementwise to all elements of the matrix and return a new matrix with the results.
    pub fn map<B, F>(&self, f: F) -> StaticMat<B, M, N>
    where
    B: Copy + Zero,
    F: Fn(T) -> B,
    {
        let mut res = [[B::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = f(self.0[i][j])
            }
        }
        StaticMat(res)
    }
    
    /// Apply a function elementwise to all elements of the matrix and a second matrix.
    /// Return a new matrix with the results.
    pub fn map_zip<B, C, F>(&self, other: &StaticMat<B, M, N>, f: F) -> StaticMat<C, M, N>
    where
        B: Copy,
        C: Copy + Zero,
        F: Fn(T, B) -> C,
    {
        let mut res = [[C::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = f(self.0[i][j], other.0[i][j])
            }
        }
        StaticMat(res)
    }

    /// Iterate over all matrix elements.
    pub fn iter(&self) -> Flatten<Iter<'_, [T; N]>> {
        self.0.iter().flatten()
    }

    /// Perform a matrix-matrix multiplication.
    /// ```
    /// use num_hyperdual::StaticMat;
    /// let a = StaticMat::new([[1, 2], [3, 4]]);
    /// let b = StaticMat::new([[2, 1], [4, 3]]);
    /// let x = a.matmul(&b);
    /// assert_eq!(x[(0,0)], 10);
    /// assert_eq!(x[(0,1)], 7);
    /// assert_eq!(x[(1,0)], 22);
    /// assert_eq!(x[(1,1)], 15);
    /// ```
    pub fn matmul<B: Copy, C, const O: usize>(
        &self,
        other: &StaticMat<B, N, O>,
    ) -> StaticMat<C, M, O>
    where
        C: Copy + Zero + AddAssign,
        T: Mul<B, Output = C>,
    {
        let mut res = [[C::zero(); O]; M];
        for i in 0..M {
            for j in 0..N {
                for k in 0..O {
                    res[i][k] += self.0[i][j] * other.0[j][k];
                }
            }
        }
        StaticMat(res)
    }

    /// Perform a matrix-matrix multiplication in which the first matrix is transposed.
    /// ```
    /// use num_hyperdual::StaticVec;
    /// let a = StaticVec::new_vec([1, 2]);
    /// let b = StaticVec::new_vec([2, 1]);
    /// let x = a.transpose_matmul(&b);
    /// assert_eq!(x[(0,0)], 2);
    /// assert_eq!(x[(0,1)], 1);
    /// assert_eq!(x[(1,0)], 4);
    /// assert_eq!(x[(1,1)], 2);
    /// ```
    pub fn transpose_matmul<B: Copy, C, const O: usize>(
        &self,
        other: &StaticMat<B, M, O>,
    ) -> StaticMat<C, N, O>
    where
        C: Copy + Zero + AddAssign,
        T: Mul<B, Output = C>,
    {
        let mut res = [[C::zero(); O]; N];
        for i in 0..M {
            for j in 0..N {
                for k in 0..O {
                    res[j][k] += self.0[i][j] * other.0[i][k];
                }
            }
        }
        StaticMat(res)
    }

    /// Perform a matrix-matrix multiplication in which the second matrix is transposed.
    /// ```
    /// use num_hyperdual::{StaticMat, StaticVec};
    /// let a = StaticMat::new([[1, 2], [3, 4]]);
    /// let b = StaticVec::new_vec([2, 1]);
    /// let x = a.matmul_transpose(&b);
    /// assert_eq!(x[(0, 0)], 4);
    /// assert_eq!(x[(1, 0)], 10);
    /// ```
    pub fn matmul_transpose<B: Copy, C, const O: usize>(
        &self,
        other: &StaticMat<B, O, N>,
    ) -> StaticMat<C, M, O>
    where
        C: Copy + Zero + AddAssign,
        T: Mul<B, Output = C>,
    {
        let mut res = [[C::zero(); O]; M];
        for i in 0..M {
            for j in 0..N {
                for k in 0..O {
                    res[i][k] += self.0[i][j] * other.0[k][j];
                }
            }
        }
        StaticMat(res)
    }

    /// Transpose the matrix.
    pub fn t(&self) -> StaticMat<T, N, M>
    where
        T: Zero,
    {
        let mut res = [[T::zero(); M]; N];
        for i in 0..M {
            for j in 0..N {
                res[j][i] = self.0[i][j];
            }
        }
        StaticMat(res)
    }
}

impl<T: Copy + Zero, const N: usize> StaticVec<T, N> {
    /// Return a new vector containing the first M elements of self.
    pub fn first<const M: usize>(&self) -> StaticVec<T, M> {
        let mut res = [[T::zero(); M]; 1];
        for i in 0..M {
            res[0][i] = self[i];
        }
        StaticMat(res)
    }
    
    /// Return a new vector containing the last M elements of self.
    pub fn last<const M: usize>(&self) -> StaticVec<T, M> {
        let mut res = [[T::zero(); M]; 1];
        for i in 0..M {
            res[0][i] = self[i + N - M];
        }
        StaticMat(res)
    }

    /// Calculate the dot product of two vectors.
    /// ```
    /// use num_hyperdual::StaticVec;
    /// let a = StaticVec::new_vec([1, 2, 3, 4]);
    /// let b = StaticVec::new_vec([2, 1, 4, 3]);
    /// assert_eq!(a.dot(&b), 28);
    /// ```
    pub fn dot(&self, other: &StaticVec<T, N>) -> T
    where
        T: Mul<Output = T> + AddAssign,
    {
        self.matmul_transpose(other)[0]
    }
}

impl<T: Scale<F>, F: Copy, const M: usize, const N: usize> Scale<F> for StaticMat<T, M, N> {
    fn scale(&mut self, f: F) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j].scale(f);
            }
        }
    }
}
