use super::Scale;
use num_traits::{One, Zero};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct StaticMat<T, const M: usize, const N: usize>([[T; N]; M]);

pub type StaticVec<T, const N: usize> = StaticMat<T, 1, N>;

impl<T, const M: usize, const N: usize> StaticMat<T, M, N> {
    pub fn new(mat: [[T; N]; M]) -> Self {
        Self(mat)
    }
}

impl<T, const N: usize> StaticVec<T, N> {
    pub fn new_vec(vec: [T; N]) -> Self {
        Self([vec])
    }
}

impl<T: Copy + Zero, const M: usize, const N: usize> StaticMat<T, M, N> {
    pub fn new_zero() -> Self {
        Self([[T::zero(); N]; M])
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
    pub fn sum(&self) -> T {
        self.0[0].iter().fold(T::zero(), |acc, &x| acc + x)
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
    pub fn map<B, F>(self, f: F) -> StaticMat<B, M, N>
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

    pub fn map_zip<B, C, F>(self, other: StaticMat<B, M, N>, f: F) -> StaticMat<C, M, N>
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

    pub fn matmul_transpose<B: Copy, C, const O: usize>(
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

impl<T: Scale<F>, F: Copy, const M: usize, const N: usize> Scale<F> for StaticMat<T, M, N> {
    fn scale(&mut self, f: F) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j].scale(f);
            }
        }
    }
}
