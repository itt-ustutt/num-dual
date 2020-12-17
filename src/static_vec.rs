use num_traits::Zero;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};
use std::slice::SliceIndex;

#[derive(Clone, Copy)]
pub struct StaticVec<T, const N: usize>([T; N]);

impl<T: Copy + Add<Output = T> + Zero, const N: usize> Add for StaticVec<T, N> {
    type Output = StaticVec<T, N>;
    fn add(self, other: Self) -> Self {
        let mut res = [T::zero(); N];
        for i in 0..N {
            res[i] = self.0[i] + other.0[i];
        }
        Self(res)
    }
}

impl<T: Copy + Sub<Output = T> + Zero, const N: usize> Sub for StaticVec<T, N> {
    type Output = StaticVec<T, N>;
    fn sub(self, other: Self) -> Self {
        let mut res = [T::zero(); N];
        for i in 0..N {
            res[i] = self.0[i] - other.0[i];
        }
        Self(res)
    }
}

macro_rules! impl_op {
    ($trt:ident, $operator:tt, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Copy + $trt_assign, const N: usize> $trt_assign for StaticVec<T, N> {
            fn $mth_assign(&mut self, other: Self) {
                for i in 0..N {
                    self.0[i] $op_assign other.0[i];
                }
            }
        }

        impl<T: Copy + $trt<Output = T> + Zero, const N: usize> $trt<T> for StaticVec<T, N> {
            type Output = StaticVec<T, N>;
            fn $mth(self, other: T) -> Self {
                let mut res = [T::zero(); N];
                for i in 0..N {
                    res[i] = self.0[i] $operator other;
                }
                Self(res)
            }
        }

        impl<T: Copy + $trt_assign, const N: usize> $trt_assign<T> for StaticVec<T, N> {
            fn $mth_assign(&mut self, other: T) {
                for i in 0..N {
                    self.0[i] $op_assign other;
                }
            }
        }
    };
}

impl_op!(Add, +, add, AddAssign, +=, add_assign);
impl_op!(Sub, -, sub, SubAssign, -=, sub_assign);
impl_op!(Mul, *, mul, MulAssign, *=, mul_assign);
impl_op!(Div, /, div, DivAssign, /=, div_assign);
impl_op!(Rem, %, rem, RemAssign, %=, rem_assign);

impl<T: Copy + Zero, const N: usize> Zero for StaticVec<T, N> {
    fn zero() -> Self {
        Self([T::zero(); N])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(T::is_zero)
    }
}

impl<T: Copy + Neg<Output = T> + Zero, const N: usize> Neg for StaticVec<T, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut res = [T::zero(); N];
        for i in 0..N {
            res[i] = -self.0[i];
        }
        Self(res)
    }
}

impl<T: Copy + PartialEq, const N: usize> PartialEq for StaticVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(&s, &o)| s == o)
    }
}

impl<T: fmt::Display, const N: usize> fmt::Display for StaticVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..N {
            write!(f, "{}", self.0[i])?;
            if i != N - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl<T, const N: usize, I> Index<I> for StaticVec<T, N>
where
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;
    fn index(&self, i: I) -> &Self::Output {
        &self.0[i]
    }
}

impl<T, const N: usize, I> IndexMut<I> for StaticVec<T, N>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        &mut self.0[i]
    }
}

#[derive(Clone, Copy)]
pub struct StaticMat<T, const M: usize, const N: usize>([[T; N]; M]);

impl<T: Copy + Add<Output = T> + Zero, const M: usize, const N: usize> Add for StaticMat<T, M, N> {
    type Output = StaticMat<T, M, N>;
    fn add(self, other: Self) -> Self {
        let mut res = [[T::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = self.0[i][j] + other.0[i][j];
            }
        }
        Self(res)
    }
}

impl<T: Copy + Sub<Output = T> + Zero, const M: usize, const N: usize> Sub for StaticMat<T, M, N> {
    type Output = StaticMat<T, M, N>;
    fn sub(self, other: Self) -> Self {
        let mut res = [[T::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = self.0[i][j] - other.0[i][j];
            }
        }
        Self(res)
    }
}

macro_rules! impl_op {
    ($trt:ident, $operator:tt, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Copy + $trt_assign, const M: usize, const N: usize> $trt_assign for StaticMat<T, M, N> {
            fn $mth_assign(&mut self, other: Self) {
                for i in 0..M {
                    for j in 0..N {
                        self.0[i][j] $op_assign other.0[i][j];
                    }
                }
            }
        }

        impl<T: Copy + $trt<Output = T> + Zero, const M: usize, const N: usize> $trt<T> for StaticMat<T, M, N> {
            type Output = StaticMat<T, M, N>;
            fn $mth(self, other: T) -> Self {
                let mut res = [[T::zero(); N]; M];
                for i in 0..M {
                    for j in 0..N {
                        res[i][j] = self.0[i][j] $operator other;
                    }
                }
                Self(res)
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

impl_op!(Add, +, add, AddAssign, +=, add_assign);
impl_op!(Sub, -, sub, SubAssign, -=, sub_assign);
impl_op!(Mul, *, mul, MulAssign, *=, mul_assign);
impl_op!(Div, /, div, DivAssign, /=, div_assign);
impl_op!(Rem, %, rem, RemAssign, %=, rem_assign);

impl<T: Copy + Zero, const M: usize, const N: usize> Zero for StaticMat<T, M, N> {
    fn zero() -> Self {
        Self([[T::zero(); N]; M])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|r| r.iter().all(T::is_zero))
    }
}

impl<T: Copy + Neg<Output = T> + Zero, const M: usize, const N: usize> Neg for StaticMat<T, M, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut res = [[T::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = -self.0[i][j];
            }
        }
        Self(res)
    }
}

impl<T: Copy + PartialEq, const M: usize, const N: usize> PartialEq for StaticMat<T, M, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(&s, &o)| s == o)
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

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for StaticMat<T, M, N> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut Self::Output {
        &mut self.0[i.0][i.1]
    }
}

impl<T: Copy + Zero + Mul<Output = T>, const M: usize, const N: usize> Mul<StaticVec<T, N>>
    for StaticVec<T, M>
{
    type Output = StaticMat<T, M, N>;
    fn mul(self, other: StaticVec<T, N>) -> Self::Output {
        let mut res = [[T::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = self.0[i] * other.0[j];
            }
        }
        StaticMat(res)
    }
}
