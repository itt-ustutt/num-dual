use num_traits::Zero;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
    SubAssign,
};
use std::slice::SliceIndex;

#[derive(Clone, Copy)]
pub struct StaticVec<T, const N: usize>([T; N]);

impl<T, const N: usize> StaticVec<T, N> {
    pub fn new(vec: [T; N]) -> Self {
        Self(vec)
    }
}

macro_rules! impl_op {
    ($trt:ident, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
        impl<T: Copy + $trt_assign, const N: usize> $trt for StaticVec<T, N> {
            type Output = StaticVec<T, N>;
            fn $mth(mut self, other: Self) -> Self {
                for i in 0..N {
                    self.0[i] $op_assign other.0[i];
                }
                self
            }
        }

        impl<T: Copy + $trt_assign, const N: usize> $trt_assign for StaticVec<T, N> {
            fn $mth_assign(&mut self, other: Self) {
                for i in 0..N {
                    self.0[i] $op_assign other.0[i];
                }
            }
        }

        impl<T: Copy + $trt_assign, const N: usize> $trt<T> for StaticVec<T, N> {
            type Output = StaticVec<T, N>;
            fn $mth(mut self, other: T) -> Self {
                for i in 0..N {
                    self.0[i] $op_assign other;
                }
                self
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

impl_op!(Add, add, AddAssign, +=, add_assign);
impl_op!(Sub, sub, SubAssign, -=, sub_assign);
impl_op!(Mul, mul, MulAssign, *=, mul_assign);
impl_op!(Div, div, DivAssign, /=, div_assign);
impl_op!(Rem, rem, RemAssign, %=, rem_assign);

impl<T: Copy + AddAssign + Zero, const N: usize> Zero for StaticVec<T, N> {
    fn zero() -> Self {
        Self([T::zero(); N])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(T::is_zero)
    }
}

impl<T: Copy + Neg<Output = T> + Zero, const N: usize> Neg for StaticVec<T, N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for i in 0..N {
            self.0[i] = -self.0[i];
        }
        self
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

impl<T, I, const N: usize> Index<I> for StaticVec<T, N>
where
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;
    fn index(&self, i: I) -> &Self::Output {
        &self.0[i]
    }
}

impl<T, I, const N: usize> IndexMut<I> for StaticVec<T, N>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl<T: Copy, const N: usize> StaticVec<T, N> {
    pub fn map<B, F>(self, f: F) -> StaticVec<B, N>
    where
        B: Copy + Zero,
        F: Fn(T) -> B,
    {
        let mut res = [B::zero(); N];
        for i in 0..N {
            res[i] = f(self.0[i])
        }
        StaticVec(res)
    }

    pub fn map_zip<B, C, F>(self, other: StaticVec<B, N>, f: F) -> StaticVec<C, N>
    where
        B: Copy,
        C: Copy + Zero,
        F: Fn(T, B) -> C,
    {
        let mut res = [C::zero(); N];
        for i in 0..N {
            res[i] = f(self.0[i], other.0[i])
        }
        StaticVec(res)
    }
}

#[derive(Clone, Copy)]
pub struct StaticMat<T, const M: usize, const N: usize>([[T; N]; M]);

impl<T: Copy + AddAssign, const M: usize, const N: usize> Add for StaticMat<T, M, N> {
    type Output = StaticMat<T, M, N>;
    fn add(mut self, other: Self) -> Self {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] += other.0[i][j];
            }
        }
        self
    }
}

impl<T: Copy + SubAssign, const M: usize, const N: usize> Sub for StaticMat<T, M, N> {
    type Output = StaticMat<T, M, N>;
    fn sub(mut self, other: Self) -> Self {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] -= other.0[i][j];
            }
        }
        self
    }
}

macro_rules! impl_op {
    ($trt:ident, $mth:ident, $trt_assign:ident, $op_assign:tt, $mth_assign:ident) => {
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

impl<T: Copy + Zero + AddAssign, const M: usize, const N: usize> Zero for StaticMat<T, M, N> {
    fn zero() -> Self {
        Self([[T::zero(); N]; M])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|r| r.iter().all(T::is_zero))
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

pub trait OuterProduct<T = Self> {
    type Output;
    fn outer_product(self, other: T) -> Self::Output;
}

impl<T: Copy + Zero + Mul<Output = T>, const M: usize, const N: usize> OuterProduct<StaticVec<T, N>>
    for StaticVec<T, M>
{
    type Output = StaticMat<T, M, N>;
    fn outer_product(self, other: StaticVec<T, N>) -> Self::Output {
        let mut res = [[T::zero(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res[i][j] = self.0[i] * other.0[j];
            }
        }
        StaticMat(res)
    }
}
