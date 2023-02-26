use crate::DualNum;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use nalgebra::*;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Derivative<T: DualNum<F>, F, R: Dim, C: Dim>(
    pub(crate) Option<OMatrix<T, R, C>>,
    PhantomData<F>,
)
where
    DefaultAllocator: Allocator<T, R, C>;

impl<T: DualNum<F> + Copy, F: Copy, const R: usize, const C: usize> Copy
    for Derivative<T, F, Const<R>, Const<C>>
{
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    pub fn new(derivative: Option<OMatrix<T, R, C>>) -> Self {
        Self(derivative, PhantomData)
    }

    pub fn some(derivative: OMatrix<T, R, C>) -> Self {
        Self::new(Some(derivative))
    }

    pub fn none() -> Self {
        Self::new(None)
    }

    #[allow(clippy::self_named_constructors)]
    pub fn derivative(r: R, c: C, i: usize) -> Self {
        let mut m = OMatrix::zeros_generic(r, c);
        m[i] = T::one();
        Self::some(m)
    }

    pub fn unwrap_generic(self, r: R, c: C) -> OMatrix<T, R, C> {
        self.0.unwrap_or_else(|| OMatrix::zeros_generic(r, c))
    }

    pub fn fmt(&self, f: &mut fmt::Formatter, symbol: &str) -> fmt::Result {
        if let Some(m) = self.0.as_ref() {
            write!(f, " + ")?;
            match m.shape() {
                (1, 1) => write!(f, "{}", m[0])?,
                (1, _) | (_, 1) => {
                    let x: Vec<_> = m.iter().map(T::to_string).collect();
                    write!(f, "[{}]", x.join(","))?
                }
                (_, _) => write!(f, "{}", m)?,
            };
            write!(f, "{symbol}")?;
        }
        write!(f, "")
    }
}

impl<T: DualNum<F>, F> Derivative<T, F, U1, U1> {
    pub fn unwrap(self) -> T {
        self.0.map_or_else(
            || T::zero(),
            |s| {
                let [[r]] = s.data.0;
                r
            },
        )
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Mul<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.map(|x| x * rhs))
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Mul<T> for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| x * rhs))
    }
}

impl<'a, 'b, T: DualNum<F>, F, R: Dim, C: Dim, R2: Dim, C2: Dim> Mul<&'b Derivative<T, F, R2, C2>>
    for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, R2, C2> + Allocator<T, R, C2>,
    ShapeConstraint: SameNumberOfRows<C, R2>,
{
    type Output = Derivative<T, F, R, C2>;

    fn mul(self, rhs: &Derivative<T, F, R2, C2>) -> Derivative<T, F, R, C2> {
        Derivative::new(self.0.as_ref().zip(rhs.0.as_ref()).map(|(s, r)| s * r))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    pub fn tr_mul<R2: Dim, C2: Dim>(
        &self,
        rhs: &Derivative<T, F, R2, C2>,
    ) -> Derivative<T, F, C, C2>
    where
        DefaultAllocator: Allocator<T, R2, C2> + Allocator<T, C, C2>,
        ShapeConstraint: SameNumberOfRows<R, R2>,
    {
        Derivative::new(
            self.0
                .as_ref()
                .zip(rhs.0.as_ref())
                .map(|(s, r)| s.tr_mul(r)),
        )
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Add for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(match (self.0, rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Add<&'a Derivative<T, F, R, C>>
    for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn add(self, rhs: &Derivative<T, F, R, C>) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(r.clone()),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Add for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn add(self, rhs: Self) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s + r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(r.clone()),
            (None, None) => None,
        })
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Sub for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(match (self.0, rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s),
            (None, Some(r)) => Some(-r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Sub<&'a Derivative<T, F, R, C>>
    for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn sub(self, rhs: &Derivative<T, F, R, C>) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(-r.clone()),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Sub for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        Derivative::new(match (&self.0, &rhs.0) {
            (Some(s), Some(r)) => Some(s - r),
            (Some(s), None) => Some(s.clone()),
            (None, Some(r)) => Some(-r),
            (None, None) => None,
        })
    }
}

impl<'a, T: DualNum<F>, F, R: Dim, C: Dim> Neg for &'a Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Neg for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> AddAssign for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn add_assign(&mut self, rhs: Self) {
        match (&mut self.0, rhs.0) {
            (Some(s), Some(r)) => *s += &r,
            (None, Some(r)) => self.0 = Some(r),
            (_, None) => (),
        };
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> SubAssign for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn sub_assign(&mut self, rhs: Self) {
        match (&mut self.0, rhs.0) {
            (Some(s), Some(r)) => *s -= &r,
            (None, Some(r)) => self.0 = Some(-&r),
            (_, None) => (),
        };
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> MulAssign<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn mul_assign(&mut self, rhs: T) {
        match &mut self.0 {
            Some(s) => *s *= rhs,
            None => (),
        }
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> DivAssign<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<T, R, C>,
{
    fn div_assign(&mut self, rhs: T) {
        match &mut self.0 {
            Some(s) => *s /= rhs,
            None => (),
        }
    }
}
