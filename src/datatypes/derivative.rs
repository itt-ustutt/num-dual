use crate::DualNum;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use nalgebra::*;
use num_traits::Zero;
use std::fmt;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Wrapper struct for a derivative vector or matrix.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Derivative<T: DualNum<F>, F, R: Dim, C: Dim>(
    pub(crate) Option<OMatrix<T, R, C>>,
    PhantomData<F>,
)
where
    DefaultAllocator: Allocator<R, C>;

impl<T: DualNum<F> + Copy, F: Copy, const R: usize, const C: usize> Copy
    for Derivative<T, F, Const<R>, Const<C>>
{
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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

    pub(crate) fn map<T2, F2>(&self, f: impl FnMut(T) -> T2) -> Derivative<T2, F2, R, C>
    where
        T2: DualNum<F2>,
        DefaultAllocator: Allocator<R, C>,
    {
        let opt = self.0.as_ref().map(|eps| eps.map(f));
        Derivative::new(opt)
    }

    // A version of map that doesn't clone values before mapping. Useful for the SimdValue impl,
    // which would be redundantly cloning all the lanes of each epsilon value before extracting
    // just one of them.
    //
    // To implement, we inline a copy of Matrix::map, which implicitly clones values, and remove
    // the cloning.
    pub(crate) fn map_borrowed<T2, F2>(
        &self,
        mut f: impl FnMut(&T) -> T2,
    ) -> Derivative<T2, F2, R, C>
    where
        T2: DualNum<F2>,
        DefaultAllocator: Allocator<R, C>,
    {
        let opt = self.0.as_ref().map(move |eps| {
            let (nrows, ncols) = eps.shape_generic();
            let mut res: Matrix<MaybeUninit<T2>, R, C, _> = Matrix::uninit(nrows, ncols);

            for j in 0..ncols.value() {
                for i in 0..nrows.value() {
                    // Safety: all indices are in range.
                    unsafe {
                        let a = eps.data.get_unchecked(i, j);
                        *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(a));
                    }
                }
            }

            // Safety: res is now fully initialized.
            unsafe { res.assume_init() }
        });
        Derivative::new(opt)
    }

    /// Same but bails out if the closure returns None
    pub(crate) fn try_map_borrowed<T2, F2>(
        &self,
        mut f: impl FnMut(&T) -> Option<T2>,
    ) -> Option<Derivative<T2, F2, R, C>>
    where
        T2: DualNum<F2>,
        DefaultAllocator: Allocator<R, C>,
    {
        self.0
            .as_ref()
            .and_then(move |eps| {
                let (nrows, ncols) = eps.shape_generic();
                let mut res: Matrix<MaybeUninit<T2>, R, C, _> = Matrix::uninit(nrows, ncols);

                for j in 0..ncols.value() {
                    for i in 0..nrows.value() {
                        // Safety: all indices are in range.
                        unsafe {
                            let a = eps.data.get_unchecked(i, j);
                            *res.data.get_unchecked_mut(i, j) = MaybeUninit::new(f(a)?);
                        }
                    }
                }

                // Safety: res is now fully initialized.
                Some(unsafe { res.assume_init() })
            })
            .map(Derivative::some)
    }

    pub fn derivative_generic(r: R, c: C, i: usize) -> Self {
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
                    write!(f, "[{}]", x.join(", "))?
                }
                (_, _) => write!(f, "{m}")?,
            };
            write!(f, "{symbol}")?;
        }
        write!(f, "")
    }
}

impl<T: DualNum<F>, F> Derivative<T, F, U1, U1> {
    #[expect(clippy::self_named_constructors)]
    pub fn derivative() -> Self {
        Self::some(SVector::identity())
    }

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
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.map(|x| x * rhs))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Mul<T> for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn mul(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| x * rhs))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim, R2: Dim, C2: Dim> Mul<&Derivative<T, F, R2, C2>>
    for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<R2, C2> + Allocator<R, C2>,
    ShapeConstraint: SameNumberOfRows<C, R2>,
{
    type Output = Derivative<T, F, R, C2>;

    fn mul(self, rhs: &Derivative<T, F, R2, C2>) -> Derivative<T, F, R, C2> {
        Derivative::new(self.0.as_ref().zip(rhs.0.as_ref()).map(|(s, r)| s * r))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Div<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.map(|x| x / rhs))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Div<T> for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn div(self, rhs: T) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| x / rhs))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    pub fn tr_mul<R2: Dim, C2: Dim>(
        &self,
        rhs: &Derivative<T, F, R2, C2>,
    ) -> Derivative<T, F, C, C2>
    where
        DefaultAllocator: Allocator<R2, C2> + Allocator<C, C2>,
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
    DefaultAllocator: Allocator<R, C>,
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

impl<T: DualNum<F>, F, R: Dim, C: Dim> Add<&Derivative<T, F, R, C>> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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

impl<T: DualNum<F>, F, R: Dim, C: Dim> Add for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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
    DefaultAllocator: Allocator<R, C>,
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

impl<T: DualNum<F>, F, R: Dim, C: Dim> Sub<&Derivative<T, F, R, C>> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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

impl<T: DualNum<F>, F, R: Dim, C: Dim> Sub for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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

impl<T: DualNum<F>, F, R: Dim, C: Dim> Neg for &Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Derivative<T, F, R, C>;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.as_ref().map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> Neg for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Derivative::new(self.0.map(|x| -x))
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> AddAssign for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
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
    DefaultAllocator: Allocator<R, C>,
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
    DefaultAllocator: Allocator<R, C>,
{
    fn mul_assign(&mut self, rhs: T) {
        if let Some(s) = &mut self.0 {
            *s *= rhs
        }
    }
}

impl<T: DualNum<F>, F, R: Dim, C: Dim> DivAssign<T> for Derivative<T, F, R, C>
where
    DefaultAllocator: Allocator<R, C>,
{
    fn div_assign(&mut self, rhs: T) {
        if let Some(s) = &mut self.0 {
            *s /= rhs
        }
    }
}

impl<T, R: Dim, C: Dim> nalgebra::SimdValue for Derivative<T, T::Element, R, C>
where
    DefaultAllocator: Allocator<R, C>,
    T: DualNum<T::Element> + SimdValue + Scalar,
    T::Element: DualNum<T::Element> + Scalar + Zero,
{
    type Element = Derivative<T::Element, T::Element, R, C>;

    type SimdBool = T::SimdBool;

    const LANES: usize = T::LANES;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        val.map(|e| T::splat(e))
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        self.map_borrowed(|e| T::extract(e, i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        let opt = self
            .map_borrowed(|e| unsafe { T::extract_unchecked(e, i) })
            .0
            // Now check it's all zeros.
            // Unfortunately there is no way to use the vectorized version of `is_zero`, which is
            // only for matrices with statically known dimensions. Specialization would be
            // required.
            .filter(|x| Iterator::any(&mut x.iter(), |e| !e.is_zero()));
        Derivative::new(opt)
    }

    // SIMD code will expect to be able to replace one lane with another Self::Element,
    // even with a None Derivative, e.g.
    //
    // let single = Derivative::none();
    // let mut x4 = Derivative::splat(single);
    // let one = Derivative::some(...);
    // x4.replace(1, one);
    //
    // So the implementation of `replace` will need to auto-upgrade to Some(zeros) in
    // order to satisfy requests like that.
    fn replace(&mut self, i: usize, val: Self::Element) {
        match (&mut self.0, val.0) {
            (Some(ours), Some(theirs)) => {
                ours.zip_apply(&theirs, |e, replacement| e.replace(i, replacement));
            }
            (ours @ None, Some(theirs)) => {
                let (r, c) = theirs.shape_generic();
                let mut init: OMatrix<T, R, C> = OMatrix::zeros_generic(r, c);
                init.zip_apply(&theirs, |e, replacement| e.replace(i, replacement));
                *ours = Some(init);
            }
            (Some(ours), None) => {
                ours.apply(|e| e.replace(i, T::Element::zero()));
            }
            _ => {}
        }
    }

    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        match (&mut self.0, val.0) {
            (Some(ours), Some(theirs)) => {
                ours.zip_apply(&theirs, |e, replacement| unsafe {
                    e.replace_unchecked(i, replacement)
                });
            }
            (ours @ None, Some(theirs)) => {
                let (r, c) = theirs.shape_generic();
                let mut init: OMatrix<T, R, C> = OMatrix::zeros_generic(r, c);
                init.zip_apply(&theirs, |e, replacement| unsafe {
                    e.replace_unchecked(i, replacement)
                });
                *ours = Some(init);
            }
            (Some(ours), None) => {
                ours.apply(|e| unsafe { e.replace_unchecked(i, T::Element::zero()) });
            }
            _ => {}
        }
    }

    fn select(mut self, cond: Self::SimdBool, other: Self) -> Self {
        // If cond is mixed, then we may need to generate big zero matrices to do the
        // component-wise select on. So check if cond is all-true or all-first to avoid that.
        if cond.all() {
            self
        } else if cond.none() {
            other
        } else {
            match (&mut self.0, other.0) {
                (Some(ours), Some(theirs)) => {
                    ours.zip_apply(&theirs, |e, other_e| {
                        // this will probably get optimized out
                        let e_ = std::mem::replace(e, T::zero());
                        *e = e_.select(cond, other_e)
                    });
                    self
                }
                (Some(ours), None) => {
                    ours.apply(|e| {
                        // this will probably get optimized out
                        let e_ = std::mem::replace(e, T::zero());
                        *e = e_.select(cond, T::zero());
                    });
                    self
                }
                (ours @ None, Some(mut theirs)) => {
                    use std::ops::Not;
                    let inverted: T::SimdBool = cond.not();
                    theirs.apply(|e| {
                        // this will probably get optimized out
                        let e_ = std::mem::replace(e, T::zero());
                        *e = e_.select(inverted, T::zero());
                    });
                    *ours = Some(theirs);
                    self
                }
                _ => self,
            }
        }
    }
}

use simba::scalar::{SubsetOf, SupersetOf};

impl<TSuper, FSuper, T, F, R: Dim, C: Dim> SubsetOf<Derivative<TSuper, FSuper, R, C>>
    for Derivative<T, F, R, C>
where
    TSuper: DualNum<FSuper> + SupersetOf<T>,
    T: DualNum<F>,
    DefaultAllocator: Allocator<R, C>,
    // DefaultAllocator: Allocator<D>
    //     + Allocator<U1, D>
    //     + Allocator<D, U1>
    //     + Allocator<D, D>,
{
    #[inline(always)]
    fn to_superset(&self) -> Derivative<TSuper, FSuper, R, C> {
        self.map_borrowed(|elem| TSuper::from_subset(elem))
    }
    #[inline(always)]
    fn from_superset(element: &Derivative<TSuper, FSuper, R, C>) -> Option<Self> {
        element.try_map_borrowed(|elem| TSuper::to_subset(elem))
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &Derivative<TSuper, FSuper, R, C>) -> Self {
        element.map_borrowed(|elem| TSuper::to_subset_unchecked(elem))
    }
    #[inline(always)]
    fn is_in_subset(element: &Derivative<TSuper, FSuper, R, C>) -> bool {
        element
            .0
            .as_ref()
            .is_none_or(|matrix| matrix.iter().all(|elem| TSuper::is_in_subset(elem)))
    }
}
