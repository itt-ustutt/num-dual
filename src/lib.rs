// #![feature(test)]
// extern crate test;

use num_traits::{Inv, One, Zero};
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Neg, Sub};
pub mod array;
pub mod dual;
pub mod hd3;
pub mod hd_scal;
pub mod hyperdual;
pub use dual::*;
pub use hd3::*;
pub use hd_scal::*;
pub use hyperdual::*;

pub trait DualNumOps<T, Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Add<T, Output = Output>
    + Sub<Rhs, Output = Output>
    + Sub<T, Output = Output>
    + Mul<Rhs, Output = Output>
    + Mul<T, Output = Output>
    + Div<Rhs, Output = Output>
    + Div<T, Output = Output>
{
}

impl<T, D, Rhs, Output> DualNumOps<T, Rhs, Output> for D where
    D: Add<Rhs, Output = Output>
        + Add<T, Output = Output>
        + Sub<Rhs, Output = Output>
        + Sub<T, Output = Output>
        + Mul<Rhs, Output = Output>
        + Mul<T, Output = Output>
        + Div<Rhs, Output = Output>
        + Div<T, Output = Output>
{
}

pub trait DualNumRef<T>: DualNumMethods<T> + for<'r> DualNumOps<&'r Self> {}
impl<T, D> DualNumRef<T> for D where D: DualNumMethods<T> + for<'r> DualNumOps<&'r Self> {}

pub trait DualRefNum<T, Base>:
    DualNumOps<T, Base, Base> + for<'r> DualNumOps<T, &'r Base, Base>
{
}
impl<T, D, Base> DualRefNum<T, Base> for D where
    D: DualNumOps<T, Base, Base> + for<'r> DualNumOps<T, &'r Base, Base>
{
}

pub trait DualNum<T>:
    DualNumMethods<T>
    + Copy
    + Zero
    + One
    + Neg<Output = Self>
    + Inv<Output = Self>
    + Sum
    + Product
    + From<T>
{
}
impl<T, D> DualNum<T> for D where
    D: DualNumMethods<T>
        + Copy
        + Zero
        + One
        + Neg<Output = Self>
        + Inv<Output = Self>
        + Sum
        + Product
        + From<T>
{
}

pub trait DualNumMethods<T>: Clone + DualNumOps<T> {
    fn re(&self) -> T;

    fn recip(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, n: T) -> Self;
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn exp_m1(&self) -> Self;
    fn ln(&self) -> Self;
    fn log(&self, base: T) -> Self;
    fn log2(&self) -> Self;
    fn log10(&self) -> Self;
    fn ln_1p(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn sin_cos(&self) -> (Self, Self);
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn sph_j0(&self) -> Self;
    fn sph_j1(&self) -> Self;
    fn sph_j2(&self) -> Self;

    #[inline]
    fn mul_add(&self, a: Self, b: Self) -> Self {
        self.clone() * a + b
    }

    #[inline]
    fn powd(&self, exp: &Self) -> Self {
        (self.ln() * exp.clone()).exp()
    }
}

macro_rules! impl_dual_num_float {
    ($float:ty) => {
        impl DualNumMethods<$float> for $float {
            fn re(&self) -> $float {
                *self
            }

            fn mul_add(&self, a: Self, b: Self) -> Self {
                <$float>::mul_add(*self, a, b)
            }
            fn recip(&self) -> Self {
                <$float>::recip(*self)
            }
            fn powi(&self, n: i32) -> Self {
                <$float>::powi(*self, n)
            }
            fn powf(&self, n: Self) -> Self {
                <$float>::powf(*self, n)
            }
            fn powd(&self, n: &Self) -> Self {
                <$float>::powf(*self, *n)
            }
            fn sqrt(&self) -> Self {
                <$float>::sqrt(*self)
            }
            fn exp(&self) -> Self {
                <$float>::exp(*self)
            }
            fn exp2(&self) -> Self {
                <$float>::exp2(*self)
            }
            fn ln(&self) -> Self {
                <$float>::ln(*self)
            }
            fn log(&self, base: Self) -> Self {
                <$float>::log(*self, base)
            }
            fn log2(&self) -> Self {
                <$float>::log2(*self)
            }
            fn log10(&self) -> Self {
                <$float>::log10(*self)
            }
            fn cbrt(&self) -> Self {
                <$float>::cbrt(*self)
            }
            fn sin(&self) -> Self {
                <$float>::sin(*self)
            }
            fn cos(&self) -> Self {
                <$float>::cos(*self)
            }
            fn tan(&self) -> Self {
                <$float>::tan(*self)
            }
            fn asin(&self) -> Self {
                <$float>::asin(*self)
            }
            fn acos(&self) -> Self {
                <$float>::acos(*self)
            }
            fn atan(&self) -> Self {
                <$float>::atan(*self)
            }
            fn sin_cos(&self) -> (Self, Self) {
                <$float>::sin_cos(*self)
            }
            fn exp_m1(&self) -> Self {
                <$float>::exp_m1(*self)
            }
            fn ln_1p(&self) -> Self {
                <$float>::ln_1p(*self)
            }
            fn sinh(&self) -> Self {
                <$float>::sinh(*self)
            }
            fn cosh(&self) -> Self {
                <$float>::cosh(*self)
            }
            fn tanh(&self) -> Self {
                <$float>::tanh(*self)
            }
            fn asinh(&self) -> Self {
                <$float>::asinh(*self)
            }
            fn acosh(&self) -> Self {
                <$float>::acosh(*self)
            }
            fn atanh(&self) -> Self {
                <$float>::atanh(*self)
            }
            fn sph_j0(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    1.0 - self * self / 6.0
                } else {
                    self.sin() / self
                }
            }
            fn sph_j1(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    self / 3.0
                } else {
                    let sc = self.sin_cos();
                    let rec = self.recip();
                    (sc.0 * rec - sc.1) * rec
                }
            }
            fn sph_j2(&self) -> Self {
                if self.abs() < <$float>::EPSILON {
                    self * self / 15.0
                } else {
                    let sc = self.sin_cos();
                    let s2 = self * self;
                    ((3.0 - s2) * sc.0 - 3.0 * self * sc.1) / (self * s2)
                }
            }
        }
    };
}

impl_dual_num_float!(f32);
impl_dual_num_float!(f64);

#[cfg(test)]
mod bench {
    use super::dual::Dual64;
    use super::hd3::HD3_64;
    use super::hyperdual::HyperDual64;
    use super::*;
    use ndarray::*;
    // use test::Bencher;

    trait HelmholtzEnergy {
        fn helmholtz_energy<T: DualNum<f64>>(
            &self,
            temperature: T,
            volume: T,
            moles: &Array1<T>,
        ) -> T;
    }

    struct HSContribution {
        m: Array1<f64>,
        sigma: Array1<f64>,
        epsilon_k: Array1<f64>,
    }

    impl HelmholtzEnergy for HSContribution {
        fn helmholtz_energy<T: DualNum<f64>>(
            &self,
            temperature: T,
            volume: T,
            moles: &Array1<T>,
        ) -> T {
            let vi = volume.recip();
            let density = moles.mapv(|m| m * vi);
            let ti = temperature.recip() * -3.0;
            let d = Array::from_shape_fn(self.m.len(), |i| {
                -((ti * self.epsilon_k[i]).exp() * 0.12 - 1.0) * self.sigma[i]
            });
            let mut zeta: [T; 4] = [T::zero(), T::zero(), T::zero(), T::zero()];
            let mut m_rho: T = T::zero();
            for i in 0..self.m.len() {
                for k in 0..4 {
                    zeta[k] = zeta[k]
                        + density[i]
                            * d[i].powi(k as i32)
                            * (std::f64::consts::PI / 6.0 * self.m[i]);
                }
                m_rho = m_rho + density[i] * self.m[i];
            }
            let frac_1mz3 = -(zeta[3] - 1.0).recip();
            let frac_z3 = zeta[3].recip();
            volume
                * m_rho
                * zeta[0].recip()
                * (zeta[1] * zeta[2] * frac_z3 * 3.0
                    + zeta[2].powi(3) * frac_1mz3.powi(2) * frac_z3
                    + (zeta[2].powi(3) * frac_z3.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
        }
    }

    fn init_state<T: Clone + From<f64>>() -> (T, T, Array1<T>) {
        let temperature = T::from(300.0);
        let volume = T::from(1.0);
        let moles = arr1(&[T::from(0.001), T::from(0.005)]);
        (temperature, volume, moles)
    }

    #[test]
    fn test_first_derivative() {
        let (mut t_d, v_d, m_d) = init_state::<Dual64>();
        t_d.eps = 1.0;
        let (mut t_hd, v_hd, m_hd) = init_state::<HyperDual64>();
        t_hd.eps1 = 1.0;
        // let (mut t_hds, v_hds, m_hds) = init_state::<HDScal64<D1>>();
        // t_hds = t_hds.derive();
        let (mut t_hd3, v_hd3, m_hd3) = init_state::<HD3_64>();
        t_hd3 = t_hd3.derive();

        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };

        let r_d = hs.helmholtz_energy(t_d, v_d, &m_d);
        let r_hd = hs.helmholtz_energy(t_hd, v_hd, &m_hd);
        // let r_hds = hs.helmholtz_energy(t_hds, v_hds, &m_hds);
        let r_hd3 = hs.helmholtz_energy(t_hd3, v_hd3, &m_hd3);
        assert_eq!(r_d.eps, r_hd.eps1);
        // assert_eq!(r_d.eps, r_hds.0[1]);
        assert_eq!(r_d.eps, r_hd3.0[1]);
    }

    #[test]
    fn test_second_derivative() {
        let (t_hd, mut v_hd, m_hd) = init_state::<HyperDual64>();
        v_hd.eps1 = 1.0;
        v_hd.eps2 = 1.0;
        // let (t_hds, mut v_hds, m_hds) = init_state::<HDScal64<D2>>();
        // v_hds = v_hds.derive();
        let (t_hd3, mut v_hd3, m_hd3) = init_state::<HD3_64>();
        v_hd3 = v_hd3.derive();

        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };

        let r_hd = hs.helmholtz_energy(t_hd, v_hd, &m_hd);
        // let r_hds = hs.helmholtz_energy(t_hds, v_hds, &m_hds);
        let r_hd3 = hs.helmholtz_energy(t_hd3, v_hd3, &m_hd3);
        // assert!((r_hd.eps1eps2 - r_hds.0[2]).abs() < 1e-10);
        assert!((r_hd.eps1eps2 - r_hd3.0[2]).abs() < 1e-10);
    }

    // #[test]
    // fn test_third_derivative() {
    //     // let (t_hds, mut v_hds, m_hds) = init_state::<HDScal64<D2>>();
    //     // v_hds = v_hds.derive();
    //     let (t_hd3, mut v_hd3, m_hd3) = init_state::<HD3_64>();
    //     v_hd3 = v_hd3.derive();

    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };

    //     // let r_hds = hs.helmholtz_energy(t_hds, v_hds, &m_hds);
    //     let r_hd3 = hs.helmholtz_energy(t_hd3, v_hd3, &m_hd3);
    //     // assert!((r_hds.0[2] - r_hd3.0[2]).abs() < 1e-10);
    // }

    //     #[bench]
    //     fn bench_dual(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<Dual64>();
    //         t_d.eps = 1.0;
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hyperdual(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HyperDual64>();
    //         t_d.eps1 = 1.0;
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd_scal_1(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HDScal64<D1>>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd_scal_2(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HDScal64<D2>>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd_scal_3(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HDScal64<D3>>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd_scal_4(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HDScal64<D4>>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd_scal_5(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HDScal64<D5>>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }

    //     #[bench]
    //     fn bench_hd3(b: &mut Bencher) {
    //         let (mut t_d, v_d, m_d) = init_state::<HD3_64>();
    //         t_d = t_d.derive();
    //         let hs = HSContribution {
    //             m: arr1(&[1.0, 2.5]),
    //             sigma: arr1(&[3.2, 3.5]),
    //             epsilon_k: arr1(&[150., 220.]),
    //         };
    //         b.iter(|| hs.helmholtz_energy(&t_d, &v_d, &m_d));
    //     }
}
