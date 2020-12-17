#![feature(const_generics)]
#![feature(test)]
extern crate test;

use num_traits::{FromPrimitive, Inv, NumAssignOps, NumOps, Signed, Zero};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, Neg, Sub};
#[macro_use]
mod macros;
pub mod dual;
pub mod hd3;
pub mod hd_scal;
pub mod hyperdual;
// pub mod linalg;
pub mod static_vec;
pub use dual::*;
pub use hd3::*;
pub use hd_scal::*;
pub use hyperdual::*;

pub trait DualNum<F>:
    DualNumMethods<F>
    + Signed
    + NumAssignOps
    + NumAssignOps<F>
    + Copy
    + Inv<Output = Self>
    + Sum
    + Product
    + FromPrimitive
    + From<F>
    + fmt::Display
    + Sync
    + Send
    + 'static
{
}
impl<D, F> DualNum<F> for D where
    D: DualNumMethods<F>
        + Signed
        + NumAssignOps
        + NumAssignOps<F>
        + Copy
        + Inv<Output = Self>
        + Sum
        + Product
        + FromPrimitive
        + From<F>
        + fmt::Display
        + Sync
        + Send
        + 'static
{
}

pub trait DualVec<T, F>:
    Copy
    + Zero
    + NumOps<T, Self>
    + NumOps<F, Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + PartialEq
{
}

impl<T, F, D> DualVec<T, F> for D where
    D: Copy
        + Zero
        + NumOps<T, Self>
        + NumOps<F, Self>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Neg<Output = Self>
        + PartialEq
{
}

pub trait DualNumMethods<F>: Clone + NumOps<Self> + NumOps<F> {
    /// indicates the highest derivative that can be calculated with this struct
    const NDERIV: usize;

    /// returns the real part (the 0th derivative) of the number
    fn re(&self) -> F;

    fn recip(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, n: F) -> Self;
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn exp2(&self) -> Self;
    fn exp_m1(&self) -> Self;
    fn ln(&self) -> Self;
    fn log(&self, base: F) -> Self;
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
            const NDERIV: usize = 0;

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
    use super::dual::{Dual64, DualN64};
    use super::hd3::HD3_64;
    use super::hyperdual::{HyperDual64, HyperDualDual64};
    use super::*;
    use ndarray::*;
    use test::Bencher;

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
        let (t_d, v_d, m_d) = init_state::<Dual64>();
        let (t_hd, v_hd, m_hd) = init_state::<HyperDual64>();
        let (t_dn, v_dn, m_dn) = init_state::<DualN64<2>>();
        let (t_hd3, v_hd3, m_hd3) = init_state::<HD3_64>();

        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };

        let r_dt = hs.helmholtz_energy(t_d.derive(), v_d, &m_d);
        let r_dv = hs.helmholtz_energy(t_d, v_d.derive(), &m_d);
        let r_hd = hs.helmholtz_energy(t_hd.derive1(), v_hd.derive2(), &m_hd);
        let r_dn = hs.helmholtz_energy(t_dn.derive(0), v_dn.derive(1), &m_dn);
        let r_hd3 = hs.helmholtz_energy(t_hd3.derive(), v_hd3, &m_hd3);

        assert_eq!(r_dt.eps, r_hd.eps1);
        assert_eq!(r_dv.eps, r_hd.eps2);
        assert_eq!(r_dt.eps, r_dn.eps[0]);
        assert_eq!(r_dv.eps, r_dn.eps[1]);
        assert_eq!(r_dt.eps, r_hd3.0[1]);
    }

    #[test]
    fn test_second_derivative() {
        let (t_hd, v_hd, m_hd) = init_state::<HyperDual64>();
        let (t_hdn, v_hdn, m_hdn) = init_state::<HyperDualN64<2>>();
        let (t_hd3, v_hd3, m_hd3) = init_state::<HD3_64>();

        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };

        let r_hd_tt = hs.helmholtz_energy(t_hd.derive1().derive2(), v_hd, &m_hd);
        let r_hd_tv = hs.helmholtz_energy(t_hd.derive1(), v_hd.derive2(), &m_hd);
        let r_hd_vv = hs.helmholtz_energy(t_hd, v_hd.derive1().derive2(), &m_hd);
        let r_hdn = hs.helmholtz_energy(t_hdn.derive(0), v_hdn.derive(1), &m_hdn);
        let r_hd3_ttt = hs.helmholtz_energy(t_hd3.derive(), v_hd3, &m_hd3);
        let r_hd3_vvv = hs.helmholtz_energy(t_hd3, v_hd3.derive(), &m_hd3);

        assert!((r_hd_tt.eps1eps2 - r_hd3_ttt.0[2]).abs() < 1e-10);
        assert!((r_hd_vv.eps1eps2 - r_hd3_vvv.0[2]).abs() < 1e-10);
        assert!((r_hd_tt.eps1eps2 - r_hdn.eps1eps2[(0, 0)]).abs() < 1e-10);
        assert!((r_hd_tv.eps1eps2 - r_hdn.eps1eps2[(0, 1)]).abs() < 1e-10);
        assert!((r_hd_vv.eps1eps2 - r_hdn.eps1eps2[(1, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_third_derivative() {
        let (t_hdd, mut v_hdd, m_hdd) = init_state::<HyperDualDual64>();
        v_hdd.re.eps = 1.0;
        v_hdd.eps1.re = 1.0;
        v_hdd.eps2.re = 1.0;
        let (t_hd3, mut v_hd3, m_hd3) = init_state::<HD3_64>();
        v_hd3 = v_hd3.derive();

        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        let r_hdd = hs.helmholtz_energy(t_hdd, v_hdd, &m_hdd);
        let r_hd3 = hs.helmholtz_energy(t_hd3, v_hd3, &m_hd3);
        assert!((r_hdd.eps1eps2.eps - r_hd3.0[3]).abs() < 1e-10);
    }

    #[bench]
    fn bench_dual(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<Dual64>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive(), v_d, &m_d));
    }

    #[bench]
    fn bench_dual_1(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<DualN64<1>>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive(0), v_d, &m_d));
    }

    #[bench]
    fn bench_dual_2(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<DualN64<2>>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive(0), v_d.derive(1), &m_d));
    }

    #[bench]
    fn bench_hyperdual(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<HyperDual64>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive1(), v_d.derive2(), &m_d));
    }

    #[bench]
    fn bench_hyperdual_1(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<HyperDualN64<1>>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive(0), v_d, &m_d));
    }

    #[bench]
    fn bench_hyperdual_2(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<HyperDualN64<2>>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d.derive(0), v_d.derive(1), &m_d));
    }

    // #[bench]
    // fn bench_hd_scal_1(b: &mut Bencher) {
    //     let (mut t_d, v_d, m_d) = init_state::<HDScal64<D1>>();
    //     t_d = t_d.derive();
    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };
    //     b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    // }

    // #[bench]
    // fn bench_hd_scal_2(b: &mut Bencher) {
    //     let (mut t_d, v_d, m_d) = init_state::<HDScal64<D2>>();
    //     t_d = t_d.derive();
    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };
    //     b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    // }

    // #[bench]
    // fn bench_hd_scal_3(b: &mut Bencher) {
    //     let (mut t_d, v_d, m_d) = init_state::<HDScal64<D3>>();
    //     t_d = t_d.derive();
    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };
    //     b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    // }

    // #[bench]
    // fn bench_hd_scal_4(b: &mut Bencher) {
    //     let (mut t_d, v_d, m_d) = init_state::<HDScal64<D4>>();
    //     t_d = t_d.derive();
    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };
    //     b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    // }

    // #[bench]
    // fn bench_hd_scal_5(b: &mut Bencher) {
    //     let (mut t_d, v_d, m_d) = init_state::<HDScal64<D5>>();
    //     t_d = t_d.derive();
    //     let hs = HSContribution {
    //         m: arr1(&[1.0, 2.5]),
    //         sigma: arr1(&[3.2, 3.5]),
    //         epsilon_k: arr1(&[150., 220.]),
    //     };
    //     b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    // }

    #[bench]
    fn bench_hd3(b: &mut Bencher) {
        let (mut t_d, v_d, m_d) = init_state::<HD3_64>();
        t_d = t_d.derive();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    }

    #[bench]
    fn bench_hyperdualdual(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<HyperDualDual64>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    }

    #[bench]
    fn bench_hd3dual(b: &mut Bencher) {
        let (t_d, v_d, m_d) = init_state::<HD3Dual64>();
        let hs = HSContribution {
            m: arr1(&[1.0, 2.5]),
            sigma: arr1(&[3.2, 3.5]),
            epsilon_k: arr1(&[150., 220.]),
        };
        b.iter(|| hs.helmholtz_energy(t_d, v_d, &m_d));
    }
}
