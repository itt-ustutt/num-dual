use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{DVector, SVector};
use num_dual::*;

trait HelmholtzEnergy<const N: usize> {
    fn helmholtz_energy<T: DualNum<f64>>(
        &self,
        temperature: T,
        volume: T,
        moles: SVector<T, N>,
    ) -> T;
}

struct HSContribution<const N: usize> {
    m: SVector<f64, N>,
    sigma: SVector<f64, N>,
    epsilon_k: SVector<f64, N>,
}

impl<const N: usize> HelmholtzEnergy<N> for HSContribution<N> {
    fn helmholtz_energy<T: DualNum<f64>>(
        &self,
        temperature: T,
        volume: T,
        moles: SVector<T, N>,
    ) -> T {
        let vi = volume.recip();
        let density = moles * vi;
        let ti = temperature.recip() * -3.0;
        let d = self.epsilon_k.zip_map(&self.sigma, |e, s| {
            -((ti.clone() * e).exp() * 0.12 - 1.0) * s
        });
        let mut zeta: [T; 4] = [T::zero(), T::zero(), T::zero(), T::zero()];
        let mut m_rho: T = T::zero();
        for i in 0..N {
            for (k, z) in zeta.iter_mut().enumerate() {
                *z += density[i].clone()
                    * d[i].powi(k as i32)
                    * (std::f64::consts::PI / 6.0 * self.m[i]);
            }
            m_rho += density[i].clone() * self.m[i];
        }
        let frac_1mz3 = -(zeta[3].clone() - 1.0).recip();
        let frac_z3 = zeta[3].recip();
        volume
            * m_rho
            * zeta[0].recip()
            * (zeta[1].clone() * &zeta[2] * &frac_z3 * 3.0
                + zeta[2].powi(3) * frac_1mz3.powi(2) * &frac_z3
                + (zeta[2].powi(3) * frac_z3.powi(2) - &zeta[0]) * (-zeta[3].clone()).ln_1p())
    }
}

fn bench<T: DualNum<f64>>(one: T) -> T {
    let temperature = one.clone() * 300.0;
    let volume = one.clone() * 1.0;
    let moles = SVector::from([one.clone() * 0.001, one * 0.005]);
    let hs = HSContribution {
        m: SVector::from([1.0, 2.5]),
        sigma: SVector::from([3.2, 3.5]),
        epsilon_k: SVector::from([150., 220.]),
    };
    hs.helmholtz_energy(temperature, volume, moles)
}

fn criterion_benchmark(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("Hard sphere contribution");
        group.bench_function("f64", |b| b.iter(|| bench(1.0)));
        group.bench_function("Dual64", |b| b.iter(|| bench(Dual64::new(1.0, 1.0))));
        group.bench_function("DualVec64<2>", |b| {
            b.iter(|| {
                bench(DualVec::new(
                    1.0,
                    Derivative::some(SVector::from([1.0, 1.0])),
                ))
            })
        });
        group.bench_function("DualVec64<3>", |b| {
            b.iter(|| {
                bench(DualVec::new(
                    1.0,
                    Derivative::some(SVector::from([1.0, 1.0, 1.0])),
                ))
            })
        });
        group.bench_function("DualDVec64", |b| {
            b.iter(|| {
                bench(DualVec::new(
                    1.0,
                    Derivative::some(DVector::from_element(3, 1.0)),
                ))
            })
        });
        // group.bench_function("HyperDual64", |b| b.iter(bench::<HyperDual64>));
        // group.bench_function("HyperDualVec64<1,2>", |b| {
        //     b.iter(bench::<HyperDualSVec64<1, 2>>)
        // });
        // group.bench_function("HyperDualVec64<2,2>", |b| {
        //     b.iter(bench::<HyperDualSVec64<2, 2>>)
        // });
        // group.bench_function("Dual2_64", |b| b.iter(bench::<Dual2_64>));
        // group.bench_function("Dual2Vec64<2>", |b| b.iter(bench::<Dual2SVec64<2>>));
        // group.bench_function("Dual2Vec64<3>", |b| b.iter(bench::<Dual2SVec64<3>>));
        group.bench_function("Dual3_64", |b| {
            b.iter(|| bench(Dual3_64::new(1.0, 1.0, 0.0, 0.0)))
        });
    }
    // let mut group = c.benchmark_group("Recursive numbers");
    // group.bench_function("HyperDualDual64", |b| {
    //     b.iter(bench::<HyperDual<Dual64, f64>>)
    // });
    // group.bench_function("Dual3Dual64", |b| b.iter(bench::<Dual3<Dual64, f64>>));
    // group.bench_function("HyperDualDualVec64<2>", |b| {
    //     b.iter(bench::<HyperDual<DualSVec64<2>, f64>>)
    // });
    // group.bench_function("Dual3DualVec64<2>", |b| {
    //     b.iter(bench::<Dual3<DualSVec64<2>, f64>>)
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
