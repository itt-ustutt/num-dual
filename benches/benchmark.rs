use criterion::{criterion_group, criterion_main, Criterion};
use num_hyperdual::*;

trait HelmholtzEnergy<const N: usize> {
    fn helmholtz_energy<T: DualNum<f64>>(
        &self,
        temperature: T,
        volume: T,
        moles: StaticVec<T, N>,
    ) -> T;
}

struct HSContribution<const N: usize> {
    m: StaticVec<f64, N>,
    sigma: StaticVec<f64, N>,
    epsilon_k: StaticVec<f64, N>,
}

impl<const N: usize> HelmholtzEnergy<N> for HSContribution<N> {
    fn helmholtz_energy<T: DualNum<f64>>(
        &self,
        temperature: T,
        volume: T,
        moles: StaticVec<T, N>,
    ) -> T {
        let vi = volume.recip();
        let density = moles * vi;
        let ti = temperature.recip() * -3.0;
        let d = self
            .epsilon_k
            .map_zip(&self.sigma, |e, s| -((ti * e).exp() * 0.12 - 1.0) * s);
        let mut zeta: [T; 4] = [T::zero(), T::zero(), T::zero(), T::zero()];
        let mut m_rho: T = T::zero();
        for i in 0..N {
            for k in 0..4 {
                zeta[k] = zeta[k]
                    + density[i] * d[i].powi(k as i32) * (std::f64::consts::PI / 6.0 * self.m[i]);
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

fn bench<T: DualNum<f64>>() -> T {
    let temperature = T::from(300.0);
    let volume = T::from(1.0);
    let moles = StaticVec::new_vec([T::from(0.001), T::from(0.005)]);
    let hs = HSContribution {
        m: StaticVec::new_vec([1.0, 2.5]),
        sigma: StaticVec::new_vec([3.2, 3.5]),
        epsilon_k: StaticVec::new_vec([150., 220.]),
    };
    hs.helmholtz_energy(temperature, volume, moles)
}

fn criterion_benchmark(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("Hard sphere contribution");
        group.bench_function("f64", |b| b.iter(|| bench::<f64>()));
        group.bench_function("Dual64", |b| b.iter(|| bench::<Dual64>()));
        group.bench_function("HyperDual64", |b| b.iter(|| bench::<HyperDual64>()));
        group.bench_function("HD3_64", |b| b.iter(|| bench::<HD3_64>()));
        group.bench_function("DualN64<1>", |b| b.iter(|| bench::<DualN64<1>>()));
        group.bench_function("DualN64<2>", |b| b.iter(|| bench::<DualN64<2>>()));
        group.bench_function("DualN64<3>", |b| b.iter(|| bench::<DualN64<3>>()));
        group.bench_function("HyperDualN64<1>", |b| b.iter(|| bench::<HyperDualN64<1>>()));
        group.bench_function("HyperDualN64<2>", |b| b.iter(|| bench::<HyperDualN64<2>>()));
        group.bench_function("HyperDualN64<3>", |b| b.iter(|| bench::<HyperDualN64<3>>()));
        group.bench_function("HyperDualMN64<1,1>", |b| {
            b.iter(|| bench::<HyperDualMN64<1, 1>>())
        });
        group.bench_function("HyperDualDual64", |b| b.iter(|| bench::<HyperDualDual64>()));
        group.bench_function("HD3Dual64", |b| b.iter(|| bench::<HD3Dual64>()));
    }
    let mut group = c.benchmark_group("Recursive numbers");
    group.bench_function("HyperDualDualN64<2>", |b| {
        b.iter(|| bench::<HyperDualDualN64<2>>())
    });
    group.bench_function("HD3DualN64<2>", |b| b.iter(|| bench::<HD3DualN64<2>>()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
