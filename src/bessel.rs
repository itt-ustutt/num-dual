#![allow(clippy::excessive_precision)]
use crate::DualNum;
use num_traits::{Float, Zero};
use std::f64::consts::{FRAC_2_PI, FRAC_PI_4};

/// Implementation of bessel functions for double precision (hyper) dual numbers.
pub trait BesselDual: DualNum<f64> + Copy {
    /// 0th order bessel function of the first kind
    fn bessel_j0(mut self) -> Self {
        if self.is_negative() {
            self = -self;
        }

        if self.re() <= 5.0 {
            let z = self * self;
            if self.re() < 1.0e-5 {
                return Self::one() - z / 4.0;
            }

            (z - DR1) * (z - DR2) * polevl(z, &RP0) / p1evl(z, &RQ0)
        } else {
            let w = self.recip() * 5.0;
            let q = w * w;
            let p = polevl(q, &PP0) / polevl(q, &PQ0);
            let q = polevl(q, &QP0) / p1evl(q, &QQ0);
            let (s, c) = (self - FRAC_PI_4).sin_cos();
            let p = p * c - w * q * s;
            p * (Self::from(FRAC_2_PI) / self).sqrt()
        }
    }

    /// 1st order bessel function of the first kind
    fn bessel_j1(self) -> Self {
        let x = self.abs();
        if x.re() <= 5.0 {
            {
                let z = self * self;
                polevl(z, &RP1) / p1evl(z, &RQ1) * self * (z - Z1) * (z - Z2)
            }
        } else {
            let x = self.abs();
            let w = x.recip() * 5.0;
            let z = w * w;
            let p = polevl(z, &PP1) / polevl(z, &PQ1);
            let q = polevl(z, &QP1) / p1evl(z, &QQ1);
            let (s, c) = (x - 3.0 * FRAC_PI_4).sin_cos();
            let p = p * c - w * q * s;
            self.signum() * p * (Self::from(FRAC_2_PI) / x).sqrt()
        }
    }

    /// 2nd order bessel function of the first kind
    fn bessel_j2(self) -> Self {
        if self.re().is_zero() {
            self * self / 8.0 * (self * self / 24.0 + 1.0)
        } else {
            self.bessel_j1() * 2.0 / self - self.bessel_j0()
        }
    }
}

impl<T: DualNum<f64> + Copy> BesselDual for T {}

const DR1: f64 = 5.78318596294678452118E0;
const DR2: f64 = 3.04712623436620863991E1;

const RP0: [f64; 4] = [
    -4.79443220978201773821E9,
    1.95617491946556577543E12,
    -2.49248344360967716204E14,
    9.70862251047306323952E15,
];
const RQ0: [f64; 8] = [
    4.99563147152651017219E2,
    1.73785401676374683123E5,
    4.84409658339962045305E7,
    1.11855537045356834862E10,
    2.11277520115489217587E12,
    3.10518229857422583814E14,
    3.18121955943204943306E16,
    1.71086294081043136091E18,
];

const PP0: [f64; 7] = [
    7.96936729297347051624E-4,
    8.28352392107440799803E-2,
    1.23953371646414299388E0,
    5.44725003058768775090E0,
    8.74716500199817011941E0,
    5.30324038235394892183E0,
    9.99999999999999997821E-1,
];
const PQ0: [f64; 7] = [
    9.24408810558863637013E-4,
    8.56288474354474431428E-2,
    1.25352743901058953537E0,
    5.47097740330417105182E0,
    8.76190883237069594232E0,
    5.30605288235394617618E0,
    1.00000000000000000218E0,
];

const QP0: [f64; 8] = [
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
];
const QQ0: [f64; 7] = [
    6.43178256118178023184E1,
    8.56430025976980587198E2,
    3.88240183605401609683E3,
    7.24046774195652478189E3,
    5.93072701187316984827E3,
    2.06209331660327847417E3,
    2.42005740240291393179E2,
];

const Z1: f64 = 1.46819706421238932572E1;
const Z2: f64 = 4.92184563216946036703E1;

const RP1: [f64; 4] = [
    -8.99971225705559398224E8,
    4.52228297998194034323E11,
    -7.27494245221818276015E13,
    3.68295732863852883286E15,
];
const RQ1: [f64; 8] = [
    6.20836478118054335476E2,
    2.56987256757748830383E5,
    8.35146791431949253037E7,
    2.21511595479792499675E10,
    4.74914122079991414898E12,
    7.84369607876235854894E14,
    8.95222336184627338078E16,
    5.32278620332680085395E18,
];

const PP1: [f64; 7] = [
    7.62125616208173112003E-4,
    7.31397056940917570436E-2,
    1.12719608129684925192E0,
    5.11207951146807644818E0,
    8.42404590141772420927E0,
    5.21451598682361504063E0,
    1.00000000000000000254E0,
];
const PQ1: [f64; 7] = [
    5.71323128072548699714E-4,
    6.88455908754495404082E-2,
    1.10514232634061696926E0,
    5.07386386128601488557E0,
    8.39985554327604159757E0,
    5.20982848682361821619E0,
    9.99999999999999997461E-1,
];

const QP1: [f64; 8] = [
    5.10862594750176621635E-2,
    4.98213872951233449420E0,
    7.58238284132545283818E1,
    3.66779609360150777800E2,
    7.10856304998926107277E2,
    5.97489612400613639965E2,
    2.11688757100572135698E2,
    2.52070205858023719784E1,
];
const QQ1: [f64; 7] = [
    7.42373277035675149943E1,
    1.05644886038262816351E3,
    4.98641058337653607651E3,
    9.56231892404756170795E3,
    7.99704160447350683650E3,
    2.82619278517639096600E3,
    3.36093607810698293419E2,
];

fn polevl<T: DualNum<F> + Copy, F: Float>(x: T, coef: &[F]) -> T {
    coef.iter()
        .skip(1)
        .fold(T::from(coef[0]), |acc, &c| acc * x + c)
}

fn p1evl<T: DualNum<F> + Copy, F: Float>(x: T, coef: &[F]) -> T {
    coef.iter().fold(T::one(), |acc, &c| acc * x + c)
}
