use nalgebra::{Const, Vector};
use num_dual::*;

#[test]
fn test_dual_vec_recip() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).recip();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((eps[0] - -0.694444444444445).abs() < 1e-12);
    assert!((eps[1] - -0.694444444444445).abs() < 1e-12);
}

#[test]
fn test_dual_vec_exp() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).exp();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((eps[0] - 3.32011692273655).abs() < 1e-12);
    assert!((eps[1] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_vec_exp_m1() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).exp_m1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((eps[0] - 3.32011692273655).abs() < 1e-12);
    assert!((eps[1] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_vec_exp2() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).exp2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((eps[0] - 1.59243405216008).abs() < 1e-12);
    assert!((eps[1] - 1.59243405216008).abs() < 1e-12);
}

#[test]
fn test_dual_vec_ln() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).ln();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((eps[0] - 0.833333333333333).abs() < 1e-12);
    assert!((eps[1] - 0.833333333333333).abs() < 1e-12);
}

#[test]
fn test_dual_vec_log() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).log(4.2);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((eps[0] - 0.580685888982970).abs() < 1e-12);
    assert!((eps[1] - 0.580685888982970).abs() < 1e-12);
}

#[test]
fn test_dual_vec_ln_1p() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).ln_1p();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((eps[0] - 0.454545454545455).abs() < 1e-12);
    assert!((eps[1] - 0.454545454545455).abs() < 1e-12);
}

#[test]
fn test_dual_vec_log2() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).log2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((eps[0] - 1.20224586740747).abs() < 1e-12);
    assert!((eps[1] - 1.20224586740747).abs() < 1e-12);
}

#[test]
fn test_dual_vec_log10() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).log10();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((eps[0] - 0.361912068252710).abs() < 1e-12);
    assert!((eps[1] - 0.361912068252710).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sqrt() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sqrt();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((eps[0] - 0.456435464587638).abs() < 1e-12);
    assert!((eps[1] - 0.456435464587638).abs() < 1e-12);
}

#[test]
fn test_dual_vec_cbrt() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).cbrt();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((eps[0] - 0.295182935884059).abs() < 1e-12);
    assert!((eps[1] - 0.295182935884059).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).powf(4.2);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((eps[0] - 7.52712759108966).abs() < 1e-12);
    assert!((eps[1] - 7.52712759108966).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf_0() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powf(0.0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf_1() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powf(1.0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0] - 1.00000000000000).abs() < 1e-12);
    assert!((eps[1] - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf_2() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powf(2.0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf_3() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powf(3.0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powf_4() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powf(4.0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).powi(6);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((eps[0] - 14.9299200000000).abs() < 1e-12);
    assert!((eps[1] - 14.9299200000000).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi_0() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powi(0);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi_1() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powi(1);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0] - 1.00000000000000).abs() < 1e-12);
    assert!((eps[1] - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi_2() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powi(2);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi_3() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powi(3);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_powi_4() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).powi(4);
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sin() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sin();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((eps[0] - 0.362357754476674).abs() < 1e-12);
    assert!((eps[1] - 0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_dual_vec_cos() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).cos();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((eps[0] - -0.932039085967226).abs() < 1e-12);
    assert!((eps[1] - -0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_dual_vec_tan() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).tan();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((eps[0] - 7.61596396720705).abs() < 1e-12);
    assert!((eps[1] - 7.61596396720705).abs() < 1e-12);
}

#[test]
fn test_dual_vec_asin() {
    let res = DualSVec64::new(0.2, Derivative::some(Vector::from([1.0, 1.0]))).asin();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((eps[0] - 1.02062072615966).abs() < 1e-12);
    assert!((eps[1] - 1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_vec_acos() {
    let res = DualSVec64::new(0.2, Derivative::some(Vector::from([1.0, 1.0]))).acos();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((eps[0] - -1.02062072615966).abs() < 1e-12);
    assert!((eps[1] - -1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_vec_atan() {
    let res = DualSVec64::new(0.2, Derivative::some(Vector::from([1.0, 1.0]))).atan();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((eps[0] - 0.961538461538462).abs() < 1e-12);
    assert!((eps[1] - 0.961538461538462).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sinh() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sinh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((eps[0] - 1.81065556732437).abs() < 1e-12);
    assert!((eps[1] - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_dual_vec_cosh() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).cosh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((eps[0] - 1.50946135541217).abs() < 1e-12);
    assert!((eps[1] - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_dual_vec_tanh() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).tanh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((eps[0] - 0.305019996207409).abs() < 1e-12);
    assert!((eps[1] - 0.305019996207409).abs() < 1e-12);
}

#[test]
fn test_dual_vec_asinh() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).asinh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((eps[0] - 0.640184399664480).abs() < 1e-12);
    assert!((eps[1] - 0.640184399664480).abs() < 1e-12);
}

#[test]
fn test_dual_vec_acosh() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).acosh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((eps[0] - 1.50755672288882).abs() < 1e-12);
    assert!((eps[1] - 1.50755672288882).abs() < 1e-12);
}

#[test]
fn test_dual_vec_atanh() {
    let res = DualSVec64::new(0.2, Derivative::some(Vector::from([1.0, 1.0]))).atanh();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((eps[0] - 1.04166666666667).abs() < 1e-12);
    assert!((eps[1] - 1.04166666666667).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sph_j0() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sph_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((eps[0] - -0.345284569857790).abs() < 1e-12);
    assert!((eps[1] - -0.345284569857790).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sph_j1() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sph_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((eps[0] - 0.201224955209705).abs() < 1e-12);
    assert!((eps[1] - 0.201224955209705).abs() < 1e-12);
}

#[test]
fn test_dual_vec_sph_j2() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).sph_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((eps[0] - 0.129004104011656).abs() < 1e-12);
    assert!((eps[1] - 0.129004104011656).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j0_0() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j1_0() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0] - 0.500000000000000).abs() < 1e-12);
    assert!((eps[1] - 0.500000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j2_0() {
    let res = DualSVec64::new(0.0, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re).abs() < 1e-12);
    assert!((eps[0]).abs() < 1e-12);
    assert!((eps[1]).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j0_1() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((eps[0] - -0.498289057567215).abs() < 1e-12);
    assert!((eps[1] - -0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j1_1() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
    assert!((eps[0] - 0.255891862958350).abs() < 1e-12);
    assert!((eps[1] - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j2_1() {
    let res = DualSVec64::new(1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((eps[0] - 0.232707360321110).abs() < 1e-12);
    assert!((eps[1] - 0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j0_2() {
    let res = DualSVec64::new(7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((eps[0] - -0.0543274202223671).abs() < 1e-12);
    assert!((eps[1] - -0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j1_2() {
    let res = DualSVec64::new(7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
    assert!((eps[0] - 0.287525216370074).abs() < 1e-12);
    assert!((eps[1] - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j2_2() {
    let res = DualSVec64::new(7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((eps[0] - 0.132099570594364).abs() < 1e-12);
    assert!((eps[1] - 0.132099570594364).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j0_3() {
    let res = DualSVec64::new(-1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((eps[0] - 0.498289057567215).abs() < 1e-12);
    assert!((eps[1] - 0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j1_3() {
    let res = DualSVec64::new(-1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
    assert!((eps[0] - 0.255891862958350).abs() < 1e-12);
    assert!((eps[1] - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j2_3() {
    let res = DualSVec64::new(-1.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((eps[0] - -0.232707360321110).abs() < 1e-12);
    assert!((eps[1] - -0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j0_4() {
    let res = DualSVec64::new(-7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j0();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((eps[0] - 0.0543274202223671).abs() < 1e-12);
    assert!((eps[1] - 0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j1_4() {
    let res = DualSVec64::new(-7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j1();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
    assert!((eps[0] - 0.287525216370074).abs() < 1e-12);
    assert!((eps[1] - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_vec_bessel_j2_4() {
    let res = DualSVec64::new(-7.2, Derivative::some(Vector::from([1.0, 1.0]))).bessel_j2();
    let eps = res.eps.unwrap_generic(Const::<2>, Const::<1>);
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((eps[0] - -0.132099570594364).abs() < 1e-12);
    assert!((eps[1] - -0.132099570594364).abs() < 1e-12);
}
