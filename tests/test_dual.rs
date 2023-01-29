use num_dual::*;

#[test]
fn test_dual_recip() {
    let res = Dual64::new_scalar(1.2, 1.0).recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps[0] - -0.694444444444445).abs() < 1e-12);
}

#[test]
fn test_dual_exp() {
    let res = Dual64::new_scalar(1.2, 1.0).exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps[0] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_exp_m1() {
    let res = Dual64::new_scalar(1.2, 1.0).exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.eps[0] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_exp2() {
    let res = Dual64::new_scalar(1.2, 1.0).exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.eps[0] - 1.59243405216008).abs() < 1e-12);
}

#[test]
fn test_dual_ln() {
    let res = Dual64::new_scalar(1.2, 1.0).ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.eps[0] - 0.833333333333333).abs() < 1e-12);
}

#[test]
fn test_dual_log() {
    let res = Dual64::new_scalar(1.2, 1.0).log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.eps[0] - 0.580685888982970).abs() < 1e-12);
}

#[test]
fn test_dual_ln_1p() {
    let res = Dual64::new_scalar(1.2, 1.0).ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.eps[0] - 0.454545454545455).abs() < 1e-12);
}

#[test]
fn test_dual_log2() {
    let res = Dual64::new_scalar(1.2, 1.0).log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.eps[0] - 1.20224586740747).abs() < 1e-12);
}

#[test]
fn test_dual_log10() {
    let res = Dual64::new_scalar(1.2, 1.0).log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.eps[0] - 0.361912068252710).abs() < 1e-12);
}

#[test]
fn test_dual_sqrt() {
    let res = Dual64::new_scalar(1.2, 1.0).sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.eps[0] - 0.456435464587638).abs() < 1e-12);
}

#[test]
fn test_dual_cbrt() {
    let res = Dual64::new_scalar(1.2, 1.0).cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.eps[0] - 0.295182935884059).abs() < 1e-12);
}

#[test]
fn test_dual_powf() {
    let res = Dual64::new_scalar(1.2, 1.0).powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.eps[0] - 7.52712759108966).abs() < 1e-12);
}

#[test]
fn test_dual_powf_0() {
    let res = Dual64::new_scalar(0.0, 1.0).powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powf_1() {
    let res = Dual64::new_scalar(0.0, 1.0).powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0] - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_powf_2() {
    let res = Dual64::new_scalar(0.0, 1.0).powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powf_3() {
    let res = Dual64::new_scalar(0.0, 1.0).powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powf_4() {
    let res = Dual64::new_scalar(0.0, 1.0).powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powi() {
    let res = Dual64::new_scalar(1.2, 1.0).powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.eps[0] - 14.9299200000000).abs() < 1e-12);
}

#[test]
fn test_dual_powi_0() {
    let res = Dual64::new_scalar(0.0, 1.0).powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powi_1() {
    let res = Dual64::new_scalar(0.0, 1.0).powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0] - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_powi_2() {
    let res = Dual64::new_scalar(0.0, 1.0).powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powi_3() {
    let res = Dual64::new_scalar(0.0, 1.0).powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_powi_4() {
    let res = Dual64::new_scalar(0.0, 1.0).powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_sin() {
    let res = Dual64::new_scalar(1.2, 1.0).sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.eps[0] - 0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_dual_cos() {
    let res = Dual64::new_scalar(1.2, 1.0).cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps[0] - -0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_dual_tan() {
    let res = Dual64::new_scalar(1.2, 1.0).tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.eps[0] - 7.61596396720705).abs() < 1e-12);
}

#[test]
fn test_dual_asin() {
    let res = Dual64::new_scalar(0.2, 1.0).asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.eps[0] - 1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_acos() {
    let res = Dual64::new_scalar(0.2, 1.0).acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.eps[0] - -1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_atan() {
    let res = Dual64::new_scalar(0.2, 1.0).atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.eps[0] - 0.961538461538462).abs() < 1e-12);
}

#[test]
fn test_dual_sinh() {
    let res = Dual64::new_scalar(1.2, 1.0).sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps[0] - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_dual_cosh() {
    let res = Dual64::new_scalar(1.2, 1.0).cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps[0] - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_dual_tanh() {
    let res = Dual64::new_scalar(1.2, 1.0).tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.eps[0] - 0.305019996207409).abs() < 1e-12);
}

#[test]
fn test_dual_asinh() {
    let res = Dual64::new_scalar(1.2, 1.0).asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.eps[0] - 0.640184399664480).abs() < 1e-12);
}

#[test]
fn test_dual_acosh() {
    let res = Dual64::new_scalar(1.2, 1.0).acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.eps[0] - 1.50755672288882).abs() < 1e-12);
}

#[test]
fn test_dual_atanh() {
    let res = Dual64::new_scalar(0.2, 1.0).atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.eps[0] - 1.04166666666667).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j0() {
    let res = Dual64::new_scalar(1.2, 1.0).sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.eps[0] - -0.345284569857790).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j1() {
    let res = Dual64::new_scalar(1.2, 1.0).sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.eps[0] - 0.201224955209705).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j2() {
    let res = Dual64::new_scalar(1.2, 1.0).sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.eps[0] - 0.129004104011656).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_0() {
    let res = Dual64::new_scalar(0.0, 1.0).bessel_j0();
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_0() {
    let res = Dual64::new_scalar(0.0, 1.0).bessel_j1();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0] - 0.500000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_0() {
    let res = Dual64::new_scalar(0.0, 1.0).bessel_j2();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps[0]).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_1() {
    let res = Dual64::new_scalar(1.2, 1.0).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps[0] - -0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_1() {
    let res = Dual64::new_scalar(1.2, 1.0).bessel_j1();
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps[0] - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_1() {
    let res = Dual64::new_scalar(1.2, 1.0).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps[0] - 0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_2() {
    let res = Dual64::new_scalar(7.2, 1.0).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps[0] - -0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_2() {
    let res = Dual64::new_scalar(7.2, 1.0).bessel_j1();
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps[0] - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_2() {
    let res = Dual64::new_scalar(7.2, 1.0).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps[0] - 0.132099570594364).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_3() {
    let res = Dual64::new_scalar(-1.2, 1.0).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps[0] - 0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_3() {
    let res = Dual64::new_scalar(-1.2, 1.0).bessel_j1();
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps[0] - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_3() {
    let res = Dual64::new_scalar(-1.2, 1.0).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps[0] - -0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_4() {
    let res = Dual64::new_scalar(-7.2, 1.0).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps[0] - 0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_4() {
    let res = Dual64::new_scalar(-7.2, 1.0).bessel_j1();
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps[0] - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_4() {
    let res = Dual64::new_scalar(-7.2, 1.0).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps[0] - -0.132099570594364).abs() < 1e-12);
}
