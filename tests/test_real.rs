use num_dual::*;

#[test]
fn test_real_recip() {
    let res = Real::<f64, f64>::from(1.2).recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
}

#[test]
fn test_real_exp() {
    let res = Real::<f64, f64>::from(1.2).exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_real_exp_m1() {
    let res = Real::<f64, f64>::from(1.2).exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
}

#[test]
fn test_real_exp2() {
    let res = Real::<f64, f64>::from(1.2).exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
}

#[test]
fn test_real_ln() {
    let res = Real::<f64, f64>::from(1.2).ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
}

#[test]
fn test_real_log() {
    let res = Real::<f64, f64>::from(1.2).log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
}

#[test]
fn test_real_ln_1p() {
    let res = Real::<f64, f64>::from(1.2).ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
}

#[test]
fn test_real_log2() {
    let res = Real::<f64, f64>::from(1.2).log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
}

#[test]
fn test_real_log10() {
    let res = Real::<f64, f64>::from(1.2).log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
}

#[test]
fn test_real_sqrt() {
    let res = Real::<f64, f64>::from(1.2).sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
}

#[test]
fn test_real_cbrt() {
    let res = Real::<f64, f64>::from(1.2).cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
}

#[test]
fn test_real_powf() {
    let res = Real::<f64, f64>::from(1.2).powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
}

#[test]
fn test_real_powf_0() {
    let res = Real::<f64, f64>::from(0.0).powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_real_powf_1() {
    let res = Real::<f64, f64>::from(0.0).powf(1.0);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powf_2() {
    let res = Real::<f64, f64>::from(0.0).powf(2.0);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powf_3() {
    let res = Real::<f64, f64>::from(0.0).powf(3.0);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powf_4() {
    let res = Real::<f64, f64>::from(0.0).powf(4.0);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powi() {
    let res = Real::<f64, f64>::from(1.2).powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
}

#[test]
fn test_real_powi_0() {
    let res = Real::<f64, f64>::from(0.0).powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_real_powi_1() {
    let res = Real::<f64, f64>::from(0.0).powi(1);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powi_2() {
    let res = Real::<f64, f64>::from(0.0).powi(2);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powi_3() {
    let res = Real::<f64, f64>::from(0.0).powi(3);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_powi_4() {
    let res = Real::<f64, f64>::from(0.0).powi(4);
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_sin() {
    let res = Real::<f64, f64>::from(1.2).sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_real_cos() {
    let res = Real::<f64, f64>::from(1.2).cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_real_tan() {
    let res = Real::<f64, f64>::from(1.2).tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
}

#[test]
fn test_real_asin() {
    let res = Real::<f64, f64>::from(0.2).asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
}

#[test]
fn test_real_acos() {
    let res = Real::<f64, f64>::from(0.2).acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
}

#[test]
fn test_real_atan() {
    let res = Real::<f64, f64>::from(0.2).atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
}

#[test]
fn test_real_atan2_1() {
    let res = Real::<f64, f64>::from(0.2).atan2((0.4).into());
    assert!((res.re - 0.463647609000806).abs() < 1e-12);
}

#[test]
fn test_real_atan2_2() {
    let res = Real::<f64, f64>::from(-0.2).atan2((0.4).into());
    assert!((res.re - -0.463647609000806).abs() < 1e-12);
}

#[test]
fn test_real_atan2_3() {
    let res = Real::<f64, f64>::from(0.2).atan2((-0.4).into());
    assert!((res.re - 2.67794504458899).abs() < 1e-12);
}

#[test]
fn test_real_atan2_4() {
    let res = Real::<f64, f64>::from(-0.2).atan2((-0.4).into());
    assert!((res.re - -2.67794504458899).abs() < 1e-12);
}

#[test]
fn test_real_sinh() {
    let res = Real::<f64, f64>::from(1.2).sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_real_cosh() {
    let res = Real::<f64, f64>::from(1.2).cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_real_tanh() {
    let res = Real::<f64, f64>::from(1.2).tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
}

#[test]
fn test_real_asinh() {
    let res = Real::<f64, f64>::from(1.2).asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
}

#[test]
fn test_real_acosh() {
    let res = Real::<f64, f64>::from(1.2).acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
}

#[test]
fn test_real_atanh() {
    let res = Real::<f64, f64>::from(0.2).atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
}

#[test]
fn test_real_sph_j0() {
    let res = Real::<f64, f64>::from(1.2).sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
}

#[test]
fn test_real_sph_j1() {
    let res = Real::<f64, f64>::from(1.2).sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
}

#[test]
fn test_real_sph_j2() {
    let res = Real::<f64, f64>::from(1.2).sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j0_0() {
    let res = Real::<f64, f64>::from(0.0).bessel_j0();
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j1_0() {
    let res = Real::<f64, f64>::from(0.0).bessel_j1();
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j2_0() {
    let res = Real::<f64, f64>::from(0.0).bessel_j2();
    assert!((res.re).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j0_1() {
    let res = Real::<f64, f64>::from(1.2).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j1_1() {
    let res = Real::<f64, f64>::from(1.2).bessel_j1();
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j2_1() {
    let res = Real::<f64, f64>::from(1.2).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j0_2() {
    let res = Real::<f64, f64>::from(7.2).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j1_2() {
    let res = Real::<f64, f64>::from(7.2).bessel_j1();
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j2_2() {
    let res = Real::<f64, f64>::from(7.2).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j0_3() {
    let res = Real::<f64, f64>::from(-1.2).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j1_3() {
    let res = Real::<f64, f64>::from(-1.2).bessel_j1();
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j2_3() {
    let res = Real::<f64, f64>::from(-1.2).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j0_4() {
    let res = Real::<f64, f64>::from(-7.2).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j1_4() {
    let res = Real::<f64, f64>::from(-7.2).bessel_j1();
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_real_bessel_j2_4() {
    let res = Real::<f64, f64>::from(-7.2).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
}
