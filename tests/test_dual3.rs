use num_dual::*;

#[test]
fn test_dual3_recip() {
    let res = Dual3_64::from(1.2).derive().recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.v1 - -0.694444444444445).abs() < 1e-12);
    assert!((res.v2 - 1.15740740740741).abs() < 1e-12);
    assert!((res.v3 - -2.89351851851852).abs() < 1e-12);
}

#[test]
fn test_dual3_exp() {
    let res = Dual3_64::from(1.2).derive().exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.v1 - 3.32011692273655).abs() < 1e-12);
    assert!((res.v2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.v3 - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual3_exp_m1() {
    let res = Dual3_64::from(1.2).derive().exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.v1 - 3.32011692273655).abs() < 1e-12);
    assert!((res.v2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.v3 - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual3_exp2() {
    let res = Dual3_64::from(1.2).derive().exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.v1 - 1.59243405216008).abs() < 1e-12);
    assert!((res.v2 - 1.10379117348241).abs() < 1e-12);
    assert!((res.v3 - 0.765089739826287).abs() < 1e-12);
}

#[test]
fn test_dual3_ln() {
    let res = Dual3_64::from(1.2).derive().ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.v1 - 0.833333333333333).abs() < 1e-12);
    assert!((res.v2 - -0.694444444444445).abs() < 1e-12);
    assert!((res.v3 - 1.15740740740741).abs() < 1e-12);
}

#[test]
fn test_dual3_log() {
    let res = Dual3_64::from(1.2).derive().log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.v1 - 0.580685888982970).abs() < 1e-12);
    assert!((res.v2 - -0.483904907485808).abs() < 1e-12);
    assert!((res.v3 - 0.806508179143013).abs() < 1e-12);
}

#[test]
fn test_dual3_ln_1p() {
    let res = Dual3_64::from(1.2).derive().ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.v1 - 0.454545454545455).abs() < 1e-12);
    assert!((res.v2 - -0.206611570247934).abs() < 1e-12);
    assert!((res.v3 - 0.187828700225394).abs() < 1e-12);
}

#[test]
fn test_dual3_log2() {
    let res = Dual3_64::from(1.2).derive().log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.v1 - 1.20224586740747).abs() < 1e-12);
    assert!((res.v2 - -1.00187155617289).abs() < 1e-12);
    assert!((res.v3 - 1.66978592695482).abs() < 1e-12);
}

#[test]
fn test_dual3_log10() {
    let res = Dual3_64::from(1.2).derive().log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.v1 - 0.361912068252710).abs() < 1e-12);
    assert!((res.v2 - -0.301593390210592).abs() < 1e-12);
    assert!((res.v3 - 0.502655650350986).abs() < 1e-12);
}

#[test]
fn test_dual3_sqrt() {
    let res = Dual3_64::from(1.2).derive().sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.v1 - 0.456435464587638).abs() < 1e-12);
    assert!((res.v2 - -0.190181443578183).abs() < 1e-12);
    assert!((res.v3 - 0.237726804472728).abs() < 1e-12);
}

#[test]
fn test_dual3_cbrt() {
    let res = Dual3_64::from(1.2).derive().cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.v1 - 0.295182935884059).abs() < 1e-12);
    assert!((res.v2 - -0.163990519935588).abs() < 1e-12);
    assert!((res.v3 - 0.227764611021650).abs() < 1e-12);
}

#[test]
fn test_dual3_powf() {
    let res = Dual3_64::from(1.2).derive().powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.v1 - 7.52712759108966).abs() < 1e-12);
    assert!((res.v2 - 20.0723402429058).abs() < 1e-12);
    assert!((res.v3 - 36.7992904453272).abs() < 1e-12);
}

#[test]
fn test_dual3_powf_0() {
    let res = Dual3_64::from(0.0).derive().powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powf_1() {
    let res = Dual3_64::from(0.0).derive().powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1 - 1.00000000000000).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powf_2() {
    let res = Dual3_64::from(0.0).derive().powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2 - 2.00000000000000).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powf_3() {
    let res = Dual3_64::from(0.0).derive().powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3 - 6.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual3_powf_4() {
    let res = Dual3_64::from(0.0).derive().powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powi() {
    let res = Dual3_64::from(1.2).derive().powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.v1 - 14.9299200000000).abs() < 1e-12);
    assert!((res.v2 - 62.2080000000000).abs() < 1e-12);
    assert!((res.v3 - 207.360000000000).abs() < 1e-12);
}

#[test]
fn test_dual3_powi_0() {
    let res = Dual3_64::from(0.0).derive().powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powi_1() {
    let res = Dual3_64::from(0.0).derive().powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1 - 1.00000000000000).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powi_2() {
    let res = Dual3_64::from(0.0).derive().powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2 - 2.00000000000000).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_powi_3() {
    let res = Dual3_64::from(0.0).derive().powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3 - 6.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual3_powi_4() {
    let res = Dual3_64::from(0.0).derive().powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.v1).abs() < 1e-12);
    assert!((res.v2).abs() < 1e-12);
    assert!((res.v3).abs() < 1e-12);
}

#[test]
fn test_dual3_sin() {
    let res = Dual3_64::from(1.2).derive().sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.v1 - 0.362357754476674).abs() < 1e-12);
    assert!((res.v2 - -0.932039085967226).abs() < 1e-12);
    assert!((res.v3 - -0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_dual3_cos() {
    let res = Dual3_64::from(1.2).derive().cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.v1 - -0.932039085967226).abs() < 1e-12);
    assert!((res.v2 - -0.362357754476674).abs() < 1e-12);
    assert!((res.v3 - 0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_dual3_tan() {
    let res = Dual3_64::from(1.2).derive().tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.v1 - 7.61596396720705).abs() < 1e-12);
    assert!((res.v2 - 39.1788281446144).abs() < 1e-12);
    assert!((res.v3 - 317.553587029949).abs() < 1e-12);
}

#[test]
fn test_dual3_asin() {
    let res = Dual3_64::from(0.2).derive().asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.v1 - 1.02062072615966).abs() < 1e-12);
    assert!((res.v2 - 0.212629317949929).abs() < 1e-12);
    assert!((res.v3 - 1.19603991346835).abs() < 1e-12);
}

#[test]
fn test_dual3_acos() {
    let res = Dual3_64::from(0.2).derive().acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.v1 - -1.02062072615966).abs() < 1e-12);
    assert!((res.v2 - -0.212629317949929).abs() < 1e-12);
    assert!((res.v3 - -1.19603991346835).abs() < 1e-12);
}

#[test]
fn test_dual3_atan() {
    let res = Dual3_64::from(0.2).derive().atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.v1 - 0.961538461538462).abs() < 1e-12);
    assert!((res.v2 - -0.369822485207101).abs() < 1e-12);
    assert!((res.v3 - -1.56463359126081).abs() < 1e-12);
}

#[test]
fn test_dual3_sinh() {
    let res = Dual3_64::from(1.2).derive().sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.v1 - 1.81065556732437).abs() < 1e-12);
    assert!((res.v2 - 1.50946135541217).abs() < 1e-12);
    assert!((res.v3 - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_dual3_cosh() {
    let res = Dual3_64::from(1.2).derive().cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.v1 - 1.50946135541217).abs() < 1e-12);
    assert!((res.v2 - 1.81065556732437).abs() < 1e-12);
    assert!((res.v3 - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_dual3_tanh() {
    let res = Dual3_64::from(1.2).derive().tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.v1 - 0.305019996207409).abs() < 1e-12);
    assert!((res.v2 - -0.508562650138273).abs() < 1e-12);
    assert!((res.v3 - 0.661856796311429).abs() < 1e-12);
}

#[test]
fn test_dual3_asinh() {
    let res = Dual3_64::from(1.2).derive().asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.v1 - 0.640184399664480).abs() < 1e-12);
    assert!((res.v2 - -0.314844786720236).abs() < 1e-12);
    assert!((res.v3 - 0.202154439560807).abs() < 1e-12);
}

#[test]
fn test_dual3_acosh() {
    let res = Dual3_64::from(1.2).derive().acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.v1 - 1.50755672288882).abs() < 1e-12);
    assert!((res.v2 - -4.11151833515132).abs() < 1e-12);
    assert!((res.v3 - 30.2134301901272).abs() < 1e-12);
}

#[test]
fn test_dual3_atanh() {
    let res = Dual3_64::from(0.2).derive().atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.v1 - 1.04166666666667).abs() < 1e-12);
    assert!((res.v2 - 0.434027777777778).abs() < 1e-12);
    assert!((res.v3 - 2.53182870370370).abs() < 1e-12);
}

#[test]
fn test_dual3_sph_j0() {
    let res = Dual3_64::from(1.2).derive().sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.v1 - -0.345284569857790).abs() < 1e-12);
    assert!((res.v2 - -0.201224955209705).abs() < 1e-12);
    assert!((res.v3 - 0.201097592627034).abs() < 1e-12);
}

#[test]
fn test_dual3_sph_j1() {
    let res = Dual3_64::from(1.2).derive().sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.v1 - 0.201224955209705).abs() < 1e-12);
    assert!((res.v2 - -0.201097592627034).abs() < 1e-12);
    assert!((res.v3 - -0.106373929549242).abs() < 1e-12);
}

#[test]
fn test_dual3_sph_j2() {
    let res = Dual3_64::from(1.2).derive().sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.v1 - 0.129004104011656).abs() < 1e-12);
    assert!((res.v2 - 0.0589484167190109).abs() < 1e-12);
    assert!((res.v3 - -0.111341070273405).abs() < 1e-12);
}

