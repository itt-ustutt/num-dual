use num_dual::*;

#[test]
fn test_hyperdual_recip() {
    let res = HyperDual64::from(1.2).derive1().derive2().recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps1[0] - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps2[0] - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 1.15740740740741).abs() < 1e-12);
}

#[test]
fn test_hyperdual_exp() {
    let res = HyperDual64::from(1.2).derive1().derive2().exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperdual_exp_m1() {
    let res = HyperDual64::from(1.2).derive1().derive2().exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.eps1[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperdual_exp2() {
    let res = HyperDual64::from(1.2).derive1().derive2().exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.eps1[0] - 1.59243405216008).abs() < 1e-12);
    assert!((res.eps2[0] - 1.59243405216008).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 1.10379117348241).abs() < 1e-12);
}

#[test]
fn test_hyperdual_ln() {
    let res = HyperDual64::from(1.2).derive1().derive2().ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.eps1[0] - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps2[0] - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.694444444444445).abs() < 1e-12);
}

#[test]
fn test_hyperdual_log() {
    let res = HyperDual64::from(1.2).derive1().derive2().log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.eps1[0] - 0.580685888982970).abs() < 1e-12);
    assert!((res.eps2[0] - 0.580685888982970).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.483904907485808).abs() < 1e-12);
}

#[test]
fn test_hyperdual_ln_1p() {
    let res = HyperDual64::from(1.2).derive1().derive2().ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.eps1[0] - 0.454545454545455).abs() < 1e-12);
    assert!((res.eps2[0] - 0.454545454545455).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.206611570247934).abs() < 1e-12);
}

#[test]
fn test_hyperdual_log2() {
    let res = HyperDual64::from(1.2).derive1().derive2().log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.eps1[0] - 1.20224586740747).abs() < 1e-12);
    assert!((res.eps2[0] - 1.20224586740747).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -1.00187155617289).abs() < 1e-12);
}

#[test]
fn test_hyperdual_log10() {
    let res = HyperDual64::from(1.2).derive1().derive2().log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.eps1[0] - 0.361912068252710).abs() < 1e-12);
    assert!((res.eps2[0] - 0.361912068252710).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.301593390210592).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sqrt() {
    let res = HyperDual64::from(1.2).derive1().derive2().sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.eps1[0] - 0.456435464587638).abs() < 1e-12);
    assert!((res.eps2[0] - 0.456435464587638).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.190181443578183).abs() < 1e-12);
}

#[test]
fn test_hyperdual_cbrt() {
    let res = HyperDual64::from(1.2).derive1().derive2().cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.eps1[0] - 0.295182935884059).abs() < 1e-12);
    assert!((res.eps2[0] - 0.295182935884059).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.163990519935588).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf() {
    let res = HyperDual64::from(1.2).derive1().derive2().powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.eps1[0] - 7.52712759108966).abs() < 1e-12);
    assert!((res.eps2[0] - 7.52712759108966).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 20.0723402429058).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf_0() {
    let res = HyperDual64::from(0.0).derive1().derive2().powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf_1() {
    let res = HyperDual64::from(0.0).derive1().derive2().powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps2[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf_2() {
    let res = HyperDual64::from(0.0).derive1().derive2().powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 2.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf_3() {
    let res = HyperDual64::from(0.0).derive1().derive2().powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powf_4() {
    let res = HyperDual64::from(0.0).derive1().derive2().powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi() {
    let res = HyperDual64::from(1.2).derive1().derive2().powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.eps1[0] - 14.9299200000000).abs() < 1e-12);
    assert!((res.eps2[0] - 14.9299200000000).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 62.2080000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi_0() {
    let res = HyperDual64::from(0.0).derive1().derive2().powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi_1() {
    let res = HyperDual64::from(0.0).derive1().derive2().powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps2[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi_2() {
    let res = HyperDual64::from(0.0).derive1().derive2().powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 2.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi_3() {
    let res = HyperDual64::from(0.0).derive1().derive2().powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_powi_4() {
    let res = HyperDual64::from(0.0).derive1().derive2().powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sin() {
    let res = HyperDual64::from(1.2).derive1().derive2().sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.eps1[0] - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps2[0] - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_hyperdual_cos() {
    let res = HyperDual64::from(1.2).derive1().derive2().cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps1[0] - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps2[0] - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_hyperdual_tan() {
    let res = HyperDual64::from(1.2).derive1().derive2().tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.eps1[0] - 7.61596396720705).abs() < 1e-12);
    assert!((res.eps2[0] - 7.61596396720705).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 39.1788281446144).abs() < 1e-12);
}

#[test]
fn test_hyperdual_asin() {
    let res = HyperDual64::from(0.2).derive1().derive2().asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.eps1[0] - 1.02062072615966).abs() < 1e-12);
    assert!((res.eps2[0] - 1.02062072615966).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.212629317949929).abs() < 1e-12);
}

#[test]
fn test_hyperdual_acos() {
    let res = HyperDual64::from(0.2).derive1().derive2().acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.eps1[0] - -1.02062072615966).abs() < 1e-12);
    assert!((res.eps2[0] - -1.02062072615966).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.212629317949929).abs() < 1e-12);
}

#[test]
fn test_hyperdual_atan() {
    let res = HyperDual64::from(0.2).derive1().derive2().atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.eps1[0] - 0.961538461538462).abs() < 1e-12);
    assert!((res.eps2[0] - 0.961538461538462).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.369822485207101).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sinh() {
    let res = HyperDual64::from(1.2).derive1().derive2().sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1[0] - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps2[0] - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_hyperdual_cosh() {
    let res = HyperDual64::from(1.2).derive1().derive2().cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1[0] - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps2[0] - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_hyperdual_tanh() {
    let res = HyperDual64::from(1.2).derive1().derive2().tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.eps1[0] - 0.305019996207409).abs() < 1e-12);
    assert!((res.eps2[0] - 0.305019996207409).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.508562650138273).abs() < 1e-12);
}

#[test]
fn test_hyperdual_asinh() {
    let res = HyperDual64::from(1.2).derive1().derive2().asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.eps1[0] - 0.640184399664480).abs() < 1e-12);
    assert!((res.eps2[0] - 0.640184399664480).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.314844786720236).abs() < 1e-12);
}

#[test]
fn test_hyperdual_acosh() {
    let res = HyperDual64::from(1.2).derive1().derive2().acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.eps1[0] - 1.50755672288882).abs() < 1e-12);
    assert!((res.eps2[0] - 1.50755672288882).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -4.11151833515132).abs() < 1e-12);
}

#[test]
fn test_hyperdual_atanh() {
    let res = HyperDual64::from(0.2).derive1().derive2().atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.eps1[0] - 1.04166666666667).abs() < 1e-12);
    assert!((res.eps2[0] - 1.04166666666667).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.434027777777778).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sph_j0() {
    let res = HyperDual64::from(1.2).derive1().derive2().sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.eps1[0] - -0.345284569857790).abs() < 1e-12);
    assert!((res.eps2[0] - -0.345284569857790).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.201224955209705).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sph_j1() {
    let res = HyperDual64::from(1.2).derive1().derive2().sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.eps1[0] - 0.201224955209705).abs() < 1e-12);
    assert!((res.eps2[0] - 0.201224955209705).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.201097592627034).abs() < 1e-12);
}

#[test]
fn test_hyperdual_sph_j2() {
    let res = HyperDual64::from(1.2).derive1().derive2().sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.eps1[0] - 0.129004104011656).abs() < 1e-12);
    assert!((res.eps2[0] - 0.129004104011656).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.0589484167190109).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j0_0() {
    let res = HyperDual64::from(0.0).derive1().derive2().bessel_j0();
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.500000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j1_0() {
    let res = HyperDual64::from(0.0).derive1().derive2().bessel_j1();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0] - 0.500000000000000).abs() < 1e-12);
    assert!((res.eps2[0] - 0.500000000000000).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j2_0() {
    let res = HyperDual64::from(0.0).derive1().derive2().bessel_j2();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1[0]).abs() < 1e-12);
    assert!((res.eps2[0]).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.250000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j0_1() {
    let res = HyperDual64::from(1.2).derive1().derive2().bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps1[0] - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps2[0] - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j1_1() {
    let res = HyperDual64::from(1.2).derive1().derive2().bessel_j1();
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps1[0] - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps2[0] - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.365498208944163).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j2_1() {
    let res = HyperDual64::from(1.2).derive1().derive2().bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps1[0] - 0.232707360321110).abs() < 1e-12);
    assert!((res.eps2[0] - 0.232707360321110).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.0893643434615870).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j0_2() {
    let res = HyperDual64::from(7.2).derive1().derive2().bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps1[0] - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps2[0] - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j1_2() {
    let res = HyperDual64::from(7.2).derive1().derive2().bessel_j1();
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1[0] - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps2[0] - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.0932134954083656).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j2_2() {
    let res = HyperDual64::from(7.2).derive1().derive2().bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps1[0] - 0.132099570594364).abs() < 1e-12);
    assert!((res.eps2[0] - 0.132099570594364).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.240029203653306).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j0_3() {
    let res = HyperDual64::from(-1.2).derive1().derive2().bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps1[0] - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps2[0] - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j1_3() {
    let res = HyperDual64::from(-1.2).derive1().derive2().bessel_j1();
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps1[0] - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps2[0] - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.365498208944163).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j2_3() {
    let res = HyperDual64::from(-1.2).derive1().derive2().bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps1[0] - -0.232707360321110).abs() < 1e-12);
    assert!((res.eps2[0] - -0.232707360321110).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.0893643434615870).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j0_4() {
    let res = HyperDual64::from(-7.2).derive1().derive2().bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps1[0] - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps2[0] - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - -0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j1_4() {
    let res = HyperDual64::from(-7.2).derive1().derive2().bessel_j1();
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1[0] - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps2[0] - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.0932134954083656).abs() < 1e-12);
}

#[test]
fn test_hyperdual_bessel_j2_4() {
    let res = HyperDual64::from(-7.2).derive1().derive2().bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps1[0] - -0.132099570594364).abs() < 1e-12);
    assert!((res.eps2[0] - -0.132099570594364).abs() < 1e-12);
    assert!((res.eps1eps2[(0,0)] - 0.240029203653306).abs() < 1e-12);
}

