use num_dual::*;

#[test]
fn test_hyperhyperdual_recip() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps1 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps2 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps3 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps1eps2 - 1.15740740740741).abs() < 1e-12);
    assert!((res.eps1eps3 - 1.15740740740741).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.15740740740741).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.15740740740741).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -2.89351851851852).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_exp() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_exp_m1() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.eps1 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps2eps3 - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_exp2() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.eps1 - 1.59243405216008).abs() < 1e-12);
    assert!((res.eps2 - 1.59243405216008).abs() < 1e-12);
    assert!((res.eps3 - 1.59243405216008).abs() < 1e-12);
    assert!((res.eps1eps2 - 1.10379117348241).abs() < 1e-12);
    assert!((res.eps1eps3 - 1.10379117348241).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.10379117348241).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.10379117348241).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.765089739826287).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_ln() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.eps1 - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps2 - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps3 - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.694444444444445).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 1.15740740740741).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_log() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.eps1 - 0.580685888982970).abs() < 1e-12);
    assert!((res.eps2 - 0.580685888982970).abs() < 1e-12);
    assert!((res.eps3 - 0.580685888982970).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.483904907485808).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.483904907485808).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.483904907485808).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.483904907485808).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.806508179143013).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_ln_1p() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.eps1 - 0.454545454545455).abs() < 1e-12);
    assert!((res.eps2 - 0.454545454545455).abs() < 1e-12);
    assert!((res.eps3 - 0.454545454545455).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.206611570247934).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.206611570247934).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.206611570247934).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.206611570247934).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.187828700225394).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_log2() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.eps1 - 1.20224586740747).abs() < 1e-12);
    assert!((res.eps2 - 1.20224586740747).abs() < 1e-12);
    assert!((res.eps3 - 1.20224586740747).abs() < 1e-12);
    assert!((res.eps1eps2 - -1.00187155617289).abs() < 1e-12);
    assert!((res.eps1eps3 - -1.00187155617289).abs() < 1e-12);
    assert!((res.eps2eps3 - -1.00187155617289).abs() < 1e-12);
    assert!((res.eps2eps3 - -1.00187155617289).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 1.66978592695482).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_log10() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.eps1 - 0.361912068252710).abs() < 1e-12);
    assert!((res.eps2 - 0.361912068252710).abs() < 1e-12);
    assert!((res.eps3 - 0.361912068252710).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.301593390210592).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.301593390210592).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.301593390210592).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.301593390210592).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.502655650350986).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sqrt() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.eps1 - 0.456435464587638).abs() < 1e-12);
    assert!((res.eps2 - 0.456435464587638).abs() < 1e-12);
    assert!((res.eps3 - 0.456435464587638).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.190181443578183).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.190181443578183).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.190181443578183).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.190181443578183).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.237726804472728).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_cbrt() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.eps1 - 0.295182935884059).abs() < 1e-12);
    assert!((res.eps2 - 0.295182935884059).abs() < 1e-12);
    assert!((res.eps3 - 0.295182935884059).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.163990519935588).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.163990519935588).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.163990519935588).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.163990519935588).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.227764611021650).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.eps1 - 7.52712759108966).abs() < 1e-12);
    assert!((res.eps2 - 7.52712759108966).abs() < 1e-12);
    assert!((res.eps3 - 7.52712759108966).abs() < 1e-12);
    assert!((res.eps1eps2 - 20.0723402429058).abs() < 1e-12);
    assert!((res.eps1eps3 - 20.0723402429058).abs() < 1e-12);
    assert!((res.eps2eps3 - 20.0723402429058).abs() < 1e-12);
    assert!((res.eps2eps3 - 20.0723402429058).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 36.7992904453272).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf_0() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf_1() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps2 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps3 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf_2() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf_3() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 6.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powf_4() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.eps1 - 14.9299200000000).abs() < 1e-12);
    assert!((res.eps2 - 14.9299200000000).abs() < 1e-12);
    assert!((res.eps3 - 14.9299200000000).abs() < 1e-12);
    assert!((res.eps1eps2 - 62.2080000000000).abs() < 1e-12);
    assert!((res.eps1eps3 - 62.2080000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 62.2080000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 62.2080000000000).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 207.360000000000).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi_0() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi_1() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps2 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps3 - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi_2() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 2.00000000000000).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi_3() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 6.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_powi_4() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sin() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.eps1 - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps2 - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps3 - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_cos() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps1 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps2 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps3 - -0.932039085967226).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.362357754476674).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.362357754476674).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.362357754476674).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.362357754476674).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_tan() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.eps1 - 7.61596396720705).abs() < 1e-12);
    assert!((res.eps2 - 7.61596396720705).abs() < 1e-12);
    assert!((res.eps3 - 7.61596396720705).abs() < 1e-12);
    assert!((res.eps1eps2 - 39.1788281446144).abs() < 1e-12);
    assert!((res.eps1eps3 - 39.1788281446144).abs() < 1e-12);
    assert!((res.eps2eps3 - 39.1788281446144).abs() < 1e-12);
    assert!((res.eps2eps3 - 39.1788281446144).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 317.553587029949).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_asin() {
    let res = HyperHyperDual64::new(0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.eps1 - 1.02062072615966).abs() < 1e-12);
    assert!((res.eps2 - 1.02062072615966).abs() < 1e-12);
    assert!((res.eps3 - 1.02062072615966).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.212629317949929).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.212629317949929).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.212629317949929).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.212629317949929).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 1.19603991346835).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_acos() {
    let res = HyperHyperDual64::new(0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.eps1 - -1.02062072615966).abs() < 1e-12);
    assert!((res.eps2 - -1.02062072615966).abs() < 1e-12);
    assert!((res.eps3 - -1.02062072615966).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.212629317949929).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.212629317949929).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.212629317949929).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.212629317949929).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -1.19603991346835).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_atan() {
    let res = HyperHyperDual64::new(0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.eps1 - 0.961538461538462).abs() < 1e-12);
    assert!((res.eps2 - 0.961538461538462).abs() < 1e-12);
    assert!((res.eps3 - 0.961538461538462).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.369822485207101).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.369822485207101).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.369822485207101).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.369822485207101).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -1.56463359126081).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sinh() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps2 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps3 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1eps2 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1eps3 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_cosh() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps2 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps3 - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps1eps2 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1eps3 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps2eps3 - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_tanh() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.eps1 - 0.305019996207409).abs() < 1e-12);
    assert!((res.eps2 - 0.305019996207409).abs() < 1e-12);
    assert!((res.eps3 - 0.305019996207409).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.508562650138273).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.508562650138273).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.508562650138273).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.508562650138273).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.661856796311429).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_asinh() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.eps1 - 0.640184399664480).abs() < 1e-12);
    assert!((res.eps2 - 0.640184399664480).abs() < 1e-12);
    assert!((res.eps3 - 0.640184399664480).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.314844786720236).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.314844786720236).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.314844786720236).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.314844786720236).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.202154439560807).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_acosh() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.eps1 - 1.50755672288882).abs() < 1e-12);
    assert!((res.eps2 - 1.50755672288882).abs() < 1e-12);
    assert!((res.eps3 - 1.50755672288882).abs() < 1e-12);
    assert!((res.eps1eps2 - -4.11151833515132).abs() < 1e-12);
    assert!((res.eps1eps3 - -4.11151833515132).abs() < 1e-12);
    assert!((res.eps2eps3 - -4.11151833515132).abs() < 1e-12);
    assert!((res.eps2eps3 - -4.11151833515132).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 30.2134301901272).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_atanh() {
    let res = HyperHyperDual64::new(0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.eps1 - 1.04166666666667).abs() < 1e-12);
    assert!((res.eps2 - 1.04166666666667).abs() < 1e-12);
    assert!((res.eps3 - 1.04166666666667).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.434027777777778).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.434027777777778).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.434027777777778).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.434027777777778).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 2.53182870370370).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sph_j0() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.eps1 - -0.345284569857790).abs() < 1e-12);
    assert!((res.eps2 - -0.345284569857790).abs() < 1e-12);
    assert!((res.eps3 - -0.345284569857790).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.201224955209705).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.201224955209705).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.201224955209705).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.201224955209705).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.201097592627034).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sph_j1() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.eps1 - 0.201224955209705).abs() < 1e-12);
    assert!((res.eps2 - 0.201224955209705).abs() < 1e-12);
    assert!((res.eps3 - 0.201224955209705).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.201097592627034).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.201097592627034).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.201097592627034).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.201097592627034).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.106373929549242).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_sph_j2() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.eps1 - 0.129004104011656).abs() < 1e-12);
    assert!((res.eps2 - 0.129004104011656).abs() < 1e-12);
    assert!((res.eps3 - 0.129004104011656).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.0589484167190109).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.0589484167190109).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0589484167190109).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0589484167190109).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.111341070273405).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j0_0() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j0();
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.500000000000000).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.500000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.500000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.500000000000000).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j1_0() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j1();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1 - 0.500000000000000).abs() < 1e-12);
    assert!((res.eps2 - 0.500000000000000).abs() < 1e-12);
    assert!((res.eps3 - 0.500000000000000).abs() < 1e-12);
    assert!((res.eps1eps2).abs() < 1e-12);
    assert!((res.eps1eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps2eps3).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.375000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j2_0() {
    let res = HyperHyperDual64::new(0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j2();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps1).abs() < 1e-12);
    assert!((res.eps2).abs() < 1e-12);
    assert!((res.eps3).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.250000000000000).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.250000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.250000000000000).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.250000000000000).abs() < 1e-12);
    assert!((res.eps1eps2eps3).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j0_1() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps1 - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps2 - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps3 - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.365498208944163).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j1_1() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j1();
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps1 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps2 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps3 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.365498208944163).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.365498208944163).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.365498208944163).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.365498208944163).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.172628103209968).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j2_1() {
    let res = HyperHyperDual64::new(1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps1 - 0.232707360321110).abs() < 1e-12);
    assert!((res.eps2 - 0.232707360321110).abs() < 1e-12);
    assert!((res.eps3 - 0.232707360321110).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.236892915552203).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j0_2() {
    let res = HyperHyperDual64::new(7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps1 - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps2 - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps3 - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.0932134954083656).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j1_2() {
    let res = HyperHyperDual64::new(7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j1();
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps2 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps3 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.0932134954083656).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.0932134954083656).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.0932134954083656).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.0932134954083656).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.263777210011690).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j2_2() {
    let res = HyperHyperDual64::new(7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps1 - 0.132099570594364).abs() < 1e-12);
    assert!((res.eps2 - 0.132099570594364).abs() < 1e-12);
    assert!((res.eps3 - 0.132099570594364).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.146694937335182).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j0_3() {
    let res = HyperHyperDual64::new(-1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps1 - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps2 - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps3 - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.365498208944163).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j1_3() {
    let res = HyperHyperDual64::new(-1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j1();
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps1 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps2 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps3 - 0.255891862958350).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.365498208944163).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.365498208944163).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.365498208944163).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.365498208944163).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.172628103209968).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j2_3() {
    let res = HyperHyperDual64::new(-1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps1 - -0.232707360321110).abs() < 1e-12);
    assert!((res.eps2 - -0.232707360321110).abs() < 1e-12);
    assert!((res.eps3 - -0.232707360321110).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0893643434615870).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.236892915552203).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j0_4() {
    let res = HyperHyperDual64::new(-7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps1 - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps2 - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps3 - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1eps2 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps2eps3 - -0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.0932134954083656).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j1_4() {
    let res = HyperHyperDual64::new(-7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j1();
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps1 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps2 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps3 - 0.287525216370074).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.0932134954083656).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.0932134954083656).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0932134954083656).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.0932134954083656).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - -0.263777210011690).abs() < 1e-12);
}

#[test]
fn test_hyperhyperdual_bessel_j2_4() {
    let res = HyperHyperDual64::new(-7.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0).bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps1 - -0.132099570594364).abs() < 1e-12);
    assert!((res.eps2 - -0.132099570594364).abs() < 1e-12);
    assert!((res.eps3 - -0.132099570594364).abs() < 1e-12);
    assert!((res.eps1eps2 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps1eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps2eps3 - 0.240029203653306).abs() < 1e-12);
    assert!((res.eps1eps2eps3 - 0.146694937335182).abs() < 1e-12);
}
