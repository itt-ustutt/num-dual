use num_hyperdual::hyperdual_n::HyperDualN64;
use num_hyperdual::DualNumMethods;

#[test]
fn test_hyperdual_n_recip() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.gradient[0] - -0.694444444444445).abs() < 1e-12);
    assert!((res.gradient[1] - -0.694444444444445).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 1.15740740740741).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 1.15740740740741).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 1.15740740740741).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 1.15740740740741).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_exp() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.gradient[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.gradient[1] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_exp_m1() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.gradient[0] - 3.32011692273655).abs() < 1e-12);
    assert!((res.gradient[1] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 3.32011692273655).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_exp2() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.gradient[0] - 1.59243405216008).abs() < 1e-12);
    assert!((res.gradient[1] - 1.59243405216008).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 1.10379117348241).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 1.10379117348241).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 1.10379117348241).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 1.10379117348241).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_ln() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.gradient[0] - 0.833333333333333).abs() < 1e-12);
    assert!((res.gradient[1] - 0.833333333333333).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.694444444444445).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.694444444444445).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.694444444444445).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.694444444444445).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_log() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.gradient[0] - 0.580685888982970).abs() < 1e-12);
    assert!((res.gradient[1] - 0.580685888982970).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.483904907485808).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.483904907485808).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.483904907485808).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.483904907485808).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_ln_1p() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.gradient[0] - 0.454545454545455).abs() < 1e-12);
    assert!((res.gradient[1] - 0.454545454545455).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.206611570247934).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.206611570247934).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.206611570247934).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.206611570247934).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_log2() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.gradient[0] - 1.20224586740747).abs() < 1e-12);
    assert!((res.gradient[1] - 1.20224586740747).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -1.00187155617289).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -1.00187155617289).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -1.00187155617289).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -1.00187155617289).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_log10() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.gradient[0] - 0.361912068252710).abs() < 1e-12);
    assert!((res.gradient[1] - 0.361912068252710).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.301593390210592).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.301593390210592).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.301593390210592).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.301593390210592).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sqrt() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.gradient[0] - 0.456435464587638).abs() < 1e-12);
    assert!((res.gradient[1] - 0.456435464587638).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.190181443578183).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.190181443578183).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.190181443578183).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.190181443578183).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_cbrt() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.gradient[0] - 0.295182935884059).abs() < 1e-12);
    assert!((res.gradient[1] - 0.295182935884059).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.163990519935588).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.163990519935588).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.163990519935588).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.163990519935588).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.gradient[0] - 7.52712759108966).abs() < 1e-12);
    assert!((res.gradient[1] - 7.52712759108966).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 20.0723402429058).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 20.0723402429058).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 20.0723402429058).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 20.0723402429058).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf_0() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf_1() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.gradient[1] - 1.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf_2() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 2.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf_3() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powf_4() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.gradient[0] - 14.9299200000000).abs() < 1e-12);
    assert!((res.gradient[1] - 14.9299200000000).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 62.2080000000000).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 62.2080000000000).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 62.2080000000000).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 62.2080000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi_0() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi_1() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0] - 1.00000000000000).abs() < 1e-12);
    assert!((res.gradient[1] - 1.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi_2() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 2.00000000000000).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 2.00000000000000).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi_3() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_powi_4() {
    let res = HyperDualN64::<2>::from(0.0).derive(0).derive(1).powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.gradient[0]).abs() < 1e-12);
    assert!((res.gradient[1]).abs() < 1e-12);
    assert!((res.hessian[(0,0)]).abs() < 1e-12);
    assert!((res.hessian[(0,1)]).abs() < 1e-12);
    assert!((res.hessian[(1,0)]).abs() < 1e-12);
    assert!((res.hessian[(1,1)]).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sin() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.gradient[0] - 0.362357754476674).abs() < 1e-12);
    assert!((res.gradient[1] - 0.362357754476674).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.932039085967226).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.932039085967226).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.932039085967226).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_cos() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.gradient[0] - -0.932039085967226).abs() < 1e-12);
    assert!((res.gradient[1] - -0.932039085967226).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.362357754476674).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.362357754476674).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.362357754476674).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_tan() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.gradient[0] - 7.61596396720705).abs() < 1e-12);
    assert!((res.gradient[1] - 7.61596396720705).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 39.1788281446144).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 39.1788281446144).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 39.1788281446144).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 39.1788281446144).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_asin() {
    let res = HyperDualN64::<2>::from(0.2).derive(0).derive(1).asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.gradient[0] - 1.02062072615966).abs() < 1e-12);
    assert!((res.gradient[1] - 1.02062072615966).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 0.212629317949929).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_acos() {
    let res = HyperDualN64::<2>::from(0.2).derive(0).derive(1).acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.gradient[0] - -1.02062072615966).abs() < 1e-12);
    assert!((res.gradient[1] - -1.02062072615966).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.212629317949929).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.212629317949929).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_atan() {
    let res = HyperDualN64::<2>::from(0.2).derive(0).derive(1).atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.gradient[0] - 0.961538461538462).abs() < 1e-12);
    assert!((res.gradient[1] - 0.961538461538462).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.369822485207101).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.369822485207101).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.369822485207101).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.369822485207101).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sinh() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.gradient[0] - 1.81065556732437).abs() < 1e-12);
    assert!((res.gradient[1] - 1.81065556732437).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 1.50946135541217).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 1.50946135541217).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 1.50946135541217).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_cosh() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.gradient[0] - 1.50946135541217).abs() < 1e-12);
    assert!((res.gradient[1] - 1.50946135541217).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 1.81065556732437).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 1.81065556732437).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 1.81065556732437).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_tanh() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.gradient[0] - 0.305019996207409).abs() < 1e-12);
    assert!((res.gradient[1] - 0.305019996207409).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.508562650138273).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.508562650138273).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.508562650138273).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.508562650138273).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_asinh() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.gradient[0] - 0.640184399664480).abs() < 1e-12);
    assert!((res.gradient[1] - 0.640184399664480).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.314844786720236).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.314844786720236).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.314844786720236).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.314844786720236).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_acosh() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.gradient[0] - 1.50755672288882).abs() < 1e-12);
    assert!((res.gradient[1] - 1.50755672288882).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -4.11151833515132).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -4.11151833515132).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -4.11151833515132).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -4.11151833515132).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_atanh() {
    let res = HyperDualN64::<2>::from(0.2).derive(0).derive(1).atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.gradient[0] - 1.04166666666667).abs() < 1e-12);
    assert!((res.gradient[1] - 1.04166666666667).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 0.434027777777778).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 0.434027777777778).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 0.434027777777778).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 0.434027777777778).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sph_j0() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.gradient[0] - -0.345284569857790).abs() < 1e-12);
    assert!((res.gradient[1] - -0.345284569857790).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.201224955209705).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.201224955209705).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.201224955209705).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.201224955209705).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sph_j1() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.gradient[0] - 0.201224955209705).abs() < 1e-12);
    assert!((res.gradient[1] - 0.201224955209705).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - -0.201097592627034).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - -0.201097592627034).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - -0.201097592627034).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - -0.201097592627034).abs() < 1e-12);
}

#[test]
fn test_hyperdual_n_sph_j2() {
    let res = HyperDualN64::<2>::from(1.2).derive(0).derive(1).sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.gradient[0] - 0.129004104011656).abs() < 1e-12);
    assert!((res.gradient[1] - 0.129004104011656).abs() < 1e-12);
    assert!((res.hessian[(0,0)] - 0.0589484167190109).abs() < 1e-12);
    assert!((res.hessian[(0,1)] - 0.0589484167190109).abs() < 1e-12);
    assert!((res.hessian[(1,0)] - 0.0589484167190109).abs() < 1e-12);
    assert!((res.hessian[(1,1)] - 0.0589484167190109).abs() < 1e-12);
}

