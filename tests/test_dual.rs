use num_dual::*;

#[test]
fn test_dual_recip() {
    let res = Dual64::from(1.2).derivative().recip();
    assert!((res.re - 0.833333333333333).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.694444444444445).abs() < 1e-12);
}

#[test]
fn test_dual_exp() {
    let res = Dual64::from(1.2).derivative().exp();
    assert!((res.re - 3.32011692273655).abs() < 1e-12);
    assert!((res.eps.unwrap() - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_exp_m1() {
    let res = Dual64::from(1.2).derivative().exp_m1();
    assert!((res.re - 2.32011692273655).abs() < 1e-12);
    assert!((res.eps.unwrap() - 3.32011692273655).abs() < 1e-12);
}

#[test]
fn test_dual_exp2() {
    let res = Dual64::from(1.2).derivative().exp2();
    assert!((res.re - 2.29739670999407).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.59243405216008).abs() < 1e-12);
}

#[test]
fn test_dual_ln() {
    let res = Dual64::from(1.2).derivative().ln();
    assert!((res.re - 0.182321556793955).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.833333333333333).abs() < 1e-12);
}

#[test]
fn test_dual_log() {
    let res = Dual64::from(1.2).derivative().log(4.2);
    assert!((res.re - 0.127045866345188).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.580685888982970).abs() < 1e-12);
}

#[test]
fn test_dual_ln_1p() {
    let res = Dual64::from(1.2).derivative().ln_1p();
    assert!((res.re - 0.788457360364270).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.454545454545455).abs() < 1e-12);
}

#[test]
fn test_dual_log2() {
    let res = Dual64::from(1.2).derivative().log2();
    assert!((res.re - 0.263034405833794).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.20224586740747).abs() < 1e-12);
}

#[test]
fn test_dual_log10() {
    let res = Dual64::from(1.2).derivative().log10();
    assert!((res.re - 0.0791812460476248).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.361912068252710).abs() < 1e-12);
}

#[test]
fn test_dual_sqrt() {
    let res = Dual64::from(1.2).derivative().sqrt();
    assert!((res.re - 1.09544511501033).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.456435464587638).abs() < 1e-12);
}

#[test]
fn test_dual_cbrt() {
    let res = Dual64::from(1.2).derivative().cbrt();
    assert!((res.re - 1.06265856918261).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.295182935884059).abs() < 1e-12);
}

#[test]
fn test_dual_powf() {
    let res = Dual64::from(1.2).derivative().powf(4.2);
    assert!((res.re - 2.15060788316847).abs() < 1e-12);
    assert!((res.eps.unwrap() - 7.52712759108966).abs() < 1e-12);
}

#[test]
fn test_dual_powf_0() {
    let res = Dual64::from(0.0).derivative().powf(0.0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powf_1() {
    let res = Dual64::from(0.0).derivative().powf(1.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_powf_2() {
    let res = Dual64::from(0.0).derivative().powf(2.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powf_3() {
    let res = Dual64::from(0.0).derivative().powf(3.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powf_4() {
    let res = Dual64::from(0.0).derivative().powf(4.0);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powi() {
    let res = Dual64::from(1.2).derivative().powi(6);
    assert!((res.re - 2.98598400000000).abs() < 1e-12);
    assert!((res.eps.unwrap() - 14.9299200000000).abs() < 1e-12);
}

#[test]
fn test_dual_powi_0() {
    let res = Dual64::from(0.0).derivative().powi(0);
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powi_1() {
    let res = Dual64::from(0.0).derivative().powi(1);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.00000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_powi_2() {
    let res = Dual64::from(0.0).derivative().powi(2);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powi_3() {
    let res = Dual64::from(0.0).derivative().powi(3);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_powi_4() {
    let res = Dual64::from(0.0).derivative().powi(4);
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_sin() {
    let res = Dual64::from(1.2).derivative().sin();
    assert!((res.re - 0.932039085967226).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.362357754476674).abs() < 1e-12);
}

#[test]
fn test_dual_cos() {
    let res = Dual64::from(1.2).derivative().cos();
    assert!((res.re - 0.362357754476674).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.932039085967226).abs() < 1e-12);
}

#[test]
fn test_dual_tan() {
    let res = Dual64::from(1.2).derivative().tan();
    assert!((res.re - 2.57215162212632).abs() < 1e-12);
    assert!((res.eps.unwrap() - 7.61596396720705).abs() < 1e-12);
}

#[test]
fn test_dual_asin() {
    let res = Dual64::from(0.2).derivative().asin();
    assert!((res.re - 0.201357920790331).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_acos() {
    let res = Dual64::from(0.2).derivative().acos();
    assert!((res.re - 1.36943840600457).abs() < 1e-12);
    assert!((res.eps.unwrap() - -1.02062072615966).abs() < 1e-12);
}

#[test]
fn test_dual_atan() {
    let res = Dual64::from(0.2).derivative().atan();
    assert!((res.re - 0.197395559849881).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.961538461538462).abs() < 1e-12);
}

#[test]
fn test_dual_sinh() {
    let res = Dual64::from(1.2).derivative().sinh();
    assert!((res.re - 1.50946135541217).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.81065556732437).abs() < 1e-12);
}

#[test]
fn test_dual_cosh() {
    let res = Dual64::from(1.2).derivative().cosh();
    assert!((res.re - 1.81065556732437).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.50946135541217).abs() < 1e-12);
}

#[test]
fn test_dual_tanh() {
    let res = Dual64::from(1.2).derivative().tanh();
    assert!((res.re - 0.833654607012155).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.305019996207409).abs() < 1e-12);
}

#[test]
fn test_dual_asinh() {
    let res = Dual64::from(1.2).derivative().asinh();
    assert!((res.re - 1.01597313417969).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.640184399664480).abs() < 1e-12);
}

#[test]
fn test_dual_acosh() {
    let res = Dual64::from(1.2).derivative().acosh();
    assert!((res.re - 0.622362503714779).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.50755672288882).abs() < 1e-12);
}

#[test]
fn test_dual_atanh() {
    let res = Dual64::from(0.2).derivative().atanh();
    assert!((res.re - 0.202732554054082).abs() < 1e-12);
    assert!((res.eps.unwrap() - 1.04166666666667).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j0() {
    let res = Dual64::from(1.2).derivative().sph_j0();
    assert!((res.re - 0.776699238306022).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.345284569857790).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j1() {
    let res = Dual64::from(1.2).derivative().sph_j1();
    assert!((res.re - 0.345284569857790).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.201224955209705).abs() < 1e-12);
}

#[test]
fn test_dual_sph_j2() {
    let res = Dual64::from(1.2).derivative().sph_j2();
    assert!((res.re - 0.0865121863384538).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.129004104011656).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_0() {
    let res = Dual64::from(0.0).derivative().bessel_j0();
    assert!((res.re - 1.00000000000000).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_0() {
    let res = Dual64::from(0.0).derivative().bessel_j1();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.500000000000000).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_0() {
    let res = Dual64::from(0.0).derivative().bessel_j2();
    assert!((res.re).abs() < 1e-12);
    assert!((res.eps.unwrap()).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_1() {
    let res = Dual64::from(1.2).derivative().bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_1() {
    let res = Dual64::from(1.2).derivative().bessel_j1();
    assert!((res.re - 0.498289057567215).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_1() {
    let res = Dual64::from(1.2).derivative().bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_2() {
    let res = Dual64::from(7.2).derivative().bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_2() {
    let res = Dual64::from(7.2).derivative().bessel_j1();
    assert!((res.re - 0.0543274202223671).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_2() {
    let res = Dual64::from(7.2).derivative().bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.132099570594364).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_3() {
    let res = Dual64::from(-1.2).derivative().bessel_j0();
    assert!((res.re - 0.671132744264363).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.498289057567215).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_3() {
    let res = Dual64::from(-1.2).derivative().bessel_j1();
    assert!((res.re - -0.498289057567215).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.255891862958350).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_3() {
    let res = Dual64::from(-1.2).derivative().bessel_j2();
    assert!((res.re - 0.159349018347663).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.232707360321110).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j0_4() {
    let res = Dual64::from(-7.2).derivative().bessel_j0();
    assert!((res.re - 0.295070691400958).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.0543274202223671).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j1_4() {
    let res = Dual64::from(-7.2).derivative().bessel_j1();
    assert!((res.re - -0.0543274202223671).abs() < 1e-12);
    assert!((res.eps.unwrap() - 0.287525216370074).abs() < 1e-12);
}

#[test]
fn test_dual_bessel_j2_4() {
    let res = Dual64::from(-7.2).derivative().bessel_j2();
    assert!((res.re - -0.279979741339189).abs() < 1e-12);
    assert!((res.eps.unwrap() - -0.132099570594364).abs() < 1e-12);
}

mod nalgebra_api {
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector2, Vector3};
    use num_dual::*;
    use num_traits::Zero;
    use std::f32::consts::*;

    fn unit_circle(t: Dual32) -> Point2<Dual32> {
        let x_dir = |x: Dual32| Vector2::new(x, Dual32::zero());
        let y_dir = |y: Dual32| Vector2::new(Dual32::zero(), y);
        let theta = t * TAU;
        Point2::from(x_dir(theta.cos()) + y_dir(theta.sin()))
    }

    // This is testing that you can type-check code that whacks DualVec in
    // nalgebra structures and tries to use them.
    #[test]
    fn use_nalgebra_2d() {
        // 1 radian around the circle
        let t = Dual32::from_re(0.25).derivative();
        let point = unit_circle(t);
        let real = point.map(|x| x.re);
        let grad = point.map(|x| x.eps.unwrap());
        println!("point: {}", point.coords);
        approx::assert_relative_eq!(real, Point2::new(0., 1.), epsilon = 1e-3);
        approx::assert_relative_eq!(grad, Point2::new(-TAU, 0.), epsilon = 1e-3);
    }

    #[test]
    fn use_nalgebra_3d() {
        // First one does nothing to the gradient. Still got no y or z direction.
        let rot1 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_8);
        // Second one goes pi/8 (22.5deg) about the y axis, which should shift some of the steepness
        // in the x direction into the z direction, but not much.
        let rot2 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), FRAC_PI_8);
        let rotation = (rot2 * rot1).cast::<Dual32>();
        let lifted_3d_circle = |t: Dual32| {
            let xy = unit_circle(t); // [0, 1] with derivatives [-1, 0]
            Point3::new(xy.x, xy.y, Dual32::zero())
        };
        let function = |t: Dual32| rotation * lifted_3d_circle(t);
        let point = function(Dual32::from_re(0.25).derivative());
        let real = point.map(|x| x.re);
        let grad = point.map(|x| x.eps.unwrap());
        println!("rotated point: {}", point.coords);
        approx::assert_relative_eq!(real.coords, real.coords.normalize(), epsilon = 1e-3);
        approx::assert_relative_eq!(real, Point3::new(0.146, 0.924, 0.354), epsilon = 1e-3);
        approx::assert_relative_eq!(grad, Point3::new(-5.805, 0., 2.404), epsilon = 1e-3);
    }
}
