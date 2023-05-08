use nalgebra::{Matrix2, Point2, Point3, SymmetricEigen, UnitQuaternion, Vector2, Vector3};
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

#[test]
fn eigenvalues() {
    let a = Dual64::from(1.0).derivative();
    let b = Dual64::from(2.0);
    let c = Dual64::from(4.0);
    let m = Matrix2::from_row_slice(&[a, b, b, c]);
    let v = SymmetricEigen::new(m).eigenvalues;
    println!("{v}");
    approx::assert_relative_eq!(v[0].re, 5.0);
    approx::assert_relative_eq!(v[0].eps.unwrap(), 0.2);
    approx::assert_relative_eq!(v[1].re, 0.0);
    approx::assert_relative_eq!(v[1].eps.unwrap(), 0.8);
}
