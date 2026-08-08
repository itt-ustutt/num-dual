//! The test that is used in the README as a file that will be executed
//! to make sure that the example stays up to date.
use nalgebra::SMatrix;
use num_dual::*;

fn f<D: DualNum>(x: D, y: D) -> D {
    x.powi(3) * y.powi(2)
}

#[test]
fn main() {
    let (x, y) = (5.0, 4.0);
    // Calculate a simple derivative using dual numbers
    let x_dual = Dual64::from(x).derivative();
    let y_dual = Dual64::from(y);
    println!("{}", f(x_dual, y_dual)); // 2000 + 1200ε

    // or use the provided function instead
    let (_, df) = first_derivative(|x| f(x, y.into()), x);
    println!("{df}"); // 1200

    // Calculate a gradient
    let (value, grad) = gradient(|v| f(v[0], v[1]), &SMatrix::from([x, y]));
    println!("{value} {grad}"); // 2000 [1200, 1000]

    // Calculate a Hessian
    let (_, _, hess) = hessian(|v| f(v[0], v[1]), &SMatrix::from([x, y]));
    println!("{hess}"); // [[480, 600], [600, 250]]

    // for x=cos(t) and y=sin(t) calculate the third derivative w.r.t. t
    let (_, _, _, d3f) = third_derivative(|t| f(t.cos(), t.sin()), 1.0);
    println!("{d3f}"); // 7.358639755305733
}
