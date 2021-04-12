use num_hyperdual::linalg::{StaticMat, StaticVec, LU};
use num_hyperdual::Dual64;

#[test]
fn test_solve_f64() {
    let a = StaticMat::new([[4.0, 3.0], [6.0, 3.0]]);
    let b = StaticVec::new_vec([10.0, 12.0]);
    let lu = LU::new(a).unwrap();
    assert_eq!(lu.determinant(), -6.0);
    assert_eq!(lu.solve(&b), StaticVec::new_vec([1.0, 2.0]));
    assert_eq!(
        lu.inverse() * lu.determinant(),
        StaticMat::new([[3.0, -3.0], [-6.0, 4.0]])
    );
}

#[test]
fn test_solve_dual64() {
    let a = StaticMat::new([
        [Dual64::new(4.0, 3.0), Dual64::new(3.0, 3.0)],
        [Dual64::new(6.0, 1.0), Dual64::new(3.0, 2.0)],
    ]);
    let b = StaticVec::new_vec([Dual64::new(10.0, 20.0), Dual64::new(12.0, 20.0)]);
    let lu = LU::new(a).unwrap();
    assert_eq!(lu.determinant(), Dual64::new(-6.0, -4.0));
    assert_eq!(
        lu.solve(&b),
        StaticVec::new_vec([Dual64::new(1.0, 2.0), Dual64::new(2.0, 1.0)])
    );
}
