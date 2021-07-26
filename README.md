# num-dual

[![crate](https://img.shields.io/crates/v/num-dual.svg)](https://crates.io/crates/num-dual)
[![documentation](https://docs.rs/num-dual/badge.svg)](https://docs.rs/num-dual)
[![minimum rustc 1.51](https://img.shields.io/badge/rustc-1.51+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
num-dual = "0.2"
```


## Example
This example defines a generic function that can be called using any (hyper) dual number and automatically calculates derivatives.
```rust
use num_dual::*;
fn f<D: DualNum<f64>>(x: D, y: D) -> D {
    x.powi(3) * y.powi(2)
}
fn main() {
    let (x, y) = (5.0, 4.0);
    // Calculate a simple derivative
    let x_dual = Dual64::from(x).derive();
    let y_dual = Dual64::from(y);
    println!("{}", f(x_dual, y_dual));                      // 2000 + [1200]ε
    // Calculate a gradient
    let xy_dual_vec = StaticVec::new_vec([x,y]).map(DualVec64::<2>::from).derive();
    println!("{}", f(xy_dual_vec[0], xy_dual_vec[1]).eps);  // [1200, 1000]
    // Calculate a Hessian
    let xy_dual2 = StaticVec::new_vec([x,y]).map(Dual2Vec64::<2>::from).derive();
    println!("{}", f(xy_dual2[0], xy_dual2[1]).v2);         // [[480, 600], [600, 250]]
    // for x=cos(t) and y=sin(t) calculate the third derivative w.r.t. t
    let t = Dual3_64::from(1.0).derive();
    println!("{}", f(t.cos(), t.sin()).v3);                 // 7.358639755305733
}
```
