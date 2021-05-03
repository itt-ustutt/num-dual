# num-hyperdual

[![crate](https://img.shields.io/crates/v/num-hyperdual.svg)](https://crates.io/crates/num-hyperdual)
[![documentation](https://docs.rs/num-hyperdual/badge.svg)](https://docs.rs/num-hyperdual)
[![minimum rustc 1.51](https://img.shields.io/badge/rustc-1.51+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
num-hyperdual = "0.1"
```


## Example
This example defines a generic function that can be called using any (hyper) dual number and automatically calculates vatives.
```rust
use num_hyperdual::*;
fn f<D: DualNum<f64>>(x: D, y: D) -> D {
    x.powi(3) * y.powi(2)
}
fn main() {
    let (x, y) = (5.0, 4.0);
    // Calculate a simple derivative
    let x_dual = Dual64::from(x).derive();
    let y_dual = Dual64::from(y);
    println!("{}", f(x_dual, y_dual));                      // 2000 + 1200ε
    // Calculate a gradient
    let x_dual2 = DualN64::<2>::from(x).derive(0);
    let y_dual2 = DualN64::<2>::from(y).derive(1);
    println!("{}", f(x_dual2, y_dual2).eps);                // [1200, 1000]
    // Calculate a Hessian
    let x_hyperdual2 = HyperDualN64::<2>::from(x).derive(0);
    let y_hyperdual2 = HyperDualN64::<2>::from(y).derive(1);
    println!("{}", f(x_hyperdual2, y_hyperdual2).hessian);  // [[480, 600], [600, 250]]
    // for x=cos(t) and y=sin(t) calculate the third derivative w.r.t. t
    let t = HD3_64::from(1.0).derive();
    println!("{}", f(t.cos(), t.sin()).v3);                 // 7.358639755305733
}
```
