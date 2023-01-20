# num-dual

[![crate](https://img.shields.io/crates/v/num-dual.svg)](https://crates.io/crates/num-dual)
[![documentation](https://docs.rs/num-dual/badge.svg)](https://docs.rs/num-dual)
[![minimum rustc 1.51](https://img.shields.io/badge/rustc-1.51+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://itt-ustutt.github.io/num-dual/)
[![PyPI version](https://badge.fury.io/py/num_dual.svg)](https://badge.fury.io/py/num_dual)

Generalized, recursive, scalar and vector (hyper) dual numbers for the automatic and exact calculation of (partial) derivatives.
Including bindings for python.


## Installation and Usage

### Python

The python package can be installed directly from PyPI:
```
pip install num_dual
```
[//]: # "or from source (you need a rust compiler for that):"
[//]: # "```"
[//]: # "pip install git+https://github.com/itt-ustutt/num-dual"
[//]: # "```"

### Rust

Add this to your `Cargo.toml`:

```toml
[dependencies]
num-dual = "0.6"
```

## Example

### Python

Compute the first and second derivative of a scalar-valued function.

```python
from num_dual import derive2
import numpy as np

def f(x):
    return np.exp(x) / np.sqrt(np.sin(x)**3 + np.cos(x)**3)

x = derive2(1.5)
result = f(x)
print('f(x)    = {}'.format(result.value))
print('df/dx   = {}'.format(result.first_derivative))
print('d2f/dx2 = {}'.format(result.second_derivative))
```

### Rust
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

## Documentation

- You can find the documentation of the rust crate [here](https://docs.rs/num-dual/).
- The documentation of the python package can be found [here](https://itt-ustutt.github.io/num-dual/).

### Python

For the following commands to work you have to have the package installed (see: installing from source).

```
cd docs
make html
```
Open `_build/html/index.html` in your browser.

## Further reading

If you want to learn more about the topic of dual numbers and automatic differentiation, we have listed some useful resources for you here:

- Initial paper about hyper-dual numbers: [Fike, J. and Alonso, 2011](https://arc.aiaa.org/doi/abs/10.2514/6.2011-886)
- Website about all topics regarding automatic differentiation: [autodiff.org](http://www.autodiff.org/)
- Our paper about dual numbers in equation of state modeling: [Rehner, P. and Bauer, G., 2021](https://www.frontiersin.org/article/10.3389/fceng.2021.758090)

## Cite us

If you find `num-dual` useful for your own scientific studies, consider [citing our publication](https://www.frontiersin.org/article/10.3389/fceng.2021.758090) accompanying this library.

```
@ARTICLE{rehner2021,
    AUTHOR={Rehner, Philipp and Bauer, Gernot},
    TITLE={Application of Generalized (Hyper-) Dual Numbers in Equation of State Modeling},
    JOURNAL={Frontiers in Chemical Engineering},
    VOLUME={3},
    YEAR={2021},
    URL={https://www.frontiersin.org/article/10.3389/fceng.2021.758090},
    DOI={10.3389/fceng.2021.758090},
    ISSN={2673-2718}
}
```
