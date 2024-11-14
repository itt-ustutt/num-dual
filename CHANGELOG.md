# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.3] - 2024-11-14
## Added
- Added nalgebra compatibility for `Dual2` and `Dual2Vec`. [#81](https://github.com/itt-ustutt/num-dual/pull/81)
- Added `atan2` to the `DualNum` trait. [#85](https://github.com/itt-ustutt/num-dual/pull/85)

## [0.10.2] - 2024-11-06
## Changed
- Exposed macros for the implementation of the `DualNum` trait publicly. [#83](https://github.com/itt-ustutt/num-dual/pull/83)

## [0.10.1] - 2024-11-05
## Added
- Expose the inner type of a generalized (hyper) dual number in the `DualNum` trait. [#82](https://github.com/itt-ustutt/num-dual/pull/82)

## [0.10.0] - 2024-10-22
### Packaging
- Updated `nalgebra` dependency to 0.33. [#75](https://github.com/itt-ustutt/num-dual/pull/75)
- Updated `simba` dependency to 0.9. [#75](https://github.com/itt-ustutt/num-dual/pull/75)
- Updated `pyo3` and `numpy` dependencies to 0.22. [#80](https://github.com/itt-ustutt/num-dual/pull/80)
- Updated `ndarray` dependency to 0.16. [#80](https://github.com/itt-ustutt/num-dual/pull/80)
- Increased minimum supported Rust version to 1.81. [#77](https://github.com/itt-ustutt/num-dual/pull/77)

## Removed
- Due to limitations in the `numpy` dependency, Python wheels for 32-bit Windows are no longer supported. [#80](https://github.com/itt-ustutt/num-dual/pull/80)

## [0.9.1] - 2024-04-15
### Added
- Added `serde` feature that enables serialization and deserialization of all scalar dual numbers. [#74](https://github.com/itt-ustutt/num-dual/pull/74)

## [0.9.0] - 2024-04-11
### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.21 and adapted to the new `Bound` API.

## [0.8.1] - 2023-10-20
### Packaging
- Un-deprecated the `linalg` module.

## [0.8.0] - 2023-10-15
### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.20.

## [0.7.1] - 2023-05-31
### Fixed
- Added dedicated implementations for scalar dual numbers (`Dual`, `Dual2`, `HyperDual`) to avoid a performance regression introduced in `0.7.0`. [#68](https://github.com/itt-ustutt/num-dual/pull/68)

## [0.7.0] - 2023-05-29
### Added
- Added new `HyerHyperDual` number for the calculation of third partial derivatives. [#51](https://github.com/itt-ustutt/num-dual/pull/51)
- Added new functions `first_derivative`, `gradient`, `jacobian`, `second_derivative`, `hessian`, `third_derivative`, `second_partial_derivative`, `partial_hessian`, `third_partial_derivative` and `third_partial_derivative_vec` that provide a convenient interface for the calculation of derivatives. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Added new functions `try_first_derivative`, `try_gradient`, `try_jacobian`, `try_second_derivative`, `try_hessian`, `try_third_derivative`, `try_second_partial_derivative`, `try_partial_hessian`, `try_third_partial_derivative` and `try_third_partial_derivative_vec` that provide the same functionalities for fallible functions. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Implemented the `RealField` and `ComplexField` traits from `nalgebra` for `DualVec`. [#59](https://github.com/itt-ustutt/num-dual/pull/59)
- Added the `python_macro` feature that provides the `impl_dual_num` macro. [#63](https://github.com/itt-ustutt/num-dual/pull/63)

### Changed
- Renamed `derive*` methods to `derivative*`. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Generalized the implementation of vector dual numbers to use both statically and dynamically sized arrays internally. [#58](https://github.com/itt-ustutt/num-dual/pull/58)
- Removed `Copy`, `Send` and `Sync` as supertraits of `DualNum`. The individual dual number data types still implement the traits if they are statically allocated. [#58](https://github.com/itt-ustutt/num-dual/pull/58)
- Renamed type aliases from, e.g., `DualVec` to `DualSVec` and `DualDVec` for statically and dynamically allocated dual numbers, respectively. [#58](https://github.com/itt-ustutt/num-dual/pull/58)

### Removed
- Removed the `StaticMat` struct in favor of the analogous implementations from `nalgebra`. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Removed the `derive*` methods for vector types due to the change to `nalgebra`. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Removed the `derive*` functions in Python in favor of the aforementioned new functions. [#52](https://github.com/itt-ustutt/num-dual/pull/52)
- Removed the `build_wheel` workspace crate. The main crate is now also used to build the Python package. [#63](https://github.com/itt-ustutt/num-dual/pull/63)

## [0.6.0] - 2023-01-20
### Added
- Publicly exposed all Python classes that are being generated. [#47](https://github.com/itt-ustutt/num-dual/pull/47)
- Exported the `impl_dual_num` macro that implements the arithmetic operators for dual numbers in Python. [#47](https://github.com/itt-ustutt/num-dual/pull/47)

### Packaging
- Updated `pyo3` and `numpy` dependencies to 0.18. [#49](https://github.com/itt-ustutt/num-dual/pull/49)

## [0.5.3] - 2022-11-11
### Added
- Implemented `ScalarOperand` for non-scalar dual numbers. [#46](https://github.com/itt-ustutt/num-dual/pull/46)

## [0.5.2] - 2022-05-23
### Added
- Added method `is_derivative_zero` to check whether all non-real parts of a generalized dual number are zero. [#43](https://github.com/itt-ustutt/num-dual/pull/43)

## [0.5.1] - 2022-03-09
### Fixed
- Removed unused features and added missing `linalg` feature. [#41](https://github.com/itt-ustutt/num-dual/pull/41)

## [0.5.0] - 2022-03-08
### Packaging
- Updated `pyo3` dependency to 0.16. [#39](https://github.com/itt-ustutt/num-dual/pull/39)

### Removed
-  Removed ndarray-linalg feature. [#38](https://github.com/itt-ustutt/num-dual/pull/38)

## [0.4.1] - 2021-12-20
### Added
- Added 0th, 1st and 2nd order Bessel functions of the first kind (`bessel_j0`, `bessel_j1`, `bessel_j2`) for double precision dual numbers. [#36](https://github.com/itt-ustutt/num-dual/pull/36)

## [0.4.0] - 2021-12-16
### Added
- Implementations for LU decomposition and eigendecomposition of symmetric matrices that do not depend on external libraries (BLAS, LAPACK). [#34](https://github.com/itt-ustutt/num-dual/pull/34)

### Packaging
- Updated `pyo3` dependency to 0.15.
