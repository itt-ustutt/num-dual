# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
