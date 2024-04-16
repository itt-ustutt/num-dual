#![cfg(feature = "serde")]
use num_dual::*;
use serde_json::Error;

#[test]
fn test_serde_dual() -> Result<(), Error> {
    let x = Dual::from_re(2.0).derivative();
    let s = serde_json::to_string(&x)?;
    println!("{s}");
    let y: Dual64 = serde_json::from_str(&s)?;
    println!("{y}");
    assert_eq!(x, y);
    Ok(())
}

#[test]
fn test_serde_dual2() -> Result<(), Error> {
    let x = Dual2::from_re(2.0).derivative();
    let s = serde_json::to_string(&x)?;
    println!("{s}");
    let y: Dual2_64 = serde_json::from_str(&s)?;
    println!("{y}");
    assert_eq!(x, y);
    Ok(())
}

#[test]
fn test_serde_dual3() -> Result<(), Error> {
    let x = Dual3::from_re(2.0).derivative();
    let s = serde_json::to_string(&x)?;
    println!("{s}");
    let y: Dual3_64 = serde_json::from_str(&s)?;
    println!("{y}");
    assert_eq!(x, y);
    Ok(())
}

#[test]
fn test_serde_hyperdual() -> Result<(), Error> {
    let x = HyperDual::from_re(2.0).derivative1().derivative2();
    let s = serde_json::to_string(&x)?;
    println!("{s}");
    let y: HyperDual64 = serde_json::from_str(&s)?;
    println!("{y}");
    assert_eq!(x, y);
    Ok(())
}

#[test]
fn test_serde_hyperhyperdual() -> Result<(), Error> {
    let x = HyperHyperDual::from_re(2.0)
        .derivative1()
        .derivative2()
        .derivative3();
    let s = serde_json::to_string(&x)?;
    println!("{s}");
    let y: HyperHyperDual64 = serde_json::from_str(&s)?;
    println!("{y}");
    assert_eq!(x, y);
    Ok(())
}
