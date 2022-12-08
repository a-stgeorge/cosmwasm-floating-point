use crate::float::*;
use std::cmp::Ordering;

fn approx_eq(val1: Float, val2: Float, diff: u128) {
    if val1.exponent.abs_diff(val2.exponent) > 1 {
        panic!("Exponents do not match!\n\t{:?}\n\t{:?}", val1, val2);
    }
    let delta = match val1.exponent.cmp(&val2.exponent) {
        Ordering::Greater => (val1.significand * 10) - val2.significand,
        Ordering::Less => (val2.significand * 10) - val1.significand,
        Ordering::Equal => val1.significand.abs_diff(val2.significand),
    };

    if delta > diff {
        panic!("Significant difference ({}) between values:\n\t{:?} and\n\t{:?}", delta, val1, val2);
    }
}

fn approx_eqf(val1: f64, val2: f64, error: f64) {
    if (val1 - val2).abs() / val2 >= error {
        panic!("Significant difference between values:\n\tval1: {}\n\tval2: {}", val1, val2);
    }
}

fn test_int(val1: u128, val2: u128, result: u128, operation: fn(Float, Float) -> Float) {
    assert_eq!(
        operation(Float::from_int(val1), Float::from_int(val2)),
        Float::from_int(result),
    )
}

#[test]
fn init() {
    assert_eq!(Float::PRECISION, 1_000_000_000_000_000_000); // Test relies on this precision

    assert_eq!(Float::new(5, 0).unwrap(), Float::no_fix(5_000_000_000_000_000_000, 0));
    assert_eq!(Float::new(17, 32).unwrap(), Float::no_fix(1_700_000_000_000_000_000, 33));
    assert_eq!(Float::new(0, 17).unwrap(), Float::zero());
    assert_eq!(Float::new(1, i32::MAX).unwrap(), Float::no_fix(1_000_000_000_000_000_000, i32::MAX));
    assert_eq!(Float::new(17, i32::MAX), Err(FloatRangeExceeded));
    assert_eq!(Float::new(u128::MAX, 17).unwrap(), Float::no_fix(3_402_823_669_209_384_634, 55));
    assert_eq!(
        Float::new(u128::MAX / Float::PRECISION, 0).unwrap(), 
        Float::no_fix(3_402_823_669_209_384_634, 20)
    );

    assert_eq!(Float::from_int(5), Float::no_fix(5_000_000_000_000_000_000, 0));
    assert_eq!(Float::from_int(123_456_789), Float::no_fix(1_234_567_890_000_000_000, 8));
    assert_eq!(Float::from_int(0), Float::no_fix(0, 0)); // 0's exponent must be 0
    assert_eq!(
        Float::from_int(100_000_000_000_000_000_000_000_000_000_000_000_000u128),
        Float::new(1, 38).unwrap(),
    );

    assert_eq!(Float::default(), Float::zero());
}

#[test]
fn init_float() {
    assert_eq!(Float::from_float(0.1f64), Float::no_fix(1_000_000_000_000_000_000, -1));
    assert_eq!(Float::from_float(10.0f64), Float::no_fix(1_000_000_000_000_000_000, 1));
    assert_eq!(Float::from_float(0.4), Float::no_fix(4_000_000_000_000_000_000, -1));
    assert_eq!(Float::from_float(0.2), Float::no_fix(2_000_000_000_000_000_000, -1));
    assert_eq!(Float::from_float(0.0001f64), Float::no_fix(1_000_000_000_000_000_000, -4));
    assert_eq!(Float::from_float(0.0004f64), Float::no_fix(4_000_000_000_000_000_000, -4));
    approx_eq(
        Float::from_float(0.0432), 
        Float::no_fix(4_320_000_000_000_000_000, -2),
        550,
    );
    assert_eq!(Float::from_float(-1000.001), Float::zero());
    assert_eq!(Float::from(0.0), Float::zero());
    approx_eq(
        Float::from_float(0.0000001234567890123456789111), 
        Float::no_fix(1_234_567_890_123_456_789, -7),
        1000,
    );
    approx_eq(
        Float::from_float(0.000000000000000000000000000000000000000000000000000000000001f64),
        Float::no_fix(1_000_000_000_000_000_000, -60),
        1500,
    );

    approx_eq(Float::from_float(1000000000000000000000000000000011f64), Float::new(1, 33).unwrap(), 3000);
    approx_eq(Float::from_float(10f64.powi(308)), Float::new(1, 308).unwrap(), 0);
    approx_eq(Float::from_float(10f64.powi(-308)), Float::new(1, -308).unwrap(), 0);
    approx_eq(Float::from_float(f64::MAX), Float::no_fix(1_797_693_134_862_315_700, 308), 10000);
    approx_eq(Float::from_float(f64::MIN_POSITIVE), Float::no_fix(2_225_073_858_507_201_400, -308), 10000);
}

#[test]
fn init_ratio() {
    assert_eq!(Float::from_ratio(1, 3), Float::no_fix(3_333_333_333_333_333_333, -1));
    assert_eq!(Float::from_ratio(1000, 3), Float::no_fix(3_333_333_333_333_333_333, 2));
    assert_eq!(Float::from_ratio(1, 3000), Float::no_fix(3_333_333_333_333_333_333, -4));
    assert_eq!(Float::from_ratio(1, 9000000000000000000), Float::no_fix(1_111_111_111_111_111_111, -19));
    assert_eq!(Float::from_ratio(1000000000000000000, 9), Float::no_fix(1_111_111_111_111_111_111, 17));
    assert_eq!(Float::from_ratio(58974032, 17), Float::no_fix(3_469_060_705_882_352_941, 6));
    assert_eq!(Float::from_ratio(0, 473821957), Float::zero());
    assert_eq!(Float::from_ratio(u128::MAX, 1), Float::from_int(u128::MAX));
    assert_eq!(Float::from_ratio(1, u128::MAX), Float::no_fix(2_938_735_877_055_718_770, -39));
    assert_eq!(Float::from_ratio(u128::MAX, u128::MAX), Float::one());

    assert_eq!(Float::checked_from_ratio(1, 3).unwrap(), Float::from_ratio(1, 3));
    assert_eq!(Float::checked_from_ratio(1000, 3).unwrap(), Float::from_ratio(1000, 3));
    assert_eq!(Float::checked_from_ratio(1, 3000).unwrap(), Float::from_ratio(1, 3000));
    assert_eq!(Float::checked_from_ratio(58974032, 17).unwrap(), Float::from_ratio(58974032, 17));
    assert_eq!(Float::checked_from_ratio(0, 473821957).unwrap(), Float::from_ratio(0, 473821957));
    assert_eq!(Float::checked_from_ratio(u128::MAX, 1).unwrap(), Float::from_ratio(u128::MAX, 1));
    assert_eq!(Float::checked_from_ratio(1, u128::MAX).unwrap(), Float::from_ratio(1, u128::MAX));
    assert_eq!(Float::checked_from_ratio(17, 0), Err(CheckedFromRatioError::DivideByZero));
}

#[test]
fn init_percent() {
    assert_eq!(Float::percent(5), Float::no_fix(5_000_000_000_000_000_000, -2));
    assert_eq!(Float::percent(100), Float::no_fix(1_000_000_000_000_000_000, 0));
    assert_eq!(Float::percent(1_000_000_000_000_000_000), Float::no_fix(1_000_000_000_000_000_000, 16));
    assert_eq!(Float::percent(0), Float::no_fix(0, 0));
    assert_eq!(Float::percent(u128::MAX), Float::no_fix(3_402_823_669_209_384_634, 36));

    assert_eq!(Float::permille(5), Float::no_fix(5_000_000_000_000_000_000, -3));
    assert_eq!(Float::permille(1000), Float::no_fix(1_000_000_000_000_000_000, 0));
    assert_eq!(Float::permille(1_000_000_000_000_000_000), Float::no_fix(1_000_000_000_000_000_000, 15));
    assert_eq!(Float::permille(0), Float::no_fix(0, 0));
    assert_eq!(Float::permille(u128::MAX), Float::no_fix(3_402_823_669_209_384_634, 35));
}

#[test]
fn init_cosmwasm() {
    assert_eq!(Float::new_uint(Uint128::new(5), 0).unwrap(), Float::no_fix(5_000_000_000_000_000_000, 0));
    assert_eq!(Float::new_uint(Uint128::new(17), 32).unwrap(), Float::no_fix(1_700_000_000_000_000_000, 33));
    assert_eq!(Float::new_uint(Uint128::new(0), 17).unwrap(), Float::zero());
    assert_eq!(Float::new_uint(Uint128::new(17), i32::MAX), Err(FloatRangeExceeded));
    assert_eq!(Float::new_uint(Uint128::MAX, 17).unwrap(), Float::no_fix(3_402_823_669_209_384_634, 55));

    assert_eq!(Float::from(Uint128::new(5)), Float::no_fix(5_000_000_000_000_000_000, 0));
    assert_eq!(Float::from(Uint128::new(123_456_789)), Float::no_fix(1_234_567_890_000_000_000, 8));
    assert_eq!(Float::from(Uint128::new(0)), Float::no_fix(0, 0)); // 0's exponent must be 0
    assert_eq!(
        Float::from(Uint128::new(100_000_000_000_000_000_000_000_000_000_000_000_000u128)),
        Float::new(1, 38).unwrap(),
    );

    assert_eq!(Float::from(Uint256::from(5u128)), Float::no_fix(5_000_000_000_000_000_000, 0));
    assert_eq!(Float::from(Uint256::from(123_456_789u128)), Float::no_fix(1_234_567_890_000_000_000, 8));
    assert_eq!(Float::from(Uint256::from(0u128)), Float::no_fix(0, 0)); // 0's exponent must be 0
    assert_eq!(
        Float::from(Uint256::from(100_000_000_000_000_000_000_000_000_000_000_000_000u128)),
        Float::new(1, 38).unwrap(),
    );
    assert_eq!(Float::from(Uint256::MAX), Float::no_fix(1_157_920_892_373_161_954, 77));

    assert_eq!(Float::from(Decimal::percent(100)), Float::new(1, 0).unwrap());
    assert_eq!(Float::from(Decimal::percent(1)), Float::new(1, -2).unwrap());
    assert_eq!(Float::from(Decimal::from_atomics(1234u128, 5).unwrap()), Float::new(1234, -5).unwrap());
    assert_eq!(Float::from(Decimal::zero()), Float::zero());
    assert_eq!(Float::from(Decimal::MAX), Float::new(u128::MAX, -18).unwrap());

    assert_eq!(Float::from(Decimal256::percent(100)), Float::new(1, 0).unwrap());
    assert_eq!(Float::from(Decimal256::percent(1)), Float::new(1, -2).unwrap());
    assert_eq!(Float::from(Decimal256::from_atomics(1234u128, 5).unwrap()), Float::new(1234, -5).unwrap());
    assert_eq!(Float::from(Decimal256::zero()), Float::zero());
    assert_eq!(Float::from(Decimal256::MAX), Float::from(Uint256::MAX) / Float::from_int(10).pow(18));
}

#[test]
#[should_panic]
fn from_ratio_zero() {
    Float::from_ratio(17, 0);
}

#[test]
fn float_out_int() {
    assert_eq!(u128::try_from(Float::from_int(5)).unwrap(), 5u128);
    assert_eq!(u128::try_from(Float::from_int(123456789)).unwrap(), 123456789u128);
    assert_eq!(u128::try_from(Float::zero()).unwrap(), 0);
    assert_eq!(
        u128::try_from(Float::from_int(100_000_000_000_000_000_000_000_000_000_000_000_000u128)).unwrap(),
        100_000_000_000_000_000_000_000_000_000_000_000_000u128,
    );
    // right to the first 18 digits
    assert_eq!(
        u128::try_from(Float::from_int(u128::MAX)).unwrap(), 
        u128::MAX / 100_000_000_000_000_000_000 * 100_000_000_000_000_000_000,
    );
    assert_eq!(
        u128::try_from(Float::MAX), 
        Err(ConversionOverflowError::new("Float", "u128", Float::MAX.to_string())),
    );
    assert_eq!(u128::try_from(Float::from_float(0.001)).unwrap(), 0);
    assert_eq!(u128::try_from(Float::MIN).unwrap(), 0);
    let round = Uint256::from(10u128).pow(59);
    // right to the frist 18 digits
    assert_eq!(
        Uint256::try_from(Float::from(Uint256::MAX)).unwrap(),
        Uint256::MAX / round * round,
    );
}

#[test]
fn float_out_decimal() {
    assert_eq!(Decimal::try_from(Float::from(Decimal::percent(100))).unwrap(), Decimal::percent(100));
    assert_eq!(Decimal::try_from(Float::from(Decimal::percent(1))).unwrap(), Decimal::percent(1));
    assert_eq!(
        Decimal::try_from(Float::from(Decimal::from_atomics(1234u128, 5).unwrap())).unwrap(),
        Decimal::from_atomics(1234u128, 5).unwrap()
    );
    assert_eq!(Decimal::try_from(Float::from(Decimal::zero())).unwrap(), Decimal::zero());
    let round = Decimal::from_atomics(100_000_000_000_000_000_000u128, 0).unwrap();
    assert_eq!(
        Decimal::try_from(Float::from(Decimal::MAX)).unwrap(),
        Decimal::MAX / round * round,
    );

    assert_eq!(Decimal256::try_from(Float::from(Decimal256::percent(100))).unwrap(), Decimal256::percent(100));
    assert_eq!(Decimal256::try_from(Float::from(Decimal256::percent(1))).unwrap(), Decimal256::percent(1));
    assert_eq!(
        Decimal256::try_from(Float::from(Decimal256::from_atomics(1234u128, 5).unwrap())).unwrap(),
        Decimal256::from_atomics(1234u128, 5).unwrap(),
    );
    let round = Decimal256::from_atomics(100_000_000_000_000_000_000_000_000_000u128, 0).unwrap();
    let round = round * round * Decimal256::from_atomics(10u128, 0).unwrap();
    assert_eq!(
        Decimal256::try_from(Float::from(Decimal256::MAX)).unwrap(), 
        Decimal256::MAX / round * round,
    );
}

#[test]
fn float_out_float() {
    let error = 0.000000000000001f64; // Close enough
    approx_eqf(f64::from(Float::from_float(0.1f64)), 0.1, error);
    approx_eqf(f64::from(Float::from_float(10.0f64)), 10.0, error);
    approx_eqf(f64::from(Float::from_float(0.4)), 0.4, error);
    approx_eqf(f64::from(Float::from_float(0.2)), 0.2, error);
    approx_eqf(f64::from(Float::from_float(0.0001f64)), 0.0001, error);
    approx_eqf(f64::from(Float::from_float(0.0004f64)), 0.0004, error);
    approx_eqf(
        f64::from(Float::from_float(0.0432)), 
        0.0432,
        error,
    );
    approx_eqf(f64::from(Float::from_float(-1000.001)), 0f64, error);
    approx_eqf(f64::from(Float::from(0.0)), 0f64, error);
    approx_eqf(
        f64::from(Float::from_float(0.0000001234567890123456789111)), 
        0.0000001234567890123456789,
        error,
    );
    approx_eqf(
        f64::from(Float::from_float(0.000000000000000000000000000000000000000000000000000000000001f64)),
        0.000000000000000000000000000000000000000000000000000000000001f64,
        error,
    );

    approx_eqf(
        f64::from(Float::from_float(1000000000000000000000000000000011f64)), 
        1000000000000000000000000000000011f64,
        error,
    );
    approx_eqf(f64::from(Float::from_float(f64::MAX)), f64::MAX, error);
    approx_eqf(f64::from(Float::from_float(f64::MIN_POSITIVE)), f64::MIN, error);
}

#[test]
fn add() {
    let add: fn(Float, Float) -> Float = |x, y| x + y;

    // Happy Tests
    test_int(1, 1, 2, add);
    test_int(4, 7, 11, add);
    test_int(7, 4, 11, add);
    test_int(10, 3, 13, add);
    test_int(3, 10, 13, add);
    test_int(100000, 5000, 105000, add);
    test_int(5000, 100000, 105000, add);
    test_int(5, 0, 5, add);
    test_int(10000, 0, 10000, add);
    test_int(0, 10000, 10000, add);

    // Loss of precision tests
    test_int(1234567890234567890, 0, 1234567890234567890, add);
    test_int(1234567890234567890, 1, 1234567890234567891, add);
    test_int(9999999999999999999, 9, 10000000000000000000, add); // Last 8 truncated
    test_int(1000000000000000000, 1, 1000000000000000001, add);
    test_int(1, 1000000000000000000, 1000000000000000001, add);

    assert_eq!(Float::new(1, 1000).unwrap() + Float::new(1, 0).unwrap(), Float::new(1, 1000).unwrap());
    assert_eq!(Float::new(1, 0).unwrap() + Float::new(1, 1000).unwrap(), Float::new(1, 1000).unwrap());

    // Checked tests
    let checked_add: fn(Float, Float) -> Float = |x, y| x.checked_add(y).unwrap();
    test_int(1, 1, 2, checked_add);
    test_int(4, 7, 11, checked_add);
    test_int(7, 4, 11, checked_add);
    test_int(10, 3, 13, checked_add);
    test_int(3, 10, 13, checked_add);
    test_int(100000, 5000, 105000, checked_add);
    test_int(5000, 100000, 105000, checked_add);
    test_int(5, 0, 5, checked_add);
    test_int(10000, 0, 10000, checked_add);
    test_int(0, 10000, 10000, checked_add);

    // Loss of precision tests
    test_int(1234567890234567890, 0, 1234567890234567890, checked_add);
    test_int(1234567890234567890, 1, 1234567890234567891, checked_add);
    test_int(9999999999999999999, 9, 10000000000000000000, checked_add); // Last 8 truncated
    test_int(1000000000000000000, 1, 1000000000000000001, checked_add);
    test_int(1, 1000000000000000000, 1000000000000000001, checked_add);

    assert_eq!(
        Float::new(1, 1000).unwrap().checked_add(Float::new(1, 0).unwrap()).unwrap(),
        Float::new(1, 1000).unwrap()
    );
    assert_eq!(
        Float::new(1, 0).unwrap().checked_add(Float::new(1, 1000).unwrap()).unwrap(),
        Float::new(1, 1000).unwrap(),
    );
}

#[test]
#[should_panic]
fn add_panic() {
    let _err = Float::new(5, i32::MAX).unwrap() + Float::new(5, i32::MAX).unwrap();
}

#[test]
fn subtract() {
    let sub: fn(Float, Float) -> Float = |x, y| x - y;

    // Happy Tests
    test_int(2, 1, 1, sub);
    test_int(11, 4, 7, sub);
    test_int(11, 7, 4, sub);
    test_int(13, 10, 3, sub);
    test_int(13, 3, 10, sub);
    test_int(105000, 5000, 100000, sub);
    test_int(105000, 100000, 5000, sub);

    test_int(5, 0, 5, sub);
    test_int(10000, 0, 10000, sub);
    test_int(100000, 99999, 1, sub);

    // Loss of precision tests
    test_int(1234567890234567890, 0, 1234567890234567890, sub);
    test_int(1234567890234567890, 1, 1234567890234567889, sub);
    test_int(10000000000000000000, 9, 10000000000000000000, sub); // 9 gets truncated
    test_int(1000000000000000000, 1, 999999999999999999, sub);

    assert_eq!(Float::new(1, 1000).unwrap() - Float::new(1, 0).unwrap(), Float::new(1, 1000).unwrap());

    // Checked tests
    let checked_sub: fn(Float, Float) -> Float = |x, y| x.checked_sub(y).unwrap();
    test_int(2, 1, 1, checked_sub);
    test_int(11, 4, 7, checked_sub);
    test_int(11, 7, 4, checked_sub);
    test_int(13, 10, 3, checked_sub);
    test_int(13, 3, 10, checked_sub);
    test_int(105000, 5000, 100000, checked_sub);
    test_int(105000, 100000, 5000, checked_sub);

    test_int(5, 0, 5, checked_sub);
    test_int(10000, 0, 10000, checked_sub);
    test_int(100000, 99999, 1, checked_sub);

    // Loss of precision tests
    test_int(1234567890234567890, 0, 1234567890234567890, checked_sub);
    test_int(1234567890234567890, 1, 1234567890234567889, checked_sub);
    test_int(10000000000000000000, 9, 10000000000000000000, checked_sub); // 9 gets truncated
    test_int(1000000000000000000, 1, 999999999999999999, checked_sub);

    assert_eq!(
        Float::new(1, 1000).unwrap().checked_sub(Float::new(1, 0).unwrap()).unwrap(),
        Float::new(1, 1000).unwrap(),
    );

    assert_eq!(
        Float::from(5u32).checked_sub(Float::from(6u32)),
        Err(OverflowError::new(OverflowOperation::Sub, Float::from(5u32), Float::from(6u32))),
    );
    assert_eq!(
        Float::zero().checked_sub(Float::from(0.01)),
        Err(OverflowError::new(OverflowOperation::Sub, Float::zero(), Float::from(0.01))),
    );

    assert_eq!(
        Float::from(0.01).checked_sub(Float::zero()).unwrap(),
        Float::from(0.01),
    );

}

#[test]
#[should_panic]
fn subtract_panic1() {
    let _err = Float::from_int(1) - Float::from_int(1_000_000);
}

#[test]
#[should_panic]
fn subtract_panic2() {
    let _err = Float::from_int(4) - Float::from_int(7);
}

#[test]
fn multiply() {
    let mul: fn(Float, Float) -> Float = |x, y| x * y;

    // Happy Tests
    test_int(2, 3, 6, mul);
    test_int(3, 2, 6, mul);
    test_int(3, 6, 18, mul);
    test_int(9, 9, 81, mul);
    test_int(2, 30, 60, mul);
    test_int(20, 3, 60, mul);
    test_int(20, 30, 600, mul);
    test_int(30, 600, 18000, mul);
    test_int(300, 600, 180000, mul);
    test_int(2, 5, 10, mul);
    test_int(3_333_333_333_333_333_333, 3, 9_999_999_999_999_999_999, mul);

    assert_eq!(
        Float::from_int(9_999_999_999_999_999_999) * Float::from_int(9_999_999_999_999_999_999),
        Float::from_int(99_999_999_999_999_999_980_000_000_000_000_000_000),
    );

    // The power of floats
    let almost_max_int = 100_000_000_000_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(
        Float::from_int(almost_max_int) * Float::from_int(almost_max_int),
        Float::new(1, 76).unwrap(),
    );

    // Checked Tests
    let checked_mul: fn(Float, Float) -> Float = |x, y| x.checked_mul(y).unwrap();
    test_int(2, 3, 6, checked_mul);
    test_int(3, 2, 6, checked_mul);
    test_int(3, 6, 18, checked_mul);
    test_int(9, 9, 81, checked_mul);
    test_int(2, 30, 60, checked_mul);
    test_int(20, 3, 60, checked_mul);
    test_int(20, 30, 600, checked_mul);
    test_int(30, 600, 18000, checked_mul);
    test_int(300, 600, 180000, checked_mul);
    test_int(2, 5, 10, checked_mul);
    test_int(3_333_333_333_333_333_333, 3, 9_999_999_999_999_999_999, checked_mul);

    assert_eq!(
        Float::from_int(9_999_999_999_999_999_999).checked_mul(Float::from_int(9_999_999_999_999_999_999)).unwrap(),
        Float::from_int(99_999_999_999_999_999_980_000_000_000_000_000_000),
    );

    // The power of floats
    let almost_max_int = 100_000_000_000_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(
        Float::from_int(almost_max_int).checked_mul(Float::from_int(almost_max_int)).unwrap(),
        Float::new(1, 76).unwrap(),
    );
}

#[test]
#[should_panic]
fn multiply_panic() {
    //let _err = Float::new(1, i32::MAX).unwrap() * Float::new(1, i32::MAX).unwrap();
    let _err = Float::new(5, i32::MAX).unwrap() * Float::new(2, 0).unwrap();
}

#[test]
fn power() {
    power_cases(|x, y| x.pow(y));
    power_cases(|x, y| x.checked_pow(y).unwrap());

    // checked_pow Errors
    assert_eq!(
        Float::from_int(11).checked_pow(i32::MAX),
        Err(OverflowError::new(OverflowOperation::Pow, Float::from_int(11), i32::MAX)),
    );
    assert_eq!(
        Float::MAX.checked_pow(i32::MAX),
        Err(OverflowError::new(OverflowOperation::Pow, Float::MAX, i32::MAX)),
    );
}

fn power_cases(pow: fn(Float, i32) -> Float) {
    assert_eq!(pow(Float::from_int(2), 2), Float::from_int(4));
    assert_eq!(pow(Float::from_int(2), 3), Float::from_int(8));
    assert_eq!(pow(Float::from_int(2), 64), Float::from_int(2u128.pow(64)));
    assert_eq!(pow(Float::from_int(17), 17), Float::from_int(17u128.pow(17)));
    approx_eq(pow(Float::from_int(17), 314), Float::no_fix(2_295_944_114_875_826_228, 386), 10);
    
    assert_eq!(pow(Float::zero(), i32::MAX), Float::zero());
    assert_eq!(pow(Float::one(), i32::MAX), Float::one());
    assert_eq!(pow(Float::zero(), 0), Float::one());
    assert_eq!(pow(Float::from_int(17), 0), Float::one());

    assert_eq!(pow(Float::from_int(2), -1), Float::from(0.5));
    assert_eq!(pow(Float::from_int(2), -17), Float::no_fix(7_629_394_531_250_000_000, -6));
    assert_eq!(pow(Float::from_int(17), -17), Float::no_fix(1_208_838_648_302_396_713, -21));
    approx_eq(pow(Float::from_int(17), -314), Float::no_fix(4_355_506_710_815_929_287, -387), 25);

    assert_eq!(
        pow(Float::from_int(Float::MAX_SIGNIFICAND), 100_000_000),
        Float::no_fix(9_999_999_999_900_000_000, 1899999999)
    );
    approx_eq(
        pow(Float::from_int(17), -1_000_000_000),
        Float::no_fix(4_185_294_969_729_125_437, -1230448922),
        60_000_000, // very imprecise calculation
    );
}

#[test]
#[should_panic]
fn power_panic() {
    Float::from_int(11).pow(i32::MAX);
}

#[test]
fn divide() {
    let div: fn(Float, Float) -> Float = |x, y| x / y;
    
    // Happy Tests
    test_int(6, 2, 3, div);
    test_int(6, 3, 2, div);
    test_int(18, 3, 6, div);
    test_int(100, 20, 5, div);
    test_int(60, 2, 30, div);
    test_int(600, 20, 30, div);
    test_int(18000, 3, 6000, div);
    test_int(18000, 3000, 6, div);
    test_int(180000, 300, 600, div);
    test_int(Float::MAX_SIGNIFICAND, 3, 3_333_333_333_333_333_333, div);

    let almost_max_int = 100_000_000_000_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(
        Float::from_int(almost_max_int) / Float::from_int(almost_max_int),
        Float::new(1, -0).unwrap(),
    );
    assert_eq!(Float::MAX / Float::MAX, Float::one());
    assert_eq!(Float::one() / Float::from(0.01), Float::from_int(100));
    approx_eq((Float::MAX / Float::from_int(2)) / Float::from(0.5), Float::MAX, 1); // Division rounding
    assert_eq!((Float::MIN * Float::from_int(2)) / Float::from_int(2), Float::MIN); // slight bug

    // Checked tests
    let checked_div: fn(Float, Float) -> Float = |x, y| x.checked_div(y).unwrap();
    test_int(6, 2, 3, checked_div);
    test_int(6, 3, 2, checked_div);
    test_int(18, 3, 6, checked_div);
    test_int(100, 20, 5, checked_div);
    test_int(60, 2, 30, checked_div);
    test_int(600, 20, 30, checked_div);
    test_int(18000, 3, 6000, checked_div);
    test_int(18000, 3000, 6, checked_div);
    test_int(180000, 300, 600, checked_div);
    test_int(Float::MAX_SIGNIFICAND, 3, 3_333_333_333_333_333_333, checked_div);

    let almost_max_int = 100_000_000_000_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(
        Float::from_int(almost_max_int).checked_div(Float::from_int(almost_max_int)).unwrap(),
        Float::new(1, -0).unwrap(),
    );
    assert_eq!(Float::MAX.checked_div(Float::MAX).unwrap(), Float::one());
    assert_eq!(Float::one().checked_div(Float::from(0.01)).unwrap(), Float::from_int(100));
    approx_eq((Float::MAX / Float::from_int(2)).checked_div(Float::from(0.5)).unwrap(), Float::MAX, 1); // Division rounding
    assert_eq!((Float::MIN * Float::from_int(2)).checked_div(Float::from_int(2)).unwrap(), Float::MIN); // slight bug
}

#[test]
fn remainder() {
    let rem: fn(Float, Float) -> Float = |x, y| x % y;

    // Happy Tests
    test_int(5, 2, 1, rem);
    test_int(12, 7, 5, rem);
    test_int(10000000000000, 3, 1, rem);
    test_int(57843290, 1000, 290, rem);
    
    approx_eq(
        Float::from(65743829.5432) % Float::from_float(0.1),
        Float::from(0.0432),
        1000,
    );

    let almost_max_int = 100_000_000_000_000_000_000_000_000_000_000_000_000u128;
    assert_eq!(
        Float::from_int(almost_max_int) / Float::from_int(almost_max_int),
        Float::new(1, -0).unwrap(),
    );
    assert_eq!(Float::MAX % Float::one(), Float::zero());
    assert_eq!(Float::MAX % Float::MIN, Float::zero());
    assert_eq!(Float::from_int(5) % Float::zero(), Float::zero());
}

#[test]
fn assigns() {
    let mut sum = Float::from_int(1);
    sum += Float::from_int(1);
    assert_eq!(sum, Float::from_int(2));

    let mut sub = Float::from_int(1);
    sub -= Float::from_int(1);
    assert_eq!(sub, Float::from_int(0));

    let mut mul = Float::from_int(2);
    mul *= Float::from_int(3);
    assert_eq!(mul, Float::from_int(6));

    let mut div = Float::from_int(6);
    div /= Float::from_int(3);
    assert_eq!(div, Float::from_int(2));

    let mut div = Float::from_int(5);
    div %= Float::from_int(2);
    assert_eq!(div, Float::from_int(1));
}

#[test]
fn square_root() {
    assert_eq!(Float::zero().sqrt(), Float::zero());
    assert_eq!(Float::one().sqrt(), Float::one());
    assert_eq!(Float::from_int(4).sqrt(), Float::from_int(2));
    assert_eq!(Float::from_int(16).sqrt(), Float::from_int(4));
    assert_eq!(Float::from_int(81).sqrt(), Float::from_int(9));
    assert_eq!(Float::from_int(121).sqrt(), Float::from_int(11));
    assert_eq!(Float::from_int(2).sqrt(), Float::no_fix(1414213562373095048, 0)); // Rounding issue
    assert_eq!(Float::from_int(10).sqrt(), Float::no_fix(3162277660168379331, 0));
    assert_eq!(Float::from_int(160).sqrt(), Float::no_fix(1264911064067351732, 1));
    assert_eq!(Float::from_int(100000000).sqrt(), Float::from_int(10000));
    assert_eq!(Float::from_int(1000000000).sqrt(), Float::no_fix(3162277660168379331, 4));
    assert_eq!(Float::MAX.sqrt(), Float::no_fix(9999999999999999999, 1073741823));

    assert_eq!(Float::from_float(0.5).sqrt(), Float::no_fix(7071067811865475244, -1));
    assert_eq!(Float::from_float(0.25).sqrt(), Float::from_float(0.5));
    assert_eq!(Float::from_float(0.1).sqrt(), Float::no_fix(3162277660168379331, -1));
    assert_eq!(Float::from_float(0.01).sqrt(), Float::from_float(0.1));
    assert_eq!(Float::new(5, -8).unwrap().sqrt(), Float::no_fix(2236067977499789696, -4));
    assert_eq!(Float::new(5, -9).unwrap().sqrt(), Float::no_fix(7071067811865475244, -5));
    assert_eq!(Float::MIN.sqrt(), Float::new(1, -1073741824).unwrap());
}

#[test]
fn display() {
    assert_eq!(format!("{}", Float::new(1, 8).unwrap()), "100000000.");
    assert_eq!(format!("{}", Float::new(1, 9).unwrap()), "1000000000.");
    assert_eq!(format!("{}", Float::new(1, 10).unwrap()), "1.e10");
    assert_eq!(format!("{}", Float::new(1_000_001, -3).unwrap()), "1000.001");
    assert_eq!(format!("{}", Float::new(1, -8).unwrap()), "0.00000001");
    assert_eq!(format!("{}", Float::new(1, -9).unwrap()), "0.000000001");
    assert_eq!(format!("{}", Float::new(1, -10).unwrap()), "1.e-10");
    assert_eq!(format!("{}", Float::from_int(123456789012345678)), "1.23456789012345678e17");
    assert_eq!(format!("{}", Float::from_int(1234567890123456789123)), "1.234567890123456789e21");
    assert_eq!(format!("{}", Float::from_float(0.1)), "0.1");
    assert_eq!(format!("{}", Float::from_float(0.000001)), "0.000001");
    assert_eq!(format!("{}", Float::new(123456789, -8).unwrap()), "1.23456789");
    assert_eq!(format!("{}", Float::new(123456789, -9).unwrap()), "0.123456789");
    assert_eq!(format!("{}", Float::new(123456789, -10).unwrap()), "1.23456789e-2");
    assert_eq!(format!("{}", Float::new(1234567890123456789, -22 - 18).unwrap()), "1.234567890123456789e-22");
    assert_eq!(format!("{}", Float::MAX), "9.999999999999999999e2147483647");
    assert_eq!(format!("{}", Float::MIN), "1.e-2147483648");
}

#[test]
fn floor_ciel() {
    assert_eq!(Float::from_float(10.543).floor(), Float::from_int(10));
    assert_eq!(Float::from_float(123456789.12345).floor(), Float::from_int(123456789));
    approx_eq(
        Float::from_float(1234567890123456789.12345).floor(), 
        Float::from_int(1234567890123456789), 
        500,
        );
    approx_eq(
        Float::from_float(12345678901234567891234.5).floor(), 
        Float::from_int(12345678901234567890000), 
        500,
        );
    assert_eq!(Float::from_float(0.001).floor(), Float::zero());
    assert_eq!(Float::from_int(19).floor(), Float::from_int(19));
    assert_eq!(Float::MAX.floor(), Float::MAX);

    assert_eq!(Float::from_float(10.543).ciel(), Float::from_int(11));
    assert_eq!(Float::from_float(123456789.12345).ciel(), Float::from_int(123456790));
    approx_eq(
        Float::from_float(1234567890123456789.12345).ciel(), 
        Float::from_int(1234567890123456790), 
        500,
    );
    approx_eq(
        Float::from_float(12345678901234567891234.5).ciel(), 
        Float::from_int(12345678901234567900000), 
        500
    );
    assert_eq!(Float::from_float(0.001).ciel(), Float::one());
    assert_eq!(Float::from_int(19).ciel(), Float::from_int(19));
    assert_eq!(Float::from_float(999.9).ciel(), Float::new(1, 3).unwrap());
    assert_eq!(Float::MAX.ciel(), Float::MAX);
}

#[test]
fn absolute_difference() {
    let abs_diff: fn(Float, Float) -> Float = |x, y| x.abs_diff(y);

    // Happy Tests
    test_int(2, 1, 1, abs_diff);
    test_int(1, 2, 1, abs_diff);
    test_int(11, 4, 7, abs_diff);
    test_int(4, 11, 7, abs_diff);
    test_int(13, 10, 3, abs_diff);
    test_int(10, 13, 3, abs_diff);
    test_int(105000, 5000, 100000, abs_diff);
    test_int(5000, 105000, 100000, abs_diff);
    test_int(105000, 100000, 5000, abs_diff);
    test_int(100000, 105000, 5000, abs_diff);

    test_int(10000, 0, 10000, abs_diff);
    test_int(0, 10000, 10000, abs_diff);
    test_int(100000, 99999, 1, abs_diff);
    test_int(99999, 100000, 1, abs_diff);

    // Loss of precision tests
    test_int(1234567890234567890, 1, 1234567890234567889, abs_diff);
    test_int(1, 1234567890234567890, 1234567890234567889, abs_diff);
    test_int(10000000000000000000, 9, 10000000000000000000, abs_diff); // 9 gets truncated
    test_int(9, 10000000000000000000, 10000000000000000000, abs_diff); // 9 gets truncated
    test_int(1000000000000000000, 1, 999999999999999999, abs_diff);
    test_int(1, 1000000000000000000, 999999999999999999, abs_diff);

    assert_eq!(Float::new(1, 1000).unwrap().abs_diff(Float::one()), Float::new(1, 1000).unwrap());
    assert_eq!(Float::one().abs_diff(Float::new(1, 1000).unwrap()), Float::new(1, 1000).unwrap());
}

#[test]
fn ordering() {
    assert!(Float::one() == Float::one());
    assert!(Float::new(0, 0).unwrap() == Float::new(0, 10).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() > Float::new(1234567890123456789, 16).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() >= Float::new(1234567890123456789, 16).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() < Float::new(1234567890123456789, 18).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() <= Float::new(1234567890123456789, 18).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() > Float::new(1234567890000000000, 17).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() >= Float::new(1234567890000000000, 17).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() < Float::new(1234567891000000000, 17).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() <= Float::new(1234567891000000000, 17).unwrap());
    assert!(Float::new(1234567890123456789, 17).unwrap() == Float::new(1234567890123456789, 17).unwrap());
    
    assert!(Float::zero() < Float::one());
    assert!(Float::one() > Float::zero());
    assert!(Float::from(100u32) > Float::one());
    assert!(Float::from(0.01) < Float::one());
    assert!(Float::from(0.0005) < Float::from(0.0006));
}

#[test]
fn checked() {
    // Add
    let half_max: Float = Float::new(5, i32::MAX).unwrap();
    assert_eq!(
        half_max.checked_add(half_max), 
        Err(OverflowError::new(OverflowOperation::Add, half_max, half_max))
    );
    assert_eq!(
        Float::from_int(10_000_000_000_000_000_000).checked_add(Float::one()),
        Ok(Float::from_int(10_000_000_000_000_000_000))
    );
    assert_eq!(
        Float::new(999, i32::MAX - 2).unwrap().checked_add(Float::new(2, i32::MAX - 2).unwrap()),
        Err(OverflowError::new(
            OverflowOperation::Add,
            Float::new(999, i32::MAX - 2).unwrap(),
            Float::new(2, i32::MAX - 2).unwrap(),
        )),
    );
    assert_eq!(
        Float::one().checked_add(Float::from_int(10_000_000_000_000_000_000)),
        Ok(Float::from_int(10_000_000_000_000_000_000))
    );
    assert_eq!(
        Float::new(999, i32::MAX - 2).unwrap().checked_add(Float::new(2, i32::MAX - 2).unwrap()),
        Err(OverflowError::new(
            OverflowOperation::Add,
            Float::new(999, i32::MAX - 2).unwrap(),
            Float::new(2, i32::MAX - 2).unwrap(),
        )),
    );

    // Subtract
    assert_eq!(
        Float::new(3, i32::MIN).unwrap().checked_sub(Float::new(29, i32::MIN + 1).unwrap()),
        Err(OverflowError::new(
            OverflowOperation::Sub,
            Float::new(3, i32::MIN).unwrap(),
            Float::new(29, i32::MIN + 1).unwrap(),
        )),
    );
    assert_eq!(
        Float::from_int(10_000_000_000_000_000_000).checked_sub(Float::one()),
        Ok(Float::from_int(10_000_000_000_000_000_000)),
    );
    assert_eq!(
        Float::one().checked_sub(Float::from_int(10)),
        Err(OverflowError::new(OverflowOperation::Sub, Float::one(), Float::from_int(10))),
    );

    // Multiply
    assert_eq!(
        half_max.checked_mul(half_max),
        Err(OverflowError::new(OverflowOperation::Mul, half_max, half_max))
    );
    assert_eq!(
        Float::new(2, i32::MAX / 2 + 1).unwrap().checked_mul(Float::new(5, i32::MAX / 2).unwrap()),
        Err(OverflowError::new(
                OverflowOperation::Mul,
                Float::new(2, i32::MAX / 2 + 1).unwrap(),
                Float::new(5, i32::MAX / 2).unwrap(),
        )),
    );

    // Divide
    assert_eq!(
        Float::one().checked_div(Float::zero()),
        Err(FloatDivisionError::DivideByZero),
    );
    assert_eq!(
        Float::MAX.checked_div(Float::from_float(0.0123)),
        Err(FloatDivisionError::OverflowError),
    );

}

#[test]
fn saturating() {
    // Add
    assert_eq!(Float::MAX.saturating_add(Float::from_int(10)), Float::MAX);

    // Subtract
    assert_eq!(Float::zero().saturating_sub(Float::from_int(10)), Float::zero());

    // Multiply
    assert_eq!(Float::MAX.saturating_mul(Float::from_int(2)), Float::MAX);
    assert_eq!(Float::MIN.saturating_mul(Float::from_float(0.5)), Float::MIN);
    assert_eq!(Float::from_int(2).saturating_mul(Float::MAX), Float::MAX);
    assert_eq!(Float::from_float(0.5).saturating_mul(Float::MIN), Float::MIN);

    // Divide
    assert_eq!(Float::MAX.saturating_div(Float::from_float(0.5)), Float::MAX);
    assert_eq!(Float::MIN.saturating_div(Float::from_int(2)), Float::MIN);
    assert_eq!(Float::from_float(0.5).saturating_div(Float::MAX), Float::MIN);
    assert_eq!(Float::from_int(2).saturating_div(Float::MIN), Float::MAX);
}

#[test]
fn ilog_10() {
    assert_eq!(ilog10(0u128), 0);
    assert_eq!(ilog10(1u128), 0);

    // Test around powers of 10
    for i in 1..39 {
        assert_eq!(ilog10(10u128.pow(i) - 1), i - 1);
        assert_eq!(ilog10(10u128.pow(i)), i);
    }

    // Test around powers of 2
    for i in 1..127 {
        assert_eq!(ilog10(2u128.pow(i) - 1), ((2u128.pow(i) - 1) as f64).log10().floor() as u32);
        assert_eq!(ilog10(2u128.pow(i)), (2u128.pow(i) as f64).log10().floor() as u32);
    }

    assert_eq!(ilog10(u128::MAX), 38);
}


