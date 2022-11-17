use forward_ref::{forward_ref_binop, forward_ref_op_assign};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Rem, RemAssign};
use thiserror::Error;

use cosmwasm_std::{
    CheckedFromRatioError, 
    DivideByZeroError, 
    Decimal,
    Isqrt,
    OverflowError, 
    OverflowOperation, 
    Uint128,
    Uint256,
};

// TODO:
//  - verify its impossible to create a float without fix() being called / make sure floats are
//  always correct
//
// Possible future additions
//  - from string method
//  - infinity value

// Panics if overflow
fn pow10(exponent: u32) -> u128 {
    if exponent > 38 {
        panic!("Overflow occured when raising 10 to the power of {}", exponent)
    }
    10u128.pow(exponent)
}

// Temporary method... Waiting for stable ilog10 support
fn ilog10(value: u128) -> u32 {
    if value == 0 {
        return 0 // 0 is always represented with a 0 exponent
    }
    
    let ilog_2 = 128 - value.leading_zeros();

    // Get equivalent log base 10
    let mut temp = ilog_2 * 3;
    if temp > 100 {
        temp += 1;
    } else if temp > 190 {
        temp += 1;
    }
    let approx_ilog10 = temp / 10;

    // Adjust if off by one
    if value < pow10(approx_ilog10) {
        return approx_ilog10 - 1
    } else {
        return approx_ilog10
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Float {
    exponent: i32, // Exponent first so PartialOrd works
    significand: u128,
}

#[derive(Error, Debug, PartialEq, Eq)]
#[error("decimal range exceeded")]
pub struct FloatRangeExceeded;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum FloatDivisionError {
    #[error("Overflow error in float division")]
    OverflowError,
    #[error("Cannot divide by 0")]
    DivideByZero,
}

impl Float {
    const PRECISION: u128 = 1_000_000_000_000_000_000u128; // 10^18, 18 digits of precision
    const MAX_SIGNIFICAND: u128 = 9_999_999_999_999_999_999u128;
    const DIPLAY_DIGITS: i32 = 9; // Number of digits used in displaying before switching to scientific notation

    pub const DIGITS: i32 = 18;
    pub const MAX: Self = Self::no_fix(Self::MAX_SIGNIFICAND, i32::MAX);
    pub const MIN: Self = Self::no_fix(Self::PRECISION, i32::MIN);

    // Could result in an invalid Float, so result is returned
    pub fn new(value: u128, exp: i32) -> Result<Self, FloatRangeExceeded> {
        match value.checked_mul(Self::PRECISION) {
            Some(val) => {
                Self {
                    significand: val, 
                    exponent: exp,
                }.fix()
            },
            None => {
                Self {
                    significand: value, 
                    exponent: exp.checked_add(Self::DIGITS).ok_or_else(|| FloatRangeExceeded)?,
                }.fix()
            },
        }
    }

    pub fn new_uint(value: Uint128, exp: i32) -> Result<Self, FloatRangeExceeded> {
        Self::new(value.u128(), exp)
    }

    // Create Float in scientific notation but without the decimal point, assumes 18 digits
    pub fn decimal(value: u128, exp: i32) -> Result<Self, FloatRangeExceeded> {
        Self {
            significand: value,
            exponent: exp,
        }.fix()
    }

    // Private testing function
    const fn no_fix(value: u128, exp: i32) -> Self {
        Self {
            significand: value,
            exponent: exp,
        }
    }

    pub const fn one() -> Self {
        Self {
            significand: Self::PRECISION,
            exponent: 0,
        }
    }

    pub const fn zero() -> Self {
        Self {
            significand: 0,
            exponent: 0,
        }
    }

    pub fn percent(value: u128) -> Self {
        Self::new(value, -2).unwrap_or_default() // Should never Err, given value is limited to u128
    }

    pub fn permille(value: u128) -> Self {
        Self::new(value, -3).unwrap_or_default() // Should never Err, given value is limited to u128
    }

    pub fn from_int(value: u128) -> Self {
        Self::new(value, 0).unwrap_or_default() // Should never Err, given value is limited to u128
    }

    // Will not produce exact result (due to f64 precision)
    pub fn from_float(value: f64) -> Self {
        if value <= 0.0 {
            return Self::zero() // No negative Floats
        }

        let ilog10 = value.log10().floor() as i32; 
        Self {
            // Will not overflow
            significand: (value * 10f64.powi(-ilog10) * Self::PRECISION as f64) as u128,
            exponent: ilog10,
        }.adjust().unwrap_or_default() // will not overflow in adjust, only cause for fixing will be 
                                       // float roundoff related, fix() not needed
    }

    // Panics on divide by zero
    pub fn from_ratio(numerator: u128, denominator: u128) -> Self {
        if numerator == 0 {
            return Self::zero()
        }

        let num = Self::from_int(numerator);
        let denom = Self::from_int(denominator);
        num / denom
    }

    pub fn checked_from_ratio(
        numerator: u128, 
        denominator: u128
    ) -> Result<Self, CheckedFromRatioError> {
        if numerator == 0 {
            return Ok(Self::zero())
        }
        if denominator == 0 {
            return Err(CheckedFromRatioError::DivideByZero)
        }

        let num = Self::from_int(numerator);
        let denom = Self::from_int(denominator);
        Ok(num.checked_div(denom).map_err(|_| CheckedFromRatioError::Overflow)?)
    }

    pub fn fix(self) -> Result<Self, FloatRangeExceeded> {
        if self.significand == 0 {
            return Ok(Self::zero())
        }

        // Can replace with native log10() when released
        let ilog10 = ilog10(self.significand) as i32;  // 0 - 38
        if ilog10 > Self::DIGITS {
            let delta = ilog10 - Self::DIGITS;
            match self.exponent.checked_add(delta) {
                Some(exp) => {
                    Ok(Self {
                        significand: self.significand / pow10(delta as u32),
                        exponent: exp,
                    })
                },
                None => Err(FloatRangeExceeded),
            }
        } else if ilog10 < Self::DIGITS {
            let delta = Self::DIGITS - ilog10;
            match self.exponent.checked_sub(delta) {
                Some(exp) => {
                    Ok(Self {
                        // Should not overflow: significand < 10^18, delta <= 18
                        significand: self.significand * pow10(delta as u32),
                        exponent: exp,
                    })
                },
                None => Err(FloatRangeExceeded),
            }
        } else {
            Ok(self)
        }
    }

    fn unchecked_fix(self) -> Self {
        if self.significand == 0 {
            return Self::zero()
        }

        // Can replace with naitive log10() when released
        let ilog10 = ilog10(self.significand) as i32;  // 0 - 38
        if ilog10 > Self::DIGITS {
            let delta = ilog10 - Self::DIGITS;
            Self {
                significand: self.significand / pow10(delta as u32),
                exponent: self.exponent + delta,
            }
        } else if ilog10 < Self::DIGITS {
            let delta = Self::DIGITS - ilog10;
            Self {
                // Should not overflow: significand < 10^18, delta <= 18
                significand: self.significand * pow10(delta as u32),
                exponent: self.exponent - delta,
            }
        } else {
            self
        }
    }
    
    // ONLY for when the exponent is at MOST ONE off
    fn adjust(self) -> Result<Self, FloatRangeExceeded> {
        if self.significand == 0 {
            return Ok(Self::zero())
        } 

        if self.significand > Self::MAX_SIGNIFICAND {
            match self.exponent.checked_add(1) {
                Some(exp) => {
                    Ok(Self {
                        significand: self.significand / 10,
                        exponent: exp,
                    })
                },
                None => Err(FloatRangeExceeded)
            }
        } else if self.significand < Self::PRECISION {
            match self.exponent.checked_sub(1) {
                Some(exp) => {
                    Ok(Self {
                        significand: self.significand * 10,
                        exponent: exp, 
                    })
                },
                None => Err(FloatRangeExceeded)
            }
        } else {
            Ok(self)
        }
    }
    
    // ONLY for when the exponent is at MOST ONE off
    fn unchecked_adjust(self) -> Self {
        if self.significand == 0 {
            return Self::zero()
        } 

        if self.significand > Self::MAX_SIGNIFICAND {
            Self {
                significand: self.significand / 10,
                exponent: self.exponent + 1,
            }
        } else if self.significand < Self::PRECISION {
            Self {
                significand: self.significand * 10,
                exponent: self.exponent - 1,
            }
        } else {
            self
        }
    }

    pub fn significand(&self) -> u128 {
        self.significand
    }
    
    pub fn exponent(&self) -> i32 {
        self.exponent
    }

    pub fn is_zero(&self) -> bool {
        self.significand == 0 // exponent doesn't matter if value is zero
    }

    pub fn floor(&self) -> Self {
        if self.exponent >= Self::DIGITS {
            *self
        } else if self.exponent < 0 {
            Self::zero()
        } else {
            let one_factor = pow10(Self::DIGITS.abs_diff(self.exponent)); // between 10 - 10^18
            Self {
                significand: (self.significand / one_factor) * one_factor,
                exponent: self.exponent,
            }.adjust().unwrap_or_default() // exponent < 18, no error
        }
    }

    // Cannot overflow - Float::MAX.ciel() = Float::MAX.ciel()
    pub fn ciel(&self) -> Self {
        if self.exponent >= Self::DIGITS {
            self.clone()
        } else if self.exponent < 0 {
            Self::one()
        } else {
            let one_factor = pow10(Self::DIGITS.abs_diff(self.exponent));
            Self {
                // subtract one to make sure 19.ciel() is still 19
                significand: ((self.significand + one_factor - 1) / one_factor) * one_factor,
                exponent: self.exponent,
            }.adjust().unwrap_or_default() // exponent < 18, no error
        }
    }

    pub fn abs_diff(self, other: Self) -> Self {
        if self > other {
            self - other
        } else {
            other - self
        }
    }

    pub fn inv(self) -> Result<Self, DivideByZeroError> {
        if self.is_zero() {
            Err(DivideByZeroError::new(self))
        } else {
            Ok(Float::one() / self)
        }
    }

    pub fn checked_add(self, other: Self) -> Result<Self, OverflowError> {
        if self.exponent == other.exponent {
            Self {
                significand: self.significand + other.significand,
                exponent: self.exponent,
            }.adjust().map_err(|_| OverflowError::new(OverflowOperation::Add, self, other))
        } else if self.exponent > other.exponent {
            let delta = (self.exponent - other.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return Ok(self) // other is insignificant
            }

            Self {
                significand: self.significand + (other.significand / pow10(delta)),
                exponent: self.exponent,
            }.adjust().map_err(|_| OverflowError::new(OverflowOperation::Add, self, other))
        } else {
            let delta = (other.exponent - self.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return Ok(other) // self is insignificant
            }

            Self {
                significand: (self.significand / pow10(delta)) + other.significand,
                exponent: other.exponent,
            }.adjust().map_err(|_| OverflowError::new(OverflowOperation::Add, self, other))
        }
    }

    pub fn checked_sub(self, other: Self) -> Result<Self, OverflowError> {
        if self.exponent == other.exponent {
            Self {
                significand: self.significand - other.significand,
                exponent: self.exponent,
            }.fix().map_err(|_| OverflowError::new(OverflowOperation::Sub, self, other))
        } else if self.exponent > other.exponent {
            let delta = (self.exponent - other.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return Ok(self) // other is insignificant
            }

            Self {
                significand: self.significand - (other.significand / pow10(delta)),
                exponent: self.exponent,
            }.fix().map_err(|_| OverflowError::new(OverflowOperation::Sub, self, other))
        } else {
            Err(OverflowError::new(OverflowOperation::Sub, self, other))
        }
    }

    pub fn checked_mul(self, other: Self) -> Result<Self, OverflowError> {
        let result = (self.significand * other.significand) / Self::PRECISION; // will not overflow
        Self {
            significand: result,
            exponent: self.exponent.checked_add(other.exponent)
                .ok_or_else(|| OverflowError::new(OverflowOperation::Mul, self, other))?,
        }.adjust().map_err(|_| OverflowError::new(OverflowOperation::Mul, self, other))
    }

    pub fn checked_div(self, other: Self) -> Result<Self, FloatDivisionError> {
        if other.is_zero() {
            return Err(FloatDivisionError::DivideByZero)
        }
    
        match self.exponent.checked_sub(1) {
            Some(exp) => 
                Self {
                    // Multiply by an extra 10 to prevent last digit from rounding. Will not overflow.
                    significand: (self.significand * Self::PRECISION * 10) / other.significand,
                    exponent: exp.checked_sub(other.exponent)
                        .ok_or_else(|| FloatDivisionError::OverflowError)?,
                }.adjust().map_err(|_| FloatDivisionError::OverflowError),
            None =>
                Self {
                    // Don't multiply by extra 10, will cause overflow
                    significand: (self.significand * Self::PRECISION) / other.significand,
                    exponent: self.exponent.checked_sub(other.exponent)
                        .ok_or_else(|| FloatDivisionError::OverflowError)?,
                }.adjust().map_err(|_| FloatDivisionError::OverflowError),
        }
    }

    // Unchecked for speed
    // has some rounding issues with big exponents
    pub fn pow(self, exp: i32) -> Self {
        // This uses the exponentiation by squaring algorithm:
        // https://en.wikipedia.org/wiki/Exponentiation_by_squaring#Basic_method
        if exp == 0 {
            return Self::one();
        }
        if self.is_zero() || self == Self::one() {
            return self
        }

        fn inner(mut x: Float, mut n: i32) -> Float {
            let mut y = Float::one();
            while n > 1 {
                if n % 2 == 0 {
                    x = x * x;
                    n = n / 2;
                } else {
                    y = x * y;
                    x = x * x;
                    n = (n - 1) / 2;
                }
            }
            x * y
        }

        if exp < 0 {
            Self::one() / inner(self, -exp)
        } else {
            inner(self, exp)
        }
    }

    pub fn checked_pow(self, exp: i32) -> Result<Self, OverflowError> {
        if exp == 0 {
            return Ok(Float::one())
        }
        if self.is_zero() || self == Self::one() {
            return Ok(self)
        }

        fn inner(mut x: Float, mut n: i32) -> Result<Float, OverflowError> {
            let mut y = Float::one();
            while n > 1 {
                if n % 2 == 0 {
                    x = x.checked_mul(x)?;
                    n = n / 2;
                } else {
                    y = x.checked_mul(y)?;
                    x = x.checked_mul(x)?;
                    n = (n - 1) / 2;
                }
            }
            x.checked_mul(y)        
        }

        if exp < 0 {
            // x^(-n) = (1/x)^n
            Self::one()
                .checked_div(inner(self, -exp)
                             .map_err(|_| OverflowError::new(OverflowOperation::Pow, self, exp))?)
                .map_err(|_| OverflowError::new(OverflowOperation::Pow, self, exp))
        } else {
            inner(self, exp).map_err(|_| OverflowError::new(OverflowOperation::Pow, self, exp))
        }
    }

    // Should not panic: self.exponent / 2 will never be near i32::MAX or i32::MIN. Thus
    // unchecked_fix()
    pub fn sqrt(&self) -> Self {
        if self.exponent % 2 == 0 {
            Self {
                significand: (self.significand * Self::PRECISION).isqrt(),
                exponent: self.exponent / 2,
            }.unchecked_fix()
        } else {
            Self {
                significand: (self.significand * Self::PRECISION * 10).isqrt(),
                exponent: self.exponent / 2 - (self.exponent < 0) as i32,
            }.unchecked_fix()
        }
    }

    pub fn saturating_add(self, other: Self) -> Self {
        match self.checked_add(other) {
            Ok(val) => val,
            Err(_) => Self::MAX,
        }
    }

    pub fn saturating_sub(self, other: Self) -> Self {
        match self.checked_sub(other) {
            Ok(val) => val,
            Err(_) => Self::zero(),
        }
    }

    pub fn saturating_mul(self, other: Self) -> Self {
        match self.checked_mul(other) {
            Ok(val) => val,
            Err(_) => {
                if self.exponent >= 0 {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }
    }

    pub fn saturating_div(self, other: Self) -> Self {
        match self.checked_div(other) {
            Ok(val) => val,
            Err(div_error) => match div_error {
                FloatDivisionError::DivideByZero => panic!("Cannot divide by zero"),
                FloatDivisionError::OverflowError => {
                    if self.exponent >= 0 {
                        Self::MAX
                    } else {
                        Self::MIN
                    }
                }
            }
        }
    }
}


impl fmt::Display for Float {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let string_value = self.significand.to_string();
        if self.exponent < -Self::DIPLAY_DIGITS || self.exponent > Self::DIPLAY_DIGITS {
            let truncated = string_value.trim_end_matches("0");
            write!(f, "{}.{}e{}", &truncated[..1], &truncated[1..], self.exponent) 
        } else if self.exponent >= 0 {
            let index = self.exponent as usize + 1; // positive
            write!(f, "{}.{}", &string_value[..index], &string_value[index..].trim_end_matches("0"))
        } else {
            let truncated = string_value.trim_end_matches("0");
            let len = truncated.len() as i32;
            if len - self.exponent > Self::DIPLAY_DIGITS + 1 {
                write!(f, "{}.{}e{}", &truncated[..1], &truncated[1..], self.exponent)
            } else {
                write!(f, "0.{:0<2$}{}", "", truncated, self.exponent.abs() as usize - 1)
            }
        }
    }
}

impl From<Uint128> for Float {
    fn from(val: Uint128) -> Self {
        Self::from_int(val.u128())
    }
}

impl From<Uint256> for Float {
    fn from(val: Uint256) -> Self {
        match Uint128::try_from(val) {
            Ok(value) => Self::from(value),
            Err(_) => {
                let mut exp = 0i32;
                let mut adjusted_value = val;
                while adjusted_value > Uint128::MAX.into() {
                    adjusted_value = adjusted_value / Uint256::from(10u128);
                    exp += 1;
                }
                Self::new_uint(Uint128::try_from(adjusted_value).unwrap_or_default(), exp)
                    .unwrap_or_default() // Range of Uint256 will not cause overflow
            },
        }
    }
}

impl From<Decimal> for Float {
    fn from(val: Decimal) -> Self {
        Self::new(val.atomics().u128(), -(val.decimal_places() as i32)).unwrap_or_default() 
            // Should never Err, Decimal's range is limited
    }
}

impl From<u128> for Float {
    fn from(val: u128) -> Self {
        Self::from_int(val)
    }
}

impl From<u64> for Float {
    fn from(val: u64) -> Self {
        Self::from_int(val.into())
    }
}

impl From<u32> for Float {
    fn from(val: u32) -> Self {
        Self::from_int(val.into())
    }
}

impl From<f64> for Float {
    fn from(val: f64) -> Self {
        Self::from_float(val)
    }
}

impl From<f32> for Float {
    fn from(val: f32) -> Self {
        Self::from_float(val.into())
    }
}


impl Add for Float {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        if self.exponent == other.exponent {
            Self {
                significand: self.significand + other.significand,
                exponent: self.exponent,
            }.unchecked_adjust()
        } else if self.exponent > other.exponent {
            let delta = (self.exponent - other.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return self // other is insignificant
            }

            Self {
                significand: self.significand + (other.significand / pow10(delta)),
                exponent: self.exponent,
            }.unchecked_adjust()
        } else {
            let delta = (other.exponent - self.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return other // self is insignificant
            }

            Self {
                significand: (self.significand / pow10(delta)) + other.significand,
                exponent: other.exponent,
            }.unchecked_adjust()
        }
    }
}
forward_ref_binop!(impl Add, add for Float, Float);

impl AddAssign for Float {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
forward_ref_op_assign!(impl AddAssign, add_assign for Float, Float);

impl Sub for Float {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if self.exponent == other.exponent {
            Self {
                significand: self.significand - other.significand,
                exponent: self.exponent,
            }.unchecked_fix()
        } else if self.exponent > other.exponent {
            let delta = (self.exponent - other.exponent) as u32; // garunteed no overflow
            if delta > 18 {
                return self // other is insignificant
            }

            Self {
                significand: self.significand - (other.significand / pow10(delta)),
                exponent: self.exponent,
            }.unchecked_fix()
        } else {
            panic!("attempt to subtract with overflow"); // other is bigger than self
        }
    }
}
forward_ref_binop!(impl Sub, sub for Float, Float);

impl SubAssign for Float {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
forward_ref_op_assign!(impl SubAssign, sub_assign for Float, Float);

impl Mul for Float {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let result = (self.significand * other.significand) / Self::PRECISION;
        Self {
            significand: result,
            exponent: self.exponent + other.exponent,
        }.unchecked_adjust()
    }
}
forward_ref_binop!(impl Mul, mul for Float, Float);

impl MulAssign for Float {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}
forward_ref_op_assign!(impl MulAssign, mul_assign for Float, Float);

impl Div for Float {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match self.exponent.checked_sub(1) {
            Some(exp) =>
                Self {
                    // Multiply by an extra 10 to prevent last digit from rounding.
                    significand: (self.significand * Self::PRECISION * 10) / other.significand,
                    exponent: exp - other.exponent,
                }.unchecked_adjust(), // Round to 0 if overflow
            None =>
                Self {
                    // Don't multiply by extra 10, will cause overflow
                    significand: (self.significand * Self::PRECISION) / other.significand,
                    exponent: self.exponent - other.exponent,
                }.unchecked_adjust(),
        }
    }
}
forward_ref_binop!(impl Div, div for Float, Float);

impl DivAssign for Float {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}
forward_ref_op_assign!(impl DivAssign, div_assign for Float, Float);

impl Rem for Float {
    type Output = Self;

    // Could be optimized?
    // Returs zero if self / other causes overflow or if other is zero
    fn rem(self, other: Self) -> Self {
        match self.checked_div(other) {
            Ok(div) => self - other * div.floor(),
            Err(_) => Self::zero(),
        }
    }
}
forward_ref_binop!(impl Rem, rem for Float, Float);

impl RemAssign for Float {
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}
forward_ref_op_assign!(impl RemAssign, rem_assign for Float, Float);

impl<A> std::iter::Sum<A> for Float
where
    Self: Add<A, Output = Self>
{
    fn sum<I: Iterator<Item = A>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

impl PartialEq<&Float> for Float {
    fn eq(&self, other: &&Float) -> bool {
        self == *other
    }
}

impl PartialEq<Float> for &Float {
    fn eq(&self, other: &Float) -> bool {
        *self == other
    }
}


#[cfg(test)]
mod tests {
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
    }

    #[test]
    #[should_panic]
    fn from_ratio_zero() {
        Float::from_ratio(17, 0);
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

