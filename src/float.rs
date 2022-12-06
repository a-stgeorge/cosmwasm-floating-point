use forward_ref::{forward_ref_binop, forward_ref_op_assign};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Rem, RemAssign};
use thiserror::Error;

use cosmwasm_std::{
    CheckedFromRatioError, 
    ConversionOverflowError,
    DivideByZeroError, 
    Decimal,
    Decimal256,
    Isqrt,
    OverflowError, 
    OverflowOperation, 
    StdError,
    Uint128,
    Uint256,
};

#[cfg(test)]
mod tests;

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
    
    // Returns actual exponent based on significand, not stored value
    pub fn exponent(&self) -> i32 {
        self.exponent - Self::DIGITS
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

    pub fn max(val1: Self, val2: Self) -> Self {
        if val1 > val2 {
            val1
        } else {
            val2
        }
    }

    pub fn min(val1: Self, val2: Self) -> Self {
        if val1 < val2 {
            val1
        } else {
            val2
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

    pub fn shift_decimal(&self, amt: i32) -> Result<Self, FloatRangeExceeded> {
        Ok(Self {
            significand: self.significand,
            exponent: self.exponent
                .checked_add(amt)
                .ok_or_else(|| FloatRangeExceeded)?,
        })
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

impl From<Decimal256> for Float {
    fn from(val: Decimal256) -> Self {
        Self::from(val.atomics()) 
            * Self::new(1, -(val.decimal_places() as i32)).unwrap_or_default()
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

impl TryFrom<Float> for Uint128 {
    type Error = ConversionOverflowError;

    fn try_from(value: Float) -> Result<Uint128, ConversionOverflowError> {
        if value > Float::from(Uint128::MAX) {
            return Err(ConversionOverflowError::new("Float", "Uint128", value.to_string()))
        } else if value < Float::one() {
            return Ok(Uint128::zero())
        }

        if value.exponent() > 0 {
            Ok(Uint128::from(value.significand() * pow10(value.exponent() as u32)))
        } else if value.exponent() < 0 {
            Ok(Uint128::from(value.significand() / pow10(-value.exponent() as u32)))
        } else {
            Ok(Uint128::from(value.significand()))
        }
    }
}

impl TryFrom<Float> for Uint256 {
    type Error = ConversionOverflowError;

    fn try_from(value: Float) -> Result<Uint256, ConversionOverflowError> {
        if value > Float::from(Uint256::MAX) {
            return Err(ConversionOverflowError::new("Float", "Uint256", value.to_string()))
        } else if value < Float::one() {
            return Ok(Uint256::zero())
        }

        if value.exponent() > 0 {
            Ok(Uint256::from(value.significand()) 
               * Uint256::from(10u128).pow(value.exponent() as u32))
        } else if value.exponent() < 0 {
            Ok(Uint256::from(value.significand()) 
               / Uint256::from(10u128).pow(-value.exponent() as u32))
        } else {
            Ok(Uint256::from(value.significand()))
        }
    }
}

impl TryFrom<Float> for Decimal {
    type Error = ConversionOverflowError;

    fn try_from(value: Float) -> Result<Decimal, ConversionOverflowError> {
        Ok(Decimal::new(
            Uint128::try_from(value.shift_decimal(18)
                    .map_err(|_| ConversionOverflowError::new("Float", "Decimal", value.to_string()))?)
                .map_err(|_| ConversionOverflowError::new("Float", "Decimal", value.to_string()))?,
        ))
    }
}

impl TryFrom<Float> for Decimal256 {
    type Error = ConversionOverflowError;

    fn try_from(value: Float) -> Result<Decimal256, ConversionOverflowError> {
        Ok(Decimal256::new(
            Uint256::try_from(value.shift_decimal(18)
                    .map_err(|_| ConversionOverflowError::new("Float", "Decimal", value.to_string()))?)
                .map_err(|_| ConversionOverflowError::new("Float", "Decimal", value.to_string()))?,
        ))
    }
}

impl TryFrom<Float> for u128 {
    type Error = ConversionOverflowError;

    fn try_from(value: Float) -> Result<u128, ConversionOverflowError> {
        if value > Float::from(u128::MAX) {
            return Err(ConversionOverflowError::new("Float", "u128", value.to_string()))
        } else if value < Float::one() {
            return Ok(0)
        }

        if value.exponent() > 0 {
            Ok(value.significand() * pow10(value.exponent() as u32))
        } else if value.exponent() < 0 {
            Ok(value.significand() / pow10(-value.exponent() as u32))
        } else {
            Ok(value.significand())
        }
    }
}

impl From<Float> for f64 {
    fn from(value: Float) -> f64 {
        // No overflow for float type (becomes NaN, 0 or inf)
        (value.significand() as f64) * (10f64.powi(value.exponent()))
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

impl From<FloatRangeExceeded> for StdError {
    fn from(_source: FloatRangeExceeded) -> Self {
        Self::generic_err("Float range exceeded")
    }
}

