#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod eval;
pub mod program;

pub use eval::{StackValue, ZapEval};
pub use program::{Instruction, InstructionParseError, opcodes};

pub const ZAP_STACK_CAPACITY: usize = 1000;

fn trim_leading_zeros(bytes: &[u8]) -> &[u8] {
    let first_non_zero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    &bytes[first_non_zero..]
}
