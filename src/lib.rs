#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use bumpalo::collections::Vec as BumpVec;
use crypto_bigint::CheckedAdd;

pub const ZAP_STACK_CAPACITY: usize = 1000;

fn trim_leading_zeros(bytes: &[u8]) -> &[u8] {
    let first_non_zero = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
    &bytes[first_non_zero..]
}

/// Represents a value that can be stored on the stack in the ZAP VM.
/// The size of this enum is 24 bytes
#[derive(Clone, Default, Debug)]
pub enum StackValue<'a> {
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Byte array allocated in the bump allocator.
    // NOTE: There is a small performance penalty for using arrays over vecs here. Presumably
    // because the array is larger in size on the stack
    Bytes(&'a [u8]),
    /// Vector of StackValues, allocated in the bump allocator.
    Vec(u64),
    #[default]
    Void,
}

pub struct ZapEval<'a> {
    /// The stack used for evaluation, which can hold a variety of StackValue types.
    /// The stack is NOT allocated in the bump allocator but is initialized with a capacity
    /// that will never be exceeded.
    pub stack: &'a mut Vec<StackValue<'a>>,
    bump: &'a bumpalo::Bump,
    pub vecs: &'a mut Vec<BumpVec<'a, StackValue<'a>>>,
    pub registers: [StackValue<'a>; 256],
}

impl<'a> ZapEval<'a> {
    /// Creates a new ZapEval instance with a mutable reference to a stack and a bump allocator.
    /// Both the stack and the bump allocator are expected to be cleared/reset before use
    pub fn new(
        stack: &'a mut Vec<StackValue<'a>>,
        bump: &'a bumpalo::Bump,
        vecs: &'a mut Vec<BumpVec<'a, StackValue<'a>>>,
    ) -> Self {
        if !stack.is_empty() {
            panic!("Stack must be empty before creating ZapEval");
        }

        if stack.capacity() < ZAP_STACK_CAPACITY {
            panic!(
                "Stack capacity must be at least ZAP_STACK_CAPACITY. Current capacity: {}. ZAP_STACK_CAPACITY: {}",
                stack.capacity(),
                ZAP_STACK_CAPACITY
            );
        }

        #[cfg(not(test))]
        if bump.allocated_bytes() != bump.chunk_capacity() {
            panic!(
                "Bump allocator must be empty before creating ZapEval, but it has {} bytes allocated",
                bump.allocated_bytes() - bump.chunk_capacity()
            );
        }

        let registers = [const { StackValue::Void }; 256];

        ZapEval {
            stack,
            bump,
            vecs,
            registers,
        }
    }

    pub fn push(&mut self, value: StackValue<'a>) {
        if self.stack.len() == self.stack.capacity() {
            panic!("Stack overflow: too many items on the stack");
        }
        self.stack.push(value);
    }

    pub fn pop(&mut self) -> Option<StackValue<'a>> {
        self.stack.pop()
    }

    pub fn op_push_int(&mut self, value: u64) {
        self.push(StackValue::U64(value));
    }

    pub fn op_push_bytes(&mut self, bytes: &'a [u8]) {
        self.push(StackValue::Bytes(bytes));
    }

    pub fn op_bytes_len(&mut self) {
        if let Some(StackValue::Bytes(bytes)) = self.pop() {
            self.push(StackValue::U64(bytes.len() as u64));
        } else {
            panic!("Expected Bytes on the stack for length operation");
        }
    }

    pub fn op_add(&mut self) {
        if let (Some(StackValue::U64(left)), Some(StackValue::U64(right))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(left.checked_add(right).unwrap()));
        } else {
            panic!("Invalid stack state for addition");
        }
    }

    pub fn op_init_vec_with_initial_capacity(&mut self) {
        let capacity = if let Some(StackValue::U64(capacity)) = self.pop() {
            capacity as usize
        } else {
            panic!("Expected a U64 value for Vec capacity");
        };

        let idx = self.vecs.len() as u64;
        let vec = BumpVec::with_capacity_in(capacity, self.bump);
        self.vecs.push(vec);
        self.push(StackValue::Vec(idx));
    }

    /// Push a value onto the Vec at the top of the stack.
    /// /// [vec, value] -> []
    pub fn op_push_vec(&mut self) {
        let value = self.pop().expect("Expected a value to push onto the Vec");
        let vec = self.pop();
        match vec {
            Some(StackValue::Vec(v)) => {
                self.vecs[v as usize].push(value);
                self.push(StackValue::Vec(v));
            }
            _ => panic!(
                "Expected a Vec on the stack to push onto. Got {:?} with value {:?} from stack {:#?}",
                vec, value, self.stack
            ),
        };
    }

    /// Store the value on the top of the stack
    /// [value, reg_idx] -> []
    pub fn op_reg_store(&mut self) {
        match self.pop() {
            Some(StackValue::U64(reg_idx)) if reg_idx < 256 => {
                let value = self
                    .pop()
                    .expect("Expected a value to store in the register");
                self.registers[reg_idx as usize] = value;
            }
            _ => panic!("Expected a U64 register index on the stack"),
        }
    }

    pub fn op_reg_load(&mut self) {
        match self.pop() {
            Some(StackValue::U64(reg_idx)) if reg_idx < 256 => {
                let value = self.registers[reg_idx as usize].clone();
                self.push(value);
            }
            _ => panic!("Expected a U64 register index on the stack"),
        }
    }

    pub fn op_byte_add(&mut self) {
        if let (Some(StackValue::Bytes(left)), Some(StackValue::Bytes(right))) =
            (self.pop(), self.pop())
        {
            let left_num: crypto_bigint::U512;
            let right_num: crypto_bigint::U512;

            if left.len() < 64 {
                let mut padded_left = [0u8; 64];
                padded_left[64 - left.len()..].copy_from_slice(left);
                left_num = crypto_bigint::U512::from_be_slice(&padded_left);
            } else {
                left_num = crypto_bigint::U512::from_be_slice(left);
            }

            if right.len() < 64 {
                let mut padded_right = [0u8; 64];
                padded_right[64 - right.len()..].copy_from_slice(right);
                right_num = crypto_bigint::U512::from_be_slice(&padded_right);
            } else {
                right_num = crypto_bigint::U512::from_be_slice(right);
            }

            let result = left_num
                .checked_add(&right_num)
                .expect("Byte addition overflow");

            let result_bytes = self.bump.alloc(result.to_be_bytes());

            self.push(StackValue::Bytes(trim_leading_zeros(result_bytes)));
        } else {
            panic!("Invalid stack state for byte addition");
        }
    }

    pub fn op_byte_sqrt(&mut self) {
        if let Some(StackValue::Bytes(bytes)) = self.pop() {
            let num: crypto_bigint::U512;
            if bytes.len() < 64 {
                let mut padded_bytes = [0u8; 64];
                padded_bytes[64 - bytes.len()..].copy_from_slice(bytes);
                num = crypto_bigint::U512::from_be_slice(&padded_bytes);
            } else {
                num = crypto_bigint::U512::from_be_slice(bytes);
            }

            let result = num.sqrt();

            let result_bytes = self.bump.alloc(result.to_be_bytes());

            self.push(StackValue::Bytes(trim_leading_zeros(result_bytes)));
        } else {
            panic!("Expected Bytes on the stack for square root operation");
        }
    }
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;

    use super::*;
    use bumpalo::Bump;

    #[test]
    fn stack_value_size() {
        assert_eq!(core::mem::size_of::<StackValue>(), 24);
    }

    #[test]
    fn bytes_len() {
        let bump = Bump::new();

        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

        let bytes = bump.alloc([1, 2, 3, 4]);
        eval.op_push_bytes(bytes);
        eval.op_bytes_len();

        if let Some(StackValue::U64(len)) = eval.pop() {
            assert_eq!(len, 4);
        } else {
            panic!("Expected a Uint result on the stack");
        }
    }

    #[test]
    fn vec_push() {
        let bump = Bump::new();
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);

        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);
        eval.op_push_int(2);
        eval.op_init_vec_with_initial_capacity();
        eval.op_push_int(42);
        eval.op_push_vec();

        if let Some(StackValue::Vec(vec_idx)) = eval.pop() {
            let vec = &eval.vecs[vec_idx as usize];
            assert_eq!(vec.len(), 1);
            if let StackValue::U64(value) = vec[0] {
                assert_eq!(value, 42);
            } else {
                panic!("Expected a U64 value in the Vec");
            }
        } else {
            panic!("Expected a Vec on the stack");
        }
    }

    #[test]
    fn add() {
        let bump = Bump::new();
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

        eval.op_push_int(5);
        eval.op_push_int(3);
        eval.op_add();
        if let Some(StackValue::U64(result)) = eval.pop() {
            assert_eq!(result, 8);
        } else {
            panic!("Expected a Uint result on the stack");
        }
    }

    #[test]
    fn reg_store_load() {
        let bump = Bump::new();
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

        eval.op_push_int(42);
        eval.op_push_int(0);
        eval.op_reg_store(); // Store 42 in register 0
        eval.op_push_int(0); // Push register index 0
        eval.op_reg_load(); // Load value from register 0

        if let Some(StackValue::U64(result)) = eval.pop() {
            assert_eq!(result, 42);
        } else {
            panic!("Expected a Uint result on the stack");
        }
    }

    #[test]
    pub fn byte_add() {
        let bump = Bump::new();
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

        let bytes1 = bump.alloc([2]);
        let bytes2 = bump.alloc([3]);

        eval.op_push_bytes(bytes1);
        eval.op_push_bytes(bytes2);
        eval.op_byte_add();

        if let Some(StackValue::Bytes(result)) = eval.pop() {
            assert_eq!(result, bump.alloc([5]));
        } else {
            panic!("Expected a Bytes result on the stack");
        }
    }
}
