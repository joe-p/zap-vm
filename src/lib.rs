#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use bumpalo::collections::Vec as BumpVec;

pub const ZAP_STACK_CAPACITY: usize = 1000;

/// Represents a value that can be stored on the stack in the ZAP.
/// Each value is 8 bytes in size, thus the total size of this enum is ~9 bytes,
/// but will be 16 bytes due to alignment.
pub enum StackValue<'a> {
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Byte array allocated in the bump allocator.
    Bytes(&'a BumpVec<'a, u8>),
    /// Vector of StackValues, allocated in the bump allocator.
    Vec(&'a mut BumpVec<'a, StackValue<'a>>),
}

pub struct ZapEval<'a> {
    /// The stack used for evaluation, which can hold a variety of StackValue types.
    /// The stack is NOT allocated in the bump allocator but is initialized with a capacity
    /// that will never be exceeded.
    pub stack: &'a mut Vec<StackValue<'a>>,
    bump: &'a bumpalo::Bump,
}

impl<'a> ZapEval<'a> {
    /// Creates a new ZapEval instance with a mutable reference to a stack and a bump allocator.
    /// Both the stack and the bump allocator are expected to be cleared/reset before use
    pub fn new(stack: &'a mut Vec<StackValue<'a>>, bump: &'a bumpalo::Bump) -> Self {
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

        if bump.allocated_bytes() != 0 {
            panic!("Bump allocator must be empty before creating ZapEval");
        }

        ZapEval { stack, bump }
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

    pub fn op_push_bytes(&mut self, bytes: &'a BumpVec<'a, u8>) {
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

        let vec = self
            .bump
            .alloc(BumpVec::with_capacity_in(capacity, self.bump));
        self.push(StackValue::Vec(vec));
    }

    pub fn op_push_vec(&mut self) {
        let value = self.pop().expect("Expected a value to push onto the Vec");
        match self.pop() {
            Some(StackValue::Vec(v)) => {
                v.push(value);
                self.push(StackValue::Vec(v));
            }
            _ => panic!("Expected a Vec on the stack to push onto"),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn stack_value_size() {
        // Ensure that StackValue is 8 bytes in size
        assert_eq!(core::mem::size_of::<StackValue>(), 16);
    }

    #[test]
    fn bytes_len() {
        let bump = Bump::new();
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut eval = ZapEval::new(&mut stack, &bump);
        let mut bytes = BumpVec::with_capacity_in(4, &bump);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        eval.op_push_bytes(&bytes);
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
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut eval = ZapEval::new(&mut stack, &bump);
        eval.op_push_int(2);
        eval.op_init_vec_with_initial_capacity();
        eval.op_push_int(42);
        eval.op_push_vec();

        if let Some(StackValue::Vec(vec)) = eval.pop() {
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
        let mut eval = ZapEval::new(&mut stack, &bump);

        eval.op_push_int(5);
        eval.op_push_int(3);
        eval.op_add();
        if let Some(StackValue::U64(result)) = eval.pop() {
            assert_eq!(result, 8);
        } else {
            panic!("Expected a Uint result on the stack");
        }
    }
}
