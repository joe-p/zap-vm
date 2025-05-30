#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use bumpalo::collections::Vec as BumpVec;

pub const ZAP_STACK_CAPACITY: usize = 1000;

/// Represents a value that can be stored on the stack in the ZAP.
/// Each value is 8 bytes in size, thus the total size of this enum is ~9 bytes,
/// but will be 16 bytes due to alignment.
#[derive(Clone, Default, Debug)]
pub enum StackValue<'a> {
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Byte array allocated in the bump allocator.
    Bytes(&'a BumpVec<'a, u8>),
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
        if bump.allocated_bytes() != 0 {
            panic!("Bump allocator must be empty before creating ZapEval");
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
}

#[cfg(test)]
mod tests {
    use core::mem::ManuallyDrop;

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

        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let mut bytes = BumpVec::with_capacity_in(4, &bump);
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

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
}
