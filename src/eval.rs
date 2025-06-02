extern crate alloc;

use alloc::vec::Vec;
use bumpalo::collections::Vec as BumpVec;
use crypto_bigint::CheckedAdd;

use crate::{Instruction, ZAP_STACK_CAPACITY, trim_leading_zeros};

#[repr(transparent)] // Guarantees same ABI as inner type
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VecHandle(u32);

/// Represents a value that can be stored on the stack in the ZAP VM.
/// The size of this enum is 24 bytes
#[derive(Clone, Default, Debug, PartialEq)]
pub enum StackValue<'a> {
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Byte array allocated in the bump allocator.
    /// This is a double reference so we can have a thinner pointer in
    /// the enum thus reducing the size of the enum from 24 bytes to 16 bytes.
    Bytes(&'a &'a [u8]),
    /// Vector of StackValues, allocated in the bump allocator.
    Vec(VecHandle),
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
    program: &'a [Instruction],
    program_counter: usize,
}

impl<'a> ZapEval<'a> {
    /// Creates a new ZapEval instance with a mutable reference to a stack and a bump allocator.
    /// Both the stack and the bump allocator are expected to be cleared/reset before use
    pub fn new(
        stack: &'a mut Vec<StackValue<'a>>,
        bump: &'a bumpalo::Bump,
        vecs: &'a mut Vec<BumpVec<'a, StackValue<'a>>>,
        program: &'a [Instruction],
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
            program,
            program_counter: 0,
        }
    }

    pub fn run(&mut self) {
        while self.program_counter < self.program.len() {
            let instruction = &self.program[self.program_counter];

            // Increment BEFORE executing the instruction so branching
            // opcodes can go back to zero without overflowing
            self.program_counter += 1;
            self.execute_instruction(instruction);
        }
    }

    pub fn execute_instruction(&mut self, instruction: &'a Instruction) {
        match instruction {
            Instruction::PushInt(value) => self.op_push_int(*value),
            Instruction::PushBytes(bytes) => self.op_push_bytes(bytes),
            Instruction::BytesLen => self.op_bytes_len(),
            Instruction::Add => self.op_add(),
            Instruction::InitVecWithInitialCapacity => self.op_init_vec_with_initial_capacity(),
            Instruction::PushVec => self.op_push_vec(),
            Instruction::RegStore => self.op_reg_store(),
            Instruction::RegLoad => self.op_reg_load(),
            Instruction::ByteAdd => self.op_byte_add(),
            Instruction::ByteSqrt => self.op_byte_sqrt(),
            Instruction::Ed25519Verify => self.op_ed25519_verify(),
            Instruction::Branch(target) => self.op_branch(*target as usize),
            Instruction::GetElement => self.op_get_element(),
            Instruction::Pop => self.op_pop(),
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

    pub fn op_branch(&mut self, target: usize) {
        if target as usize >= self.program.len() {
            panic!("Branch target out of bounds: {}", target);
        }
        self.program_counter = target
    }

    pub fn op_push_int(&mut self, value: u64) {
        self.push(StackValue::U64(value));
    }

    pub fn op_push_bytes(&mut self, bytes: &'a [u8]) {
        // Store the bytes reference in the bump allocator and then store a reference to that
        let bytes_ref = self.bump.alloc(bytes);
        self.push(StackValue::Bytes(bytes_ref));
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

        let idx = VecHandle {
            0: self.vecs.len() as u32,
        };
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
                self.vecs[v.0 as usize].push(value);
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
                padded_left[64 - left.len()..].copy_from_slice(*left);
                left_num = crypto_bigint::U512::from_be_slice(&padded_left);
            } else {
                left_num = crypto_bigint::U512::from_be_slice(*left);
            }

            if right.len() < 64 {
                let mut padded_right = [0u8; 64];
                padded_right[64 - right.len()..].copy_from_slice(*right);
                right_num = crypto_bigint::U512::from_be_slice(&padded_right);
            } else {
                right_num = crypto_bigint::U512::from_be_slice(*right);
            }

            let result = left_num
                .checked_add(&right_num)
                .expect("Byte addition overflow");

            let result_bytes = self.bump.alloc(result.to_be_bytes());
            let bytes_ref = self.bump.alloc(trim_leading_zeros(result_bytes));
            self.push(StackValue::Bytes(bytes_ref));
        } else {
            panic!("Invalid stack state for byte addition");
        }
    }

    pub fn op_byte_sqrt(&mut self) {
        if let Some(StackValue::Bytes(bytes)) = self.pop() {
            let num: crypto_bigint::U512;
            if bytes.len() < 64 {
                let mut padded_bytes = [0u8; 64];
                padded_bytes[64 - bytes.len()..].copy_from_slice(*bytes);
                num = crypto_bigint::U512::from_be_slice(&padded_bytes);
            } else {
                num = crypto_bigint::U512::from_be_slice(*bytes);
            }

            let result = num.sqrt_vartime();

            let result_bytes = self.bump.alloc(result.to_be_bytes());
            let bytes_ref = self.bump.alloc(trim_leading_zeros(result_bytes));
            self.push(StackValue::Bytes(bytes_ref));
        } else {
            panic!("Expected Bytes on the stack for square root operation");
        }
    }

    pub fn op_ed25519_verify(&mut self) {
        if let (
            Some(StackValue::Bytes(signature)),
            Some(StackValue::Bytes(public_key)),
            Some(StackValue::Bytes(message)),
        ) = (self.pop(), self.pop(), self.pop())
        {
            let public_key = ed25519_dalek::VerifyingKey::try_from(*public_key).unwrap();
            let signature = ed25519_dalek::Signature::try_from(*signature).unwrap();
            let is_valid = public_key.verify_strict(*message, &signature).is_ok();
            self.push(StackValue::U64(if is_valid { 1 } else { 0 }));
        } else {
            panic!("Expected Bytes for signature, message, and public key on the stack");
        }
    }

    pub fn op_get_element(&mut self) {
        if let Some(StackValue::U64(index)) = self.pop() {
            if let Some(StackValue::Vec(vec_handle)) = self.pop() {
                let vec = &self.vecs[vec_handle.0 as usize];
                if index as usize >= vec.len() {
                    panic!("Index out of bounds for Vec access");
                }
                self.push(vec[index as usize].clone());
            } else {
                panic!("Expected a U64 index on the stack");
            }
        } else {
            panic!("Expected a Vec on the stack");
        }
    }

    pub fn op_pop(&mut self) {
        self.stack.pop();
    }
}

#[cfg(test)]
mod tests {
    extern crate stats_alloc;

    use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
    use std::alloc::System;

    use core::mem::ManuallyDrop;

    use super::*;
    use bumpalo::Bump;
    #[global_allocator]
    static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

    fn run_test(
        program: &[Instruction],
        expected_stack: &[StackValue],
        additional_assertions: impl FnOnce(&ZapEval),
    ) {
        let bump = Bump::with_capacity(1_000);
        bump.set_allocation_limit(Some(1_000));

        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));

        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs, program);

        let region = Region::new(&GLOBAL);
        eval.run();
        let alloc_stats = region.change();

        assert_eq!(alloc_stats.allocations, 0);
        assert_eq!(alloc_stats.reallocations, 0);
        assert_eq!(eval.stack, expected_stack);

        additional_assertions(&eval);
    }

    #[test]
    fn stack_value_size() {
        assert_eq!(core::mem::size_of::<StackValue>(), 16);
    }

    #[test]
    fn bytes_len() {
        // Create a program with PushBytes and BytesLen instructions
        let test_bytes = [1, 2, 3, 4].to_vec();
        let program = [Instruction::PushBytes(test_bytes), Instruction::BytesLen];

        let expected_stack = [StackValue::U64(4)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn vec_push() {
        let program = [
            Instruction::PushInt(2),                 // Push initial capacity for Vec
            Instruction::InitVecWithInitialCapacity, // Initialize Vec with capacity 2
            Instruction::PushInt(42),                // Value to add to Vec
            Instruction::PushVec,                    // Push the value onto the Vec
        ];

        run_test(&program, &[StackValue::Vec(VecHandle(0))], |eval| {
            // Check the final state - Vec should be on the stack
            assert_eq!(eval.stack.len(), 1);
            if let StackValue::Vec(vec_idx) = &eval.stack[0] {
                let vec = &eval.vecs[vec_idx.0 as usize];
                assert_eq!(vec.len(), 1);
                if let StackValue::U64(value) = vec[0] {
                    assert_eq!(value, 42);
                } else {
                    panic!("Expected a U64 value in the Vec");
                }
            } else {
                panic!("Expected a Vec on the stack");
            }
        });
    }

    #[test]
    fn add() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::Add,
        ];

        let expected_stack = [StackValue::U64(8)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn reg_store_load() {
        let program = [
            Instruction::PushInt(42), // Push the value to store
            Instruction::PushInt(0),  // Push register index
            Instruction::RegStore,    // Store 42 in register 0
            Instruction::PushInt(0),  // Push register index again
            Instruction::RegLoad,     // Load value from register 0
        ];

        let expected_stack = [StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn byte_add() {
        // Create a program that tests byte addition
        let program = [
            Instruction::PushBytes(vec![2]), // Push first byte
            Instruction::PushBytes(vec![3]), // Push second byte
            Instruction::ByteAdd,            // Add the bytes
        ];

        run_test(&program, &[StackValue::Bytes(&&[5].as_slice())], |eval| {
            // Check the final state - we should have the sum of bytes on the stack
            assert_eq!(eval.stack.len(), 1);
            if let StackValue::Bytes(result) = &eval.stack[0] {
                assert_eq!(**result, [5]);
            } else {
                panic!("Expected a Bytes result on the stack");
            }
        });
    }

    #[test]
    fn test_run_program() {
        // Create a program that:
        // 1. Pushes 10 onto the stack
        // 2. Pushes 20 onto the stack
        // 3. Adds the two numbers
        // 4. Stores the result in register 0
        // 5. Pushes 5 onto the stack
        // 6. Loads the value from register 0
        // 7. Adds the two numbers
        let program = [
            Instruction::PushInt(10),
            Instruction::PushInt(20),
            Instruction::Add,
            Instruction::PushInt(0),
            Instruction::RegStore,
            Instruction::PushInt(5),
            Instruction::PushInt(0),
            Instruction::RegLoad,
            Instruction::Add,
        ];

        let expected_stack = [StackValue::U64(35)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn op_branch() {
        // Create a program with a branch
        let program = [
            Instruction::Branch(2),  // Branch to instruction 2
            Instruction::PushInt(1), // This should be skipped
            Instruction::PushInt(2), // This should be executed
        ];

        let expected_stack = [StackValue::U64(2)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn get_element() {
        // Create a program that:
        // 1. Initializes a Vec with capacity 2
        // 2. Pushes 42 into the Vec
        // 3. Pushes 0 onto the stack to get the first element
        // 4. Gets the element at index 0 from the Vec
        let program = [
            Instruction::PushInt(2),                 // Initial capacity for Vec
            Instruction::InitVecWithInitialCapacity, // Initialize Vec with capacity 2
            Instruction::PushInt(42),                // Value to add to Vec
            Instruction::PushVec,                    // Push the Vec onto the stack
            Instruction::PushInt(0),                 // Index to get from Vec
            Instruction::GetElement,                 // Get element at index 0
        ];

        let expected_stack = [StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn nested_vec_mutation() {
        let program = [
            // Create Vec
            Instruction::PushInt(2), // Push initial capacity for Vec
            Instruction::InitVecWithInitialCapacity, // Initialize Vec with capacity 2
            Instruction::PushInt(11), // Push value to add to Vec
            Instruction::PushVec,    // Push the Vec onto the stack
            // Store vec in register 0
            Instruction::PushInt(0), // Push register index 0
            Instruction::RegStore,   // Store the Vec in register 0
            // Now create a new Vec that will reference the first Vec
            Instruction::PushInt(2), // Push initial capacity for new Vec
            Instruction::InitVecWithInitialCapacity, // Initialize new Vec with capacity 2
            // Push the first Vec onto the stack
            Instruction::PushInt(0), // Push register index 0
            Instruction::RegLoad,    // Load the Vec from register 0
            // Push the first Vec onto the stack again to add another value
            Instruction::PushVec,     // Push the Vec again to add another value
            Instruction::PushInt(22), // Push value to add to new Vec
            Instruction::PushVec,     // Push the new Vec onto the stack
            // Store new Vec in register 1
            Instruction::PushInt(1), // Push register index 1
            Instruction::RegStore,   // Store the new Vec in register 1
            // Load the first vec from register 0
            Instruction::PushInt(0), // Push register index 0
            Instruction::RegLoad,    // Load the Vec from register 0
            // Push new value to add to the first Vec
            Instruction::PushInt(33), // Push value to add to first Vec
            Instruction::PushVec,     // Push the first Vec onto the stack
            // Get second element
            Instruction::PushInt(1), // Push index 1 to get second element
            Instruction::GetElement, // Get element at index 1
            // Now load the vec from register 1
            Instruction::PushInt(1), // Push register index 1
            Instruction::RegLoad,    // Load the Vec from register 1
            // And then get the first element
            Instruction::PushInt(0), // Push index 0 to get first element
            Instruction::GetElement, // Get element at index 0
            // Then get the second element
            Instruction::PushInt(1), // Push index 1 to get second element
            Instruction::GetElement, // Get element at index 1
        ];

        let expected_stack = [StackValue::U64(33), StackValue::U64(33)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn pop() {
        let program = [
            Instruction::PushInt(42), // Push a value onto stack
            Instruction::Pop,         // Pop it off
        ];

        let expected_stack = [];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }
}
