extern crate alloc;

use core::mem;

use bumpalo::collections::Vec as ArenaVec;
use crypto_bigint::CheckedAdd;

use crate::{Instruction, ZAP_STACK_CAPACITY, trim_leading_zeros};

#[repr(transparent)] // Guarantees same ABI as inner type
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VecHandle(u32);

/// Represents a call frame on the call stack
#[derive(Clone, Debug, PartialEq)]
pub struct CallFrame {
    /// The return address (program counter to return to)
    pub return_address: usize,
    /// The previous frame pointer
    pub previous_frame_pointer: usize,
    /// The stack pointer when this frame was created
    pub stack_pointer: usize,
    /// Index on the stack where function arguments start
    pub arguments_start_index: usize,
    /// Number of arguments for this function
    pub argument_count: usize,
    /// Expected number of return values from this function
    pub expected_return_count: usize,
}

/// Represents a value that can be stored on the stack in the ZAP VM.
/// The size of this enum is 24 bytes
#[derive(Clone, Default, Debug, PartialEq)]
pub enum StackValue<'eval_arena> {
    /// Unsigned 64-bit integer value.
    U64(u64),
    /// Byte array allocated in the arena.
    /// This is a double reference so we can have a thinner pointer in
    /// the enum thus reducing the size of the enum from 24 bytes to 16 bytes.
    Bytes(&'eval_arena &'eval_arena [u8]),
    /// Vector of StackValues, allocated in the arena.
    Vec(VecHandle),
    #[default]
    Void,
}

pub struct ZapEval<'eval_arena, 'program_arena: 'eval_arena> {
    /// The stack used for evaluation, which can hold a variety of StackValue types.
    pub stack: &'eval_arena mut ArenaVec<'eval_arena, StackValue<'eval_arena>>,
    /// The arena used for allocating memory for the stack, scratch, StackValue::Bytes and StackValue::Vec
    arena: &'eval_arena bumpalo::Bump,
    /// An arena-allocated vector of arena-allocated vectors.
    pub vecs:
        &'eval_arena mut ArenaVec<'eval_arena, ArenaVec<'eval_arena, StackValue<'eval_arena>>>,
    /// Scratch slots used for storing StackValues accessible throughout the entire program.
    pub scratch_slots: &'eval_arena mut [StackValue<'eval_arena>],
    /// The program being executed, which is a sequence of instructions.
    program: &'program_arena [Instruction<'program_arena>],
    /// The current position in the program being executed. May go backwards with branching instructions.
    program_counter: usize,
    /// Frame pointer pointing to the base of the current stack frame.
    /// This allows access to local variables and parameters via offsets.
    frame_pointer: usize,
    /// Call stack for managing function calls and returns.
    /// Stores return addresses and previous frame pointers.
    call_stack: &'eval_arena mut ArenaVec<'eval_arena, CallFrame>,
    /// Current stack boundary for function calls. Functions cannot pop below this boundary.
    /// This is updated when entering/exiting function calls for performance.
    stack_boundary: usize,
}

impl<'eval_arena, 'program_arena: 'eval_arena> ZapEval<'eval_arena, 'program_arena> {
    /// Creates a new ZapEval instance with a mutable reference to a stack and a bump allocator.
    /// The arena must be reset before each new evaluation to ensure no leftover data from previous evaluations.
    pub fn new(arena: &'eval_arena bumpalo::Bump, program: &'program_arena [Instruction]) -> Self {
        let used_capacity = arena.allocated_bytes() - arena.chunk_capacity();

        // We need to check against 6*word size after resets (instead of 0) until this PR is released:
        // https://github.com/fitzgen/bumpalo/pull/275
        if used_capacity > 6 * mem::size_of::<usize>() {
            panic!("Bump allocator must be reset before creating ZapEval",);
        }

        let scratch_slots = arena.alloc_slice_fill_default(256);

        let stack_vec = ArenaVec::with_capacity_in(ZAP_STACK_CAPACITY, arena);
        let stack = arena.alloc(stack_vec);
        stack.clear();

        let vecs_vec = ArenaVec::with_capacity_in(256, arena);
        let vecs = arena.alloc(vecs_vec);

        let call_stack_vec = ArenaVec::with_capacity_in(256, arena);
        let call_stack = arena.alloc(call_stack_vec);

        ZapEval {
            stack,
            arena,
            vecs,
            scratch_slots,
            program,
            program_counter: 0,
            frame_pointer: 0,
            call_stack,
            stack_boundary: 0,
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

    pub fn execute_instruction(&mut self, instruction: &'program_arena Instruction) {
        match instruction {
            Instruction::PushInt(value) => self.op_push_int(*value),
            Instruction::PushBytes(bytes) => self.op_push_bytes(bytes),
            Instruction::BytesLen => self.op_bytes_len(),
            Instruction::Add => self.op_add(),
            Instruction::Sub => self.op_sub(),
            Instruction::Mul => self.op_mul(),
            Instruction::Div => self.op_div(),
            Instruction::InitVecWithInitialCapacity => self.op_init_vec_with_initial_capacity(),
            Instruction::PushVec => self.op_push_vec(),
            Instruction::ScratchStore => self.op_scratch_store(),
            Instruction::ScratchLoad => self.op_scratch_load(),
            Instruction::ByteAdd => self.op_byte_add(),
            Instruction::ByteSqrt => self.op_byte_sqrt(),
            Instruction::Ed25519Verify => self.op_ed25519_verify(),
            Instruction::Branch(target) => self.op_branch(*target as usize),
            Instruction::GetElement => self.op_get_element(),
            Instruction::Pop => self.op_pop(),
            Instruction::BranchZero(target) => self.op_branch_zero(*target as usize),
            Instruction::BranchNonZero(target) => self.op_branch_non_zero(*target as usize),
            Instruction::Equal => self.op_equal(),
            Instruction::NotEqual => self.op_not_equal(),
            Instruction::LessThan => self.op_less_than(),
            Instruction::GreaterThan => self.op_greater_than(),
            Instruction::LessThanOrEqual => self.op_less_than_or_equal(),
            Instruction::GreaterThanOrEqual => self.op_greater_than_or_equal(),
            Instruction::Exit => self.op_exit(),
            Instruction::Dup => self.op_dup(),
            // Function call instructions
            Instruction::Call(target) => self.op_call(*target as usize),
            Instruction::DefineFunctionSignature(arg_count, local_count, return_count) => self
                .op_define_function_signature(
                    *arg_count as usize,
                    *local_count as usize,
                    *return_count as usize,
                ),
            Instruction::ReturnFunction => self.op_return_function(),
            // Frame pointer instructions
            Instruction::LoadLocal(offset) => self.op_load_local(*offset as usize),
            Instruction::StoreLocal(offset) => self.op_store_local(*offset as usize),
            Instruction::LoadArg(offset) => self.op_load_arg(*offset as usize),
        }
    }

    pub fn push(&mut self, value: StackValue<'eval_arena>) {
        if self.stack.len() == self.stack.capacity() {
            panic!("Stack overflow: too many items on the stack");
        }
        self.stack.push(value);
    }

    /// Pop a value from the stack with boundary checking.
    ///
    /// When inside a function call, this method enforces that the function cannot
    /// pop values below its stack boundary, preventing it from accessing or corrupting
    /// the calling function's stack data.
    pub fn pop(&mut self) -> Option<StackValue<'eval_arena>> {
        // Check stack boundary
        if self.stack.len() <= self.stack_boundary {
            panic!(
                "Stack underflow: function attempted to pop below its stack boundary. \
                 Stack size: {}, boundary: {}",
                self.stack.len(),
                self.stack_boundary
            );
        }

        self.stack.pop()
    }

    pub fn op_branch_zero(&mut self, target: usize) {
        if let Some(StackValue::U64(value)) = self.pop() {
            if value == 0 {
                self.program_counter = target
            }
        } else {
            panic!("Expected a U64 value on the stack for branch zero");
        }
    }

    pub fn op_branch_non_zero(&mut self, target: usize) {
        if let Some(StackValue::U64(value)) = self.pop() {
            if value != 0 {
                self.program_counter = target
            }
        } else {
            panic!("Expected a U64 value on the stack for branch non-zero");
        }
    }

    pub fn op_branch(&mut self, target: usize) {
        if target >= self.program.len() {
            panic!("Branch target out of bounds: {}", target);
        }
        self.program_counter = target
    }

    pub fn op_push_int(&mut self, value: u64) {
        self.push(StackValue::U64(value));
    }

    pub fn op_push_bytes(&mut self, bytes: &'program_arena &'program_arena [u8]) {
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

    pub fn op_sub(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(left.checked_sub(right).unwrap()));
        } else {
            panic!("Invalid stack state for subtraction");
        }
    }

    pub fn op_mul(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(left.checked_mul(right).unwrap()));
        } else {
            panic!("Invalid stack state for multiplication");
        }
    }

    pub fn op_div(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(left.checked_div(right).unwrap()));
        } else {
            panic!("Invalid stack state for division");
        }
    }

    pub fn op_init_vec_with_initial_capacity(&mut self) {
        let capacity = if let Some(StackValue::U64(capacity)) = self.pop() {
            capacity as usize
        } else {
            panic!("Expected a U64 value for Vec capacity");
        };

        let idx = VecHandle(self.vecs.len() as u32);
        let vec = ArenaVec::with_capacity_in(capacity, self.arena);
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
    /// [value, scratch_slot] -> []
    pub fn op_scratch_store(&mut self) {
        match self.pop() {
            Some(StackValue::U64(scratch_slot)) if scratch_slot < 256 => {
                let value = self.pop().expect("Expected a value to store in the slot");
                self.scratch_slots[scratch_slot as usize] = value;
            }
            _ => panic!("Expected a U64 scratch slot index on the stack"),
        }
    }

    pub fn op_scratch_load(&mut self) {
        match self.pop() {
            Some(StackValue::U64(scratch_idx)) if scratch_idx < 256 => {
                let value = self.scratch_slots[scratch_idx as usize].clone();
                self.push(value);
            }
            _ => panic!("Expected a U64 scratch slot index on the stack"),
        }
    }

    pub fn op_byte_add(&mut self) {
        if let (Some(StackValue::Bytes(left)), Some(StackValue::Bytes(right))) =
            (self.pop(), self.pop())
        {
            let left_num = if left.len() < 64 {
                let mut padded_left = [0u8; 64];
                padded_left[64 - left.len()..].copy_from_slice(left);
                crypto_bigint::U512::from_be_slice(&padded_left)
            } else {
                crypto_bigint::U512::from_be_slice(left)
            };

            let right_num = if right.len() < 64 {
                let mut padded_right = [0u8; 64];
                padded_right[64 - right.len()..].copy_from_slice(right);
                crypto_bigint::U512::from_be_slice(&padded_right)
            } else {
                crypto_bigint::U512::from_be_slice(right)
            };

            let result = left_num
                .checked_add(&right_num)
                .expect("Byte addition overflow");

            let result_bytes = self.arena.alloc(result.to_be_bytes());
            let bytes_ref = self.arena.alloc(trim_leading_zeros(result_bytes));
            self.push(StackValue::Bytes(bytes_ref));
        } else {
            panic!("Invalid stack state for byte addition");
        }
    }

    pub fn op_byte_sqrt(&mut self) {
        if let Some(StackValue::Bytes(bytes)) = self.pop() {
            let num = if bytes.len() < 64 {
                let mut padded_bytes = [0u8; 64];
                padded_bytes[64 - bytes.len()..].copy_from_slice(bytes);
                crypto_bigint::U512::from_be_slice(&padded_bytes)
            } else {
                crypto_bigint::U512::from_be_slice(bytes)
            };

            let result = num.sqrt_vartime();

            let result_bytes = self.arena.alloc(result.to_be_bytes());
            let bytes_ref = self.arena.alloc(trim_leading_zeros(result_bytes));
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
            let signature = ed25519_dalek::Signature::try_from(*signature).unwrap_or_else(|_| {
                panic!(
                    "Invalid signature format: expected 64 bytes, got {} bytes",
                    signature.len()
                )
            });
            let is_valid = public_key.verify_strict(message, &signature).is_ok();
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
        self.pop();
    }

    pub fn op_equal(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left == right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for equality comparison");
        }
    }

    pub fn op_not_equal(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left != right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for not-equal comparison");
        }
    }

    pub fn op_less_than(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left < right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for less-than comparison");
        }
    }

    pub fn op_greater_than(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left > right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for greater-than comparison");
        }
    }

    pub fn op_less_than_or_equal(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left <= right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for less-than-or-equal comparison");
        }
    }

    pub fn op_greater_than_or_equal(&mut self) {
        if let (Some(StackValue::U64(right)), Some(StackValue::U64(left))) =
            (self.pop(), self.pop())
        {
            self.push(StackValue::U64(if left >= right { 1 } else { 0 }));
        } else {
            panic!("Invalid stack state for greater-than-or-equal comparison");
        }
    }

    pub fn op_exit(&mut self) {
        // Set program counter to end of program to terminate execution
        self.program_counter = self.program.len();
    }

    pub fn op_dup(&mut self) {
        if let Some(value) = self.pop() {
            self.push(value.clone());
            self.push(value);
        } else {
            panic!("Expected a value to duplicate on the stack");
        }
    }

    // Function call and frame pointer operations

    /// Call function at the specified address.
    /// Arguments are consumed when the function executes DefineArgCount.
    pub fn op_call(&mut self, target: usize) {
        if target >= self.program.len() {
            panic!("Call target out of bounds: {}", target);
        }

        // Create a new call frame without arguments (they'll be consumed by DefineFunctionSignature)
        let frame = CallFrame {
            return_address: self.program_counter,
            previous_frame_pointer: self.frame_pointer,
            stack_pointer: self.stack.len(), // This will be updated by DefineFunctionSignature after consuming args
            arguments_start_index: 0,        // Will be set by DefineFunctionSignature
            argument_count: 0,               // Will be set by DefineFunctionSignature
            expected_return_count: 0,        // Will be set by DefineFunctionSignature
        };

        // Push the call frame onto the call stack
        self.call_stack.push(frame);

        // Set up new frame pointer (current stack position)
        self.frame_pointer = self.stack.len();

        // Jump to the target address
        self.program_counter = target;
    }

    /// Define the complete function signature: arguments, local variables, and return values.
    /// This records argument positions on the stack, allocates local variable space, and sets up return value expectations.
    pub fn op_define_function_signature(
        &mut self,
        arg_count: usize,
        local_count: usize,
        return_count: usize,
    ) {
        if self.call_stack.is_empty() {
            panic!("Cannot define function signature: not inside a function call");
        }

        // Check that we have enough arguments on the stack
        if self.stack.len() < arg_count {
            panic!("Not enough arguments on stack for function call");
        }

        // Calculate where arguments start on the stack (they're at the top, so subtract arg_count)
        let arguments_start_index = self.stack.len() - arg_count;

        // Update the current call frame with argument information and expected return count
        let current_frame_index = self.call_stack.len() - 1;
        self.call_stack[current_frame_index].arguments_start_index = arguments_start_index;
        self.call_stack[current_frame_index].argument_count = arg_count;
        self.call_stack[current_frame_index].expected_return_count = return_count;
        // Stack boundary is set to prevent popping below the arguments
        self.call_stack[current_frame_index].stack_pointer = arguments_start_index;

        // Update the stack boundary for performance (avoids call_stack.last() lookup)
        self.stack_boundary = arguments_start_index;

        // Pre-allocate space for local variables on the stack (initialize to Void)
        for _ in 0..local_count {
            self.push(StackValue::Void);
        }
    }

    /// Return from the current function, restoring the previous frame.
    pub fn op_return_function(&mut self) {
        if let Some(frame) = self.call_stack.pop() {
            // Extract return values from the stack (should be on top)
            let mut return_values =
                ArenaVec::with_capacity_in(frame.expected_return_count, self.arena);
            for _ in 0..frame.expected_return_count {
                if let Some(value) = self.pop() {
                    return_values.push(value);
                } else {
                    panic!("Not enough return values on stack");
                }
            }
            // Return values are in reverse order, so reverse them
            return_values.reverse();

            // Restore the stack to the state before the function call
            self.stack.truncate(frame.stack_pointer);

            // Push the return values back onto the stack
            for value in return_values {
                self.push(value);
            }

            // Restore the program counter to the return address
            self.program_counter = frame.return_address;

            // Restore the previous frame pointer
            self.frame_pointer = frame.previous_frame_pointer;

            // Restore the stack boundary to the previous function's boundary
            self.stack_boundary = if let Some(previous_frame) = self.call_stack.last() {
                previous_frame.stack_pointer
            } else {
                0 // No more function calls, reset to global boundary
            };
        } else {
            panic!("Cannot return from function: call stack is empty");
        }
    }

    /// Load a local variable onto the stack.
    /// Local variables are stored at frame_pointer + offset.
    pub fn op_load_local(&mut self, offset: usize) {
        let index = self.frame_pointer + offset;
        if index >= self.stack.len() {
            panic!(
                "Local variable access out of bounds: frame_pointer={}, offset={}, stack_len={}",
                self.frame_pointer,
                offset,
                self.stack.len()
            );
        }
        let value = self.stack[index].clone();
        self.push(value);
    }

    /// Store a value from the stack into a local variable.
    /// [value] -> []
    pub fn op_store_local(&mut self, offset: usize) {
        let value = self
            .pop()
            .expect("Expected a value to store in local variable");
        let index = self.frame_pointer + offset;

        // Extend the stack if necessary to accommodate the local variable
        while self.stack.len() <= index {
            self.stack.push(StackValue::Void);
        }

        self.stack[index] = value;
    }

    /// Load a function argument onto the stack.
    /// Arguments are stored in the current call frame.
    pub fn op_load_arg(&mut self, offset: usize) {
        if let Some(frame) = self.call_stack.last() {
            if offset >= frame.argument_count {
                panic!(
                    "Argument access out of bounds: offset={}, arg_count={}",
                    offset, frame.argument_count
                );
            }
            let value = self.stack[frame.arguments_start_index + offset].clone();
            self.push(value);
        } else {
            panic!("Cannot load argument: no function call frame available");
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate stats_alloc;

    use crate::GLOBAL;
    use stats_alloc::Region;

    use super::*;
    use bumpalo::Bump;

    fn run_test(
        program: &[Instruction],
        expected_stack: &[StackValue],
        additional_assertions: impl FnOnce(&ZapEval),
    ) {
        let bump = Bump::with_capacity(1_000);
        bump.set_allocation_limit(Some(1_000));

        let mut eval = ZapEval::new(&bump, program);

        let region = Region::new(GLOBAL);
        eval.run();
        let alloc_stats = region.change();

        assert_eq!(alloc_stats.allocations, 0);
        assert_eq!(alloc_stats.reallocations, 0);
        assert_eq!(eval.stack.as_slice(), expected_stack);

        additional_assertions(&eval);
    }

    #[test]
    fn stack_value_size() {
        assert_eq!(core::mem::size_of::<StackValue>(), 16);
    }

    #[test]
    fn bytes_len() {
        // Create a program with PushBytes and BytesLen instructions
        let test_bytes = [1, 2, 3, 4].as_slice();
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
    fn sub() {
        let program = [
            Instruction::PushInt(10),
            Instruction::PushInt(3),
            Instruction::Sub,
        ];

        let expected_stack = [StackValue::U64(7)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn mul() {
        let program = [
            Instruction::PushInt(4),
            Instruction::PushInt(5),
            Instruction::Mul,
        ];

        let expected_stack = [StackValue::U64(20)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn div() {
        let program = [
            Instruction::PushInt(15),
            Instruction::PushInt(3),
            Instruction::Div,
        ];

        let expected_stack = [StackValue::U64(5)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn scratch_store_load() {
        let program = [
            Instruction::PushInt(42),  // Push the value to store
            Instruction::PushInt(0),   // Push scratch index
            Instruction::ScratchStore, // Store 42 in scratch slot 0
            Instruction::PushInt(0),   // Push scratch slot index again
            Instruction::ScratchLoad,  // Load value from scratch slot 0
        ];

        let expected_stack = [StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn byte_add() {
        // Create a program that tests byte addition

        let b1 = [2].as_slice();
        let b2 = [3].as_slice();
        let program = [
            Instruction::PushBytes(b1), // Push first byte
            Instruction::PushBytes(b2), // Push second byte
            Instruction::ByteAdd,       // Add the bytes
        ];

        run_test(&program, &[StackValue::Bytes(&[5].as_slice())], |eval| {
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
        // 4. Stores the result in scratch slot 0
        // 5. Pushes 5 onto the stack
        // 6. Loads the value from scratch slot 0
        // 7. Adds the two numbers
        let program = [
            Instruction::PushInt(10),
            Instruction::PushInt(20),
            Instruction::Add,
            Instruction::PushInt(0),
            Instruction::ScratchStore,
            Instruction::PushInt(5),
            Instruction::PushInt(0),
            Instruction::ScratchLoad,
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
            // Store vec in scratch slot 0
            Instruction::PushInt(0),   // Push scratch slot index 0
            Instruction::ScratchStore, // Store the Vec in scratch slot 0
            // Now create a new Vec that will reference the first Vec
            Instruction::PushInt(2), // Push initial capacity for new Vec
            Instruction::InitVecWithInitialCapacity, // Initialize new Vec with capacity 2
            // Push the first Vec onto the stack
            Instruction::PushInt(0),  // Push scratch slot index 0
            Instruction::ScratchLoad, // Load the Vec from scratch slot 0
            // Push the first Vec onto the stack again to add another value
            Instruction::PushVec,     // Push the Vec again to add another value
            Instruction::PushInt(22), // Push value to add to new Vec
            Instruction::PushVec,     // Push the new Vec onto the stack
            // Store new Vec in scratch slot 1
            Instruction::PushInt(1),   // Push scratch slot index 1
            Instruction::ScratchStore, // Store the new Vec in scratch slot 1
            // Load the first vec from scratch slot 0
            Instruction::PushInt(0),  // Push scratch slot index 0
            Instruction::ScratchLoad, // Load the Vec from scratch slot 0
            // Push new value to add to the first Vec
            Instruction::PushInt(33), // Push value to add to first Vec
            Instruction::PushVec,     // Push the first Vec onto the stack
            // Get second element
            Instruction::PushInt(1), // Push index 1 to get second element
            Instruction::GetElement, // Get element at index 1
            // Now load the vec from scratch slot 1
            Instruction::PushInt(1),  // Push scratch slot index 1
            Instruction::ScratchLoad, // Load the Vec from scratch slot 1
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

    #[test]
    pub fn op_branch_zero_takes_branch() {
        // Test that BranchZero branches when stack value is 0
        let program = [
            Instruction::PushInt(0),    // Push 0 onto stack
            Instruction::BranchZero(4), // Branch to instruction 4 if top of stack is 0
            Instruction::PushInt(1),    // This should be skipped
            Instruction::PushInt(2),    // This should be skipped
            Instruction::PushInt(42),   // This should be executed (target of branch)
        ];

        let expected_stack = [StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn op_branch_zero_no_branch() {
        // Test that BranchZero doesn't branch when stack value is non-zero
        let program = [
            Instruction::PushInt(5),    // Push non-zero value onto stack
            Instruction::BranchZero(4), // Should not branch since value is non-zero
            Instruction::PushInt(1),    // This should be executed
            Instruction::PushInt(2),    // This should be executed
            Instruction::PushInt(42),   // This should also be executed
        ];

        let expected_stack = [StackValue::U64(1), StackValue::U64(2), StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn op_branch_non_zero_takes_branch() {
        // Test that BranchNonZero branches when stack value is non-zero
        let program = [
            Instruction::PushInt(7),       // Push non-zero value onto stack
            Instruction::BranchNonZero(4), // Branch to instruction 4 since value is non-zero
            Instruction::PushInt(1),       // This should be skipped
            Instruction::PushInt(2),       // This should be skipped
            Instruction::PushInt(99),      // This should be executed (target of branch)
        ];

        let expected_stack = [StackValue::U64(99)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    pub fn op_branch_non_zero_no_branch() {
        // Test that BranchNonZero doesn't branch when stack value is 0
        let program = [
            Instruction::PushInt(0),       // Push 0 onto stack
            Instruction::BranchNonZero(4), // Should not branch since value is 0
            Instruction::PushInt(10),      // This should be executed
            Instruction::PushInt(20),      // This should be executed
            Instruction::PushInt(99),      // This should also be executed
        ];

        let expected_stack = [
            StackValue::U64(10),
            StackValue::U64(20),
            StackValue::U64(99),
        ];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn equal_true() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(5),
            Instruction::Equal,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn equal_false() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::Equal,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn not_equal_true() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::NotEqual,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn not_equal_false() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(5),
            Instruction::NotEqual,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn less_than_true() {
        let program = [
            Instruction::PushInt(3),
            Instruction::PushInt(5),
            Instruction::LessThan,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn less_than_false() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::LessThan,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn greater_than_true() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::GreaterThan,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn greater_than_false() {
        let program = [
            Instruction::PushInt(3),
            Instruction::PushInt(5),
            Instruction::GreaterThan,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn less_than_or_equal_true_less() {
        let program = [
            Instruction::PushInt(3),
            Instruction::PushInt(5),
            Instruction::LessThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn less_than_or_equal_true_equal() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(5),
            Instruction::LessThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn less_than_or_equal_false() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::LessThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn greater_than_or_equal_true_greater() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(3),
            Instruction::GreaterThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn greater_than_or_equal_true_equal() {
        let program = [
            Instruction::PushInt(5),
            Instruction::PushInt(5),
            Instruction::GreaterThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(1)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn greater_than_or_equal_false() {
        let program = [
            Instruction::PushInt(3),
            Instruction::PushInt(5),
            Instruction::GreaterThanOrEqual,
        ];

        let expected_stack = [StackValue::U64(0)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn return_early_termination() {
        // Test that Return terminates program execution early
        let program = [
            Instruction::PushInt(42), // This should be executed
            Instruction::Exit,        // This should terminate the program
            Instruction::PushInt(99), // This should NOT be executed
        ];

        let expected_stack = [StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn dup() {
        // Test that Dup duplicates the top value on the stack
        let program = [
            Instruction::PushInt(42), // Push 42 onto the stack
            Instruction::Dup,         // Duplicate the top value
        ];

        let expected_stack = [StackValue::U64(42), StackValue::U64(42)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn dup_bytes() {
        // Test that Dup works with bytes
        let test_bytes = [1, 2, 3, 4].as_slice();
        let program = [
            Instruction::PushBytes(test_bytes), // Push bytes onto the stack
            Instruction::Dup,                   // Duplicate the top value
        ];

        let expected_stack = [
            StackValue::Bytes(&test_bytes),
            StackValue::Bytes(&test_bytes),
        ];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn dup_with_arithmetic() {
        // Test that Dup works correctly with arithmetic operations
        let program = [
            Instruction::PushInt(5), // Push 5 onto the stack
            Instruction::Dup,        // Duplicate 5, so stack is [5, 5]
            Instruction::Add,        // Add the two 5s, so stack is [10]
            Instruction::Dup,        // Duplicate 10, so stack is [10, 10]
            Instruction::Mul,        // Multiply the two 10s, so stack is [100]
        ];

        let expected_stack = [StackValue::U64(100)];

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn function_call_and_return() {
        // Test basic function call and return
        let program = [
            Instruction::PushInt(42), // 0: arg1: Push 42 as argument
            Instruction::PushInt(10), // 1: arg2: Push 10 as argument
            Instruction::Call(4),     // 2: Call function at address 4
            Instruction::Exit,        // 3: This should not be reached due to early return
            Instruction::DefineFunctionSignature(2, 0, 1), // 4: Function: 2 args, 0 locals, 1 return
            Instruction::LoadArg(0),                       // 5: Load first argument
            Instruction::LoadArg(1),                       // 6: Load second argument
            Instruction::Add,                              // 7: Add arguments
            Instruction::ReturnFunction,                   // 8: Return from function
        ];

        let expected_stack = [StackValue::U64(52)]; // 42 + 10 = 52

        run_test(&program, &expected_stack, |eval| {
            // Check that we returned to the main function
            assert_eq!(eval.call_stack.len(), 0);
            assert_eq!(eval.frame_pointer, 0);
        });
    }

    #[test]
    fn local_variables() {
        // Test storing and loading local variables
        let program = [
            Instruction::PushInt(100),  // Push value to store
            Instruction::StoreLocal(0), // Store in local variable 0
            Instruction::PushInt(200),  // Push another value
            Instruction::StoreLocal(1), // Store in local variable 1
            Instruction::LoadLocal(0),  // Load local variable 0
            Instruction::LoadLocal(1),  // Load local variable 1
            Instruction::Add,           // Add them
        ];

        let expected_stack = [
            StackValue::U64(100),
            StackValue::U64(200),
            StackValue::U64(300),
        ]; // Local vars remain, result is 100 + 200 = 300

        run_test(&program, &expected_stack, |_eval| {
            // No additional assertions needed
        });
    }

    #[test]
    fn function_with_arguments() {
        // Test function that uses arguments
        let program = [
            Instruction::PushInt(15),                      // arg1: Push 15 as argument
            Instruction::PushInt(25),                      // arg2: Push 25 as argument
            Instruction::Call(4),                          // Call function at address 4
            Instruction::Exit,                             // This should not be reached
            Instruction::DefineFunctionSignature(2, 0, 1), // Function: 2 args, 0 locals, 1 return
            Instruction::LoadArg(0),                       // Load arg0 (15)
            Instruction::LoadArg(1),                       // Load arg1 (25)
            Instruction::Mul,                              // Multiply arguments
            Instruction::ReturnFunction,                   // Return from function
        ];

        let expected_stack = [StackValue::U64(375)]; // 15 * 25 = 375

        run_test(&program, &expected_stack, |eval| {
            // Check that we returned to the main function
            assert_eq!(eval.call_stack.len(), 0);
            assert_eq!(eval.frame_pointer, 0);
        });
    }

    #[test]
    fn nested_function_calls() {
        // Test nested function calls
        let program = [
            Instruction::PushInt(5),                       // 0: arg: Push 5 as argument
            Instruction::Call(3),                          // 1: Call function A at address 3
            Instruction::Exit,                             // 2: End of main
            Instruction::DefineFunctionSignature(1, 0, 1), // 3: Function A: 1 arg, 0 locals, 1 return
            Instruction::Call(8), // 4: Function A: Call function B at address 8
            Instruction::LoadArg(0), // 5: Function A: Load argument and push to stack
            Instruction::Add,     // 6: Function A: Add result from B to argument
            Instruction::ReturnFunction, // 7: Function A: Return
            Instruction::DefineFunctionSignature(0, 0, 1), // 8: Function B: 0 args, 0 locals, 1 return
            Instruction::PushInt(10),                      // 9: Function B: Push 10
            Instruction::ReturnFunction,                   // 10: Function B: Return
        ];

        let expected_stack = [StackValue::U64(15)]; // Arguments consumed, result is 5 + 10 = 15

        run_test(&program, &expected_stack, |eval| {
            // Check that all functions have returned
            assert_eq!(eval.call_stack.len(), 0);
            assert_eq!(eval.frame_pointer, 0);
        });
    }

    #[test]
    fn function_with_local_variables() {
        // Test function that uses both arguments and local variables
        let program = [
            Instruction::PushInt(5),                       // 0: arg: Push 5 as argument
            Instruction::Call(3),                          // 1: Call function at address 3
            Instruction::Exit,                             // 2: End of main
            Instruction::DefineFunctionSignature(1, 2, 1), // 3: Function: 1 arg, 2 locals, 1 return
            Instruction::PushInt(10),                      // 4: Push 10
            Instruction::StoreLocal(0),                    // 5: Store 10 in local variable 0
            Instruction::PushInt(20),                      // 6: Push 20
            Instruction::StoreLocal(1),                    // 7: Store 20 in local variable 1
            Instruction::LoadArg(0),                       // 8: Load argument (5)
            Instruction::LoadLocal(0),                     // 9: Load local variable 0 (10)
            Instruction::LoadLocal(1),                     // 10: Load local variable 1 (20)
            Instruction::Add,                              // 11: Add locals: 10 + 20 = 30
            Instruction::Add,                              // 12: Add with argument: 5 + 30 = 35
            Instruction::ReturnFunction,                   // 13: Return from function
        ];

        let expected_stack = [StackValue::U64(35)]; // 5 + 10 + 20 = 35

        run_test(&program, &expected_stack, |eval| {
            // Check that we returned to the main function
            assert_eq!(eval.call_stack.len(), 0);
            assert_eq!(eval.frame_pointer, 0);
        });
    }

    #[test]
    #[should_panic(
        expected = "Stack underflow: function attempted to pop below its stack boundary"
    )]
    fn stack_boundary_enforcement() {
        // Test that functions cannot pop below their stack boundary
        let program = [
            Instruction::PushInt(100), // 0: Push value for main function
            Instruction::PushInt(200), // 1: Push another value for main function
            Instruction::Call(4),      // 2: Call function at address 4
            Instruction::Exit,         // 3: End of main
            Instruction::DefineFunctionSignature(0, 1, 0), // 4: Function: 0 args, 1 local, 0 returns
            Instruction::Pop, // 5: Pop the local variable (this should work)
            Instruction::Pop, // 6: This should panic - trying to pop caller's values
            Instruction::ReturnFunction, // 7: This won't be reached due to panic
        ];

        let bump = Bump::with_capacity(1_000);
        let mut eval = ZapEval::new(&bump, &program);

        // Execute manually to be more precise about where the panic should happen
        eval.execute_instruction(&program[0]); // PushInt(100)
        eval.execute_instruction(&program[1]); // PushInt(200)
        eval.execute_instruction(&program[2]); // Call(4)
        eval.execute_instruction(&program[4]); // DefineFunctionSignature(0, 1, 0)
        eval.execute_instruction(&program[5]); // Pop - should work
        eval.execute_instruction(&program[6]); // Pop - should panic
    }
}
