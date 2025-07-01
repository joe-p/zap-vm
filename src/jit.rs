use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::program::Instruction;

pub struct JitCompiler {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    module: JITModule,
}

pub struct CompiledFunction {
    ptr: *const u8,
}

impl CompiledFunction {
    pub fn call(&self) -> i64 {
        let code_fn = unsafe { std::mem::transmute::<_, fn() -> i64>(self.ptr) };
        code_fn()
    }
}

#[derive(Debug)]
pub enum JitError {
    CompilationError(String),
    UnsupportedInstruction,
}

impl JitCompiler {
    pub fn new() -> Result<Self, JitError> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
        })
    }

    pub fn compile_function(
        &mut self,
        instructions: &[Instruction],
    ) -> Result<CompiledFunction, JitError> {
        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(types::I64));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Initialize virtual stack using a simple array approach
        // Create a stack slot for our virtual stack (1000 i64 values)
        let stack_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8000, // 1000 * 8 bytes
            8,    // 8-byte alignment
        ));

        // Get the address of the stack slot
        let stack_memory = builder.ins().stack_addr(types::I64, stack_slot, 0);

        // Variables for stack management
        let stack_pointer = Variable::new(0);
        let stack_base = Variable::new(1);

        // Declare variable types
        builder.declare_var(stack_pointer, types::I64);
        builder.declare_var(stack_base, types::I64);

        // Store stack base and initialize stack pointer
        builder.def_var(stack_base, stack_memory);
        builder.def_var(stack_pointer, stack_memory);

        // Compile each instruction
        for instruction in instructions {
            match instruction {
                Instruction::PushInt(value) => {
                    let current_sp = builder.use_var(stack_pointer);
                    let val = builder.ins().iconst(types::I64, *value as i64);

                    // Store value at current stack pointer
                    builder.ins().store(MemFlags::new(), val, current_sp, 0);

                    // Increment stack pointer
                    let new_sp = builder.ins().iadd_imm(current_sp, 8);
                    builder.def_var(stack_pointer, new_sp);
                }
                Instruction::Add => {
                    // Pop two values and add them
                    let current_sp = builder.use_var(stack_pointer);

                    // Pop second operand (b)
                    let sp_after_b = builder.ins().iadd_imm(current_sp, -8);
                    let b = builder
                        .ins()
                        .load(types::I64, MemFlags::new(), sp_after_b, 0);

                    // Pop first operand (a)
                    let sp_after_a = builder.ins().iadd_imm(sp_after_b, -8);
                    let a = builder
                        .ins()
                        .load(types::I64, MemFlags::new(), sp_after_a, 0);

                    // Compute result
                    let result = builder.ins().iadd(a, b);

                    // Push result back onto stack
                    builder.ins().store(MemFlags::new(), result, sp_after_a, 0);
                    let new_sp = builder.ins().iadd_imm(sp_after_a, 8);
                    builder.def_var(stack_pointer, new_sp);
                }
                Instruction::Sub => {
                    // Pop two values and subtract them
                    let current_sp = builder.use_var(stack_pointer);

                    // Pop second operand (b)
                    let sp_after_b = builder.ins().iadd_imm(current_sp, -8);
                    let b = builder
                        .ins()
                        .load(types::I64, MemFlags::new(), sp_after_b, 0);

                    // Pop first operand (a)
                    let sp_after_a = builder.ins().iadd_imm(sp_after_b, -8);
                    let a = builder
                        .ins()
                        .load(types::I64, MemFlags::new(), sp_after_a, 0);

                    // Compute result (a - b)
                    let result = builder.ins().isub(a, b);

                    // Push result back onto stack
                    builder.ins().store(MemFlags::new(), result, sp_after_a, 0);
                    let new_sp = builder.ins().iadd_imm(sp_after_a, 8);
                    builder.def_var(stack_pointer, new_sp);
                }
                _ => return Err(JitError::UnsupportedInstruction),
            }
        }

        // For now, return the top of stack or 0 if empty
        let current_sp = builder.use_var(stack_pointer);
        let stack_base_val = builder.use_var(stack_base);

        // Check if stack is empty
        let is_empty = builder.ins().icmp(IntCC::Equal, current_sp, stack_base_val);

        let empty_block = builder.create_block();
        let non_empty_block = builder.create_block();
        let return_block = builder.create_block();

        builder
            .ins()
            .brif(is_empty, empty_block, &[], non_empty_block, &[]);

        // Empty stack case - return 0
        builder.switch_to_block(empty_block);
        builder.seal_block(empty_block);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().jump(return_block, &[zero]);

        // Non-empty stack case - return top value
        builder.switch_to_block(non_empty_block);
        builder.seal_block(non_empty_block);
        let top_addr = builder.ins().iadd_imm(current_sp, -8);
        let top_value = builder.ins().load(types::I64, MemFlags::new(), top_addr, 0);
        builder.ins().jump(return_block, &[top_value]);

        // Return block
        builder.switch_to_block(return_block);
        builder.append_block_param(return_block, types::I64);
        builder.seal_block(return_block);
        let return_val = builder.block_params(return_block)[0];
        builder.ins().return_(&[return_val]);

        builder.finalize();

        let id = self
            .module
            .declare_function("jit_function", Linkage::Export, &self.ctx.func.signature)
            .map_err(|e| JitError::CompilationError(e.to_string()))?;

        self.module
            .define_function(id, &mut self.ctx)
            .map_err(|e| JitError::CompilationError(e.to_string()))?;

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();

        let code_ptr = self.module.get_finalized_function(id);

        Ok(CompiledFunction { ptr: code_ptr })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Instruction;

    #[test]
    fn test_jit_push_int() {
        let mut jit = JitCompiler::new().unwrap();
        let instructions = vec![Instruction::PushInt(42)];

        let compiled = jit.compile_function(&instructions).unwrap();
        let result = compiled.call();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_jit_add() {
        let mut jit = JitCompiler::new().unwrap();
        let instructions = vec![
            Instruction::PushInt(10),
            Instruction::PushInt(32),
            Instruction::Add,
        ];

        let compiled = jit.compile_function(&instructions).unwrap();
        let result = compiled.call();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_jit_sub() {
        let mut jit = JitCompiler::new().unwrap();
        let instructions = vec![
            Instruction::PushInt(50),
            Instruction::PushInt(8),
            Instruction::Sub,
        ];

        let compiled = jit.compile_function(&instructions).unwrap();
        let result = compiled.call();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_jit_complex_arithmetic() {
        let mut jit = JitCompiler::new().unwrap();
        let instructions = vec![
            Instruction::PushInt(10),
            Instruction::PushInt(5),
            Instruction::Add, // Stack: [15]
            Instruction::PushInt(3),
            Instruction::Sub, // Stack: [12]
            Instruction::PushInt(30),
            Instruction::Add, // Stack: [42]
        ];

        let compiled = jit.compile_function(&instructions).unwrap();
        let result = compiled.call();

        assert_eq!(result, 42);
    }
}

