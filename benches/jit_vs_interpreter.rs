use bumpalo::Bump;
use bumpalo::collections::Vec as ArenaVec;
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use zap_vm::jit::JitCompiler;
use zap_vm::{Assembler, Instruction, ZapEval};

fn create_simple_arithmetic_program<'bytes_arena, 'program_arena: 'bytes_arena>(
    bytes_arena: &'bytes_arena Bump,
    program_arena: &'program_arena Bump,
) -> ArenaVec<'program_arena, Instruction<'bytes_arena>> {
    let source = r#"
        int 10
        int 32
        +
        int 8
        -
    "#;

    let mut assembler = Assembler::new(bytes_arena);
    assembler
        .assemble(source, program_arena)
        .expect("Failed to assemble simple arithmetic program")
}

fn create_complex_arithmetic_program<'bytes_arena, 'program_arena: 'bytes_arena>(
    bytes_arena: &'bytes_arena Bump,
    program_arena: &'program_arena Bump,
) -> ArenaVec<'program_arena, Instruction<'bytes_arena>> {
    let source = r#"
        int 5
        int 3
        +
        int 2
        +
        int 4
        -
        int 10
        +
        int 3
        -
    "#;

    let mut assembler = Assembler::new(bytes_arena);
    assembler
        .assemble(source, program_arena)
        .expect("Failed to assemble complex arithmetic program")
}

fn create_fibonacci_program<'bytes_arena, 'program_arena: 'bytes_arena>(
    bytes_arena: &'bytes_arena Bump,
    program_arena: &'program_arena Bump,
    n: u64,
) -> ArenaVec<'program_arena, Instruction<'bytes_arena>> {
    let source = format!(
        r#"
        int {n}
        call fib_function
        return

    fib_function:
        func_def 1 0 1
        
        arg_load 0
        int 1
        <=
        bz recursive_case

        arg_load 0
        return_func

    recursive_case:
        arg_load 0
        int 1
        -
        call fib_function
        
        arg_load 0
        int 2
        -
        call fib_function
        
        +
        return_func
    "#
    );

    let mut assembler = Assembler::new(bytes_arena);
    assembler
        .assemble(&source, program_arena)
        .expect("Failed to assemble fibonacci program")
}

fn benchmark_simple_arithmetic_interpreter(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_simple_arithmetic_program(&bytes_arena, &program_arena);

    let mut bump = Bump::with_capacity(16_000);
    let bump_ptr = &mut bump as *mut Bump;

    c.bench_function("simple_arithmetic_interpreter", |b| {
        b.iter_batched(
            || {
                bump.reset();
                unsafe { ZapEval::new(&*bump_ptr, &instructions) }
            },
            |mut eval| {
                eval.run();
                black_box(eval.stack.last().cloned())
            },
            BatchSize::PerIteration,
        );
    });
}

fn benchmark_simple_arithmetic_jit(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_simple_arithmetic_program(&bytes_arena, &program_arena);

    // Convert to Vec for JIT compiler
    let instruction_vec: Vec<Instruction> = instructions.iter().cloned().collect();

    c.bench_function("simple_arithmetic_jit", |b| {
        b.iter_batched(
            || {
                let mut jit = JitCompiler::new().expect("Failed to create JIT compiler");
                jit.compile_function(&instruction_vec)
                    .expect("Failed to compile function")
            },
            |compiled_fn| black_box(compiled_fn.call()),
            BatchSize::PerIteration,
        );
    });
}

fn benchmark_complex_arithmetic_interpreter(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_complex_arithmetic_program(&bytes_arena, &program_arena);

    let mut bump = Bump::with_capacity(16_000);
    let bump_ptr = &mut bump as *mut Bump;

    c.bench_function("complex_arithmetic_interpreter", |b| {
        b.iter_batched(
            || {
                bump.reset();
                unsafe { ZapEval::new(&*bump_ptr, &instructions) }
            },
            |mut eval| {
                eval.run();
                black_box(eval.stack.last().cloned())
            },
            BatchSize::PerIteration,
        );
    });
}

fn benchmark_complex_arithmetic_jit(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_complex_arithmetic_program(&bytes_arena, &program_arena);

    // Convert to Vec for JIT compiler
    let instruction_vec: Vec<Instruction> = instructions.iter().cloned().collect();

    c.bench_function("complex_arithmetic_jit", |b| {
        b.iter_batched(
            || {
                let mut jit = JitCompiler::new().expect("Failed to create JIT compiler");
                jit.compile_function(&instruction_vec)
                    .expect("Failed to compile function")
            },
            |compiled_fn| black_box(compiled_fn.call()),
            BatchSize::PerIteration,
        );
    });
}

fn benchmark_fibonacci_interpreter(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_fibonacci_program(&bytes_arena, &program_arena, 8);

    let mut bump = Bump::with_capacity(16_000);
    let bump_ptr = &mut bump as *mut Bump;

    c.bench_function("fibonacci_8_interpreter", |b| {
        b.iter_batched(
            || {
                bump.reset();
                unsafe { ZapEval::new(&*bump_ptr, &instructions) }
            },
            |mut eval| {
                eval.run();
                black_box(eval.stack.last().cloned())
            },
            BatchSize::PerIteration,
        );
    });
}

// Note: JIT fibonacci benchmark is commented out because the current JIT implementation
// doesn't support function calls, branches, and other complex instructions needed for fibonacci
// fn benchmark_fibonacci_jit(c: &mut Criterion) {
//     let bytes_arena = Bump::new();
//     let program_arena = Bump::new();
//     let instructions = create_fibonacci_program(&bytes_arena, &program_arena, 8);
//
//     // Convert to Vec for JIT compiler
//     let instruction_vec: Vec<Instruction> = instructions.iter().cloned().collect();
//
//     c.bench_function("fibonacci_8_jit", |b| {
//         b.iter_batched(
//             || {
//                 let mut jit = JitCompiler::new().expect("Failed to create JIT compiler");
//                 jit.compile_function(&instruction_vec).expect("Failed to compile function")
//             },
//             |compiled_fn| {
//                 black_box(compiled_fn.call())
//             },
//             BatchSize::PerIteration,
//         );
//     });
// }

fn benchmark_compilation_overhead(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_complex_arithmetic_program(&bytes_arena, &program_arena);

    // Convert to Vec for JIT compiler
    let instruction_vec: Vec<Instruction> = instructions.iter().cloned().collect();

    c.bench_function("jit_compilation_overhead", |b| {
        b.iter(|| {
            let mut jit = JitCompiler::new().expect("Failed to create JIT compiler");
            black_box(
                jit.compile_function(&instruction_vec)
                    .expect("Failed to compile function"),
            )
        });
    });
}

fn benchmark_jit_execution_only(c: &mut Criterion) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = create_complex_arithmetic_program(&bytes_arena, &program_arena);

    // Convert to Vec for JIT compiler
    let instruction_vec: Vec<Instruction> = instructions.iter().cloned().collect();

    // Pre-compile the function
    let mut jit = JitCompiler::new().expect("Failed to create JIT compiler");
    let compiled_fn = jit
        .compile_function(&instruction_vec)
        .expect("Failed to compile function");

    c.bench_function("jit_execution_only", |b| {
        b.iter(|| black_box(compiled_fn.call()));
    });
}

criterion_group!(
    benches,
    benchmark_simple_arithmetic_interpreter,
    benchmark_simple_arithmetic_jit,
    benchmark_complex_arithmetic_interpreter,
    benchmark_complex_arithmetic_jit,
    benchmark_fibonacci_interpreter,
    benchmark_compilation_overhead,
    benchmark_jit_execution_only,
);
criterion_main!(benches);

