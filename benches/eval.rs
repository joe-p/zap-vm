use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ed25519_dalek::{SigningKey, ed25519::signature::SignerMut};
use rand::Rng;
extern crate alloc;

use bumpalo::Bump;
use bumpalo::collections::Vec as ArenaVec;
use zap_vm::{Assembler, Instruction, ZapEval};

// Helper function to reduce code duplication across benchmarks
fn run_benchmark<S, M>(c: &mut Criterion, name: &str, source: &str, setup: S, measure: M)
where
    S: Fn(&mut ZapEval) + Copy,
    M: Fn(&mut ZapEval) + Copy,
{
    let mut bump = Bump::with_capacity(16_000);
    let bump_ptr = &mut bump as *mut Bump;

    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let instructions = assemble_program(source, &bytes_arena, &program_arena);

    c.bench_function(name, |b| {
        b.iter_batched(
            || {
                bump.reset();
                let mut eval = unsafe { ZapEval::new(&*bump_ptr, &instructions) };
                setup(&mut eval);
                eval
            },
            |mut eval| {
                measure(&mut eval);
            },
            BatchSize::PerIteration,
        );
    });
}

fn benchmark_op_add(c: &mut Criterion) {
    run_benchmark(
        c,
        "op_add",
        "",
        |eval| {
            // Push two integers onto the stack
            eval.op_push_int(black_box(5));
            eval.op_push_int(black_box(3));
        },
        |eval| {
            eval.op_add();
        },
    );
}

fn benchmark_op_init_vec(c: &mut Criterion) {
    run_benchmark(
        c,
        "op_init_vec_capacity_10",
        "",
        |eval| {
            eval.op_push_int(black_box(10));
        },
        |eval| {
            eval.op_init_vec_with_initial_capacity();
        },
    );
}

fn benchmark_op_push_vec(c: &mut Criterion) {
    run_benchmark(
        c,
        "op_push_vec_single_item",
        "",
        |eval| {
            // Create a vector first
            eval.op_push_int(black_box(10));
            eval.op_init_vec_with_initial_capacity();

            // Push an item to the vector
            eval.op_push_int(black_box(42));
        },
        |eval| {
            eval.op_push_vec();
        },
    );
}

fn benchmark_multiple_push_ops(c: &mut Criterion) {
    run_benchmark(
        c,
        "push_10_items_to_vec",
        "",
        |eval| {
            // Create a vector
            eval.op_push_int(black_box(10));
            eval.op_init_vec_with_initial_capacity();
        },
        |eval| {
            // Push 10 items
            for i in 0..10 {
                eval.op_push_int(black_box(i));
                eval.op_push_vec();
            }
        },
    );
}

fn benchmark_multiple_push_over_capacity(c: &mut Criterion) {
    run_benchmark(
        c,
        "push_10_items_over_capacity",
        "",
        |eval| {
            // Create a vector
            eval.op_push_int(black_box(1));
            eval.op_init_vec_with_initial_capacity();
        },
        |eval| {
            // Push 10 items
            for i in 0..11 {
                eval.op_push_int(black_box(i));
                eval.op_push_vec();
            }
        },
    );
}

fn benchmark_alternating_vecs_over_capacity(c: &mut Criterion) {
    run_benchmark(
        c,
        "alternating_vecs_over_capacity",
        "",
        |eval| {
            // Create a vector with initial capacity
            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(2);
            eval.op_scratch_store(); // Store in scratch 2

            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(3);
            eval.op_scratch_store(); // Store in scratch 3
        },
        |eval| {
            // Push 10 items
            for i in 0..11 {
                eval.op_push_int(2);
                eval.op_scratch_load();
                eval.op_push_int(black_box(i + 10));
                eval.op_push_vec();

                eval.op_push_int(3);
                eval.op_scratch_load();
                eval.op_push_int(black_box(i + 10));
                eval.op_push_vec();
            }
        },
    );
}

fn benchmark_op_byte_add_u512(c: &mut Criterion) {
    let hex128 = "102030405060708090a0b0c0d0e0f000";
    let u512 = hex128.repeat(4).to_string();
    let u512_bytes = Box::leak(
        hex::decode(u512)
            .expect("Failed to decode hex string")
            .into_boxed_slice(),
    );
    let u512_bytes_ref = Box::leak(Box::new(u512_bytes as &[u8]));

    run_benchmark(
        c,
        "op_byte_add_u512",
        "",
        |eval| {
            // Push two bytes onto the stack
            eval.op_push_bytes(black_box(u512_bytes_ref));
            eval.op_push_bytes(black_box(u512_bytes_ref));
        },
        |eval| {
            eval.op_byte_add();
        },
    );
}

fn benchmark_op_byte_add_u256(c: &mut Criterion) {
    let hex128 = "102030405060708090a0b0c0d0e0f000";
    let u256 = hex128.repeat(4).to_string();
    let u256_bytes = Box::leak(
        hex::decode(u256)
            .expect("Failed to decode hex string")
            .into_boxed_slice(),
    );
    let u256_bytes_ref = Box::leak(Box::new(u256_bytes as &[u8]));

    run_benchmark(
        c,
        "op_byte_add_u256",
        "",
        |eval| {
            // Push two bytes onto the stack
            eval.op_push_bytes(black_box(u256_bytes_ref));
            eval.op_push_bytes(black_box(u256_bytes_ref));
        },
        |eval| {
            eval.op_byte_add();
        },
    );
}

fn benchmark_op_byte_sqrt_u512(c: &mut Criterion) {
    let hex128 = "102030405060708090a0b0c0d0e0f000";
    let u512 = hex128.repeat(4).to_string();
    let u512_bytes = Box::leak(
        hex::decode(u512)
            .expect("Failed to decode hex string")
            .into_boxed_slice(),
    );
    let u512_bytes_ref = Box::leak(Box::new(u512_bytes as &[u8]));

    run_benchmark(
        c,
        "op_byte_byte_sqrt_u512",
        "",
        |eval| {
            // Push two bytes onto the stack
            eval.op_push_bytes(black_box(u512_bytes_ref));
        },
        |eval| {
            eval.op_byte_sqrt();
        },
    );
}

fn benchmark_op_ed25516_verify(c: &mut Criterion) {
    // Create a random number generator
    let mut rng = rand::rng();

    // Generate 32 random bytes
    let random_bytes: [u8; 32] = rng.random();
    let mut signing_key: SigningKey = SigningKey::from(random_bytes);

    let public_key = signing_key.verifying_key();

    let message = b"Hello, world!";
    let signature = signing_key.sign(message);

    let sig_bytes = Box::leak(signature.to_bytes().to_vec().into_boxed_slice());
    let sig_bytes_ref = Box::leak(Box::new(sig_bytes as &[u8]));

    let public_key_bytes = Box::leak(public_key.to_bytes().to_vec().into_boxed_slice());
    let public_key_bytes_ref = Box::leak(Box::new(public_key_bytes as &[u8]));

    let message_bytes = Box::leak(message.to_vec().into_boxed_slice());
    let message_bytes_ref = Box::leak(Box::new(message_bytes as &[u8]));

    run_benchmark(
        c,
        "op_ed25519_verify",
        "",
        |eval| {
            eval.op_push_bytes(black_box(message_bytes_ref));
            eval.op_push_bytes(black_box(public_key_bytes_ref));
            eval.op_push_bytes(black_box(sig_bytes_ref));
        },
        |eval| {
            eval.op_ed25519_verify();
        },
    );
}

fn benchmark_op_pop(c: &mut Criterion) {
    run_benchmark(
        c,
        "op_pop",
        "",
        |eval| {
            // Push an integer onto the stack
            eval.op_push_int(black_box(42));
        },
        |eval| {
            eval.op_pop();
        },
    );
}

fn assemble_program<'bytes_arena, 'program_arena: 'bytes_arena>(
    source: &str,
    bytes_arena: &'bytes_arena Bump,
    program_arena: &'program_arena Bump,
) -> ArenaVec<'program_arena, Instruction<'bytes_arena>> {
    // Assemble the source code
    let mut assembler = Assembler::new(bytes_arena);
    let instructions = assembler
        .assemble(source, program_arena)
        .expect("Failed to assemble program");

    instructions
}

// For context, the same program benchmarked in go-algorand against the AVM is ~18us on an Apple M4
// Pro. To be clear, the AVM does much more computation during eval due to the EvalParams, opcode
// budget, etc. That being said, we want to make sure we are at least as fast as the AVM and as of
// this writing, the Zap VM is about 3x faster.
//
// See https://gist.github.com/joe-p/639b5d4a61d78d0c6a0027ae8278953f
fn benchmark_fibonacci(c: &mut Criterion) {
    let n = 10;
    let source = format!(
        r#"
        int {n}
        call fib_function
        return

    fib_function:
        func_def 1 0 1  // 1 arg, 0 locals, 1 return
        
        // Check if n <= 1 (base cases)
        arg_load 0      // load n
        int 1
        <=              // n <= 1?
        bz recursive_case

        // Base case: return n (0 or 1)
        arg_load 0
        return_func

    recursive_case:
        // Compute fib(n-1)
        arg_load 0      // load n
        int 1
        -               // n - 1
        call fib_function // call fib(n-1)
        
        // Compute fib(n-2) 
        arg_load 0      // load n
        int 2
        -               // n - 2
        call fib_function // call fib(n-2)
        
        // Add fib(n-1) + fib(n-2)
        +               // fib(n-1) + fib(n-2)
        return_func
    "#
    );

    run_benchmark(
        c,
        "fibonacci",
        &source,
        |_| {},
        |eval| {
            eval.run();
        },
    );
}

criterion_group!(
    benches,
    benchmark_op_pop,
    benchmark_op_ed25516_verify,
    benchmark_op_byte_sqrt_u512,
    benchmark_op_byte_add_u256,
    benchmark_op_byte_add_u512,
    benchmark_alternating_vecs_over_capacity,
    benchmark_op_add,
    benchmark_op_init_vec,
    benchmark_op_push_vec,
    benchmark_multiple_push_ops,
    benchmark_multiple_push_over_capacity,
    benchmark_fibonacci,
);
criterion_main!(benches);
