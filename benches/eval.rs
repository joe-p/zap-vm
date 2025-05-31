use std::mem::ManuallyDrop;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
extern crate alloc;

use alloc::vec::Vec;
use bumpalo::Bump;
use zap_vm::{ZAP_STACK_CAPACITY, ZapEval};

// Helper function to reduce code duplication across benchmarks
fn run_benchmark<S, M>(c: &mut Criterion, name: &str, setup: S, measure: M)
where
    S: Fn(&mut ZapEval) + Copy,
    M: Fn(&mut ZapEval) + Copy,
{
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
    let mut bump = Bump::with_capacity(16_000);
    let bump_ptr = &mut bump as *mut Bump;

    unsafe {
        let mut eval = ZapEval::new(&mut stack, &*bump_ptr, &mut vecs);
        let eval_ptr = &mut eval as *mut ZapEval;

        c.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let eval = &mut *eval_ptr;
                    eval.stack.clear();
                    bump.reset();
                    setup(eval);
                },
                |_| {
                    measure(&mut *eval_ptr);
                },
                BatchSize::PerIteration,
            );
        });
    }
}

fn benchmark_op_add(c: &mut Criterion) {
    run_benchmark(
        c,
        "op_add",
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
        |eval| {
            // Create a vector with initial capacity
            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(2);
            eval.op_reg_store(); // Store in reg 2

            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(3);
            eval.op_reg_store(); // Store in reg 3
        },
        |eval| {
            // Push 10 items
            for i in 0..11 {
                eval.op_push_int(2);
                eval.op_reg_load();
                eval.op_push_int(black_box(i + 10));
                eval.op_push_vec();

                eval.op_push_int(3);
                eval.op_reg_load();
                eval.op_push_int(black_box(i + 10));
                eval.op_push_vec();
            }
        },
    );
}

fn benchmark_op_byte_add_u512(c: &mut Criterion) {
    let hex128 = "102030405060708090a0b0c0d0e0f000";
    let u512 = format!("{}", hex128.repeat(4));
    let u512_bytes = ManuallyDrop::new(hex::decode(u512).expect("Failed to decode hex string"));

    let u512_bytes_ref = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&u512_bytes) };

    run_benchmark(
        c,
        "op_byte_add_u512",
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
    let u256 = format!("{}", hex128.repeat(4));
    let u256_bytes = ManuallyDrop::new(hex::decode(u256).expect("Failed to decode hex string"));

    let bytes_ref = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&u256_bytes) };

    run_benchmark(
        c,
        "op_byte_add_u256",
        |eval| {
            // Push two bytes onto the stack
            eval.op_push_bytes(black_box(bytes_ref));
            eval.op_push_bytes(black_box(bytes_ref));
        },
        |eval| {
            eval.op_byte_add();
        },
    );
}

fn benchmark_op_byte_sqrt_u512(c: &mut Criterion) {
    let hex128 = "102030405060708090a0b0c0d0e0f000";
    let u512 = format!("{}", hex128.repeat(4));
    let u512_bytes = ManuallyDrop::new(hex::decode(u512).expect("Failed to decode hex string"));

    let u512_bytes_ref = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&u512_bytes) };

    run_benchmark(
        c,
        "op_byte_byte_sqrt_u512",
        |eval| {
            // Push two bytes onto the stack
            eval.op_push_bytes(black_box(u512_bytes_ref));
        },
        |eval| {
            eval.op_byte_sqrt();
        },
    );
}

criterion_group!(
    benches,
    benchmark_op_byte_sqrt_u512,
    benchmark_op_byte_add_u256,
    benchmark_op_byte_add_u512,
    benchmark_alternating_vecs_over_capacity,
    benchmark_op_add,
    benchmark_op_init_vec,
    benchmark_op_push_vec,
    benchmark_multiple_push_ops,
    benchmark_multiple_push_over_capacity
);
criterion_main!(benches);
