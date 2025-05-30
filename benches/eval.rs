use std::mem::ManuallyDrop;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
extern crate alloc;

use alloc::vec::Vec;
use bumpalo::Bump;
use zap_vm::{ZAP_STACK_CAPACITY, ZapEval};

fn benchmark_op_add(c: &mut Criterion) {
    let bump = Bump::new();
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));

    let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

    c.bench_function("op_add", |b| {
        b.iter(|| {
            // Push two integers onto the stack
            eval.op_push_int(black_box(5));
            eval.op_push_int(black_box(3));
            // Perform the addition operation
            //           eval.op_add();
            // Clear the stack for the next iteration
            eval.stack.clear();
        });
    });
}

fn benchmark_op_init_vec(c: &mut Criterion) {
    c.bench_function("op_init_vec_capacity_10", |b| {
        let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
        let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
        let bump = Bump::new();

        let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

        b.iter(|| {
            // Initialize a vector with capacity 10
            eval.op_push_int(black_box(10));
            eval.op_init_vec_with_initial_capacity();
            eval.stack.clear()
        });
    });
}

fn benchmark_op_push_vec(c: &mut Criterion) {
    let bump = Bump::new();
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));
    let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

    c.bench_function("op_push_vec_single_item", |b| {
        b.iter(|| {
            // Create a vector first
            eval.op_push_int(black_box(10));
            eval.op_init_vec_with_initial_capacity();

            // Push an item to the vector
            eval.op_push_int(black_box(42));
            eval.op_push_vec();

            eval.stack.clear();
        });
    });
}

fn benchmark_multiple_push_ops(c: &mut Criterion) {
    let bump = Bump::new();
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));

    let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

    c.bench_function("push_10_items_to_vec", |b| {
        b.iter(|| {
            // Create a vector
            eval.op_push_int(black_box(10));
            eval.op_init_vec_with_initial_capacity();

            // Push 10 items
            for i in 0..10 {
                eval.op_push_int(black_box(i));
                eval.op_push_vec();
            }

            eval.stack.clear()
        });
    });
}

fn benchmark_multiple_push_over_capacity(c: &mut Criterion) {
    let bump = Bump::new();
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));

    let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

    c.bench_function("push_10_items_over_capacity", |b| {
        b.iter(|| {
            // Create a vector
            eval.op_push_int(black_box(1));
            eval.op_init_vec_with_initial_capacity();

            // Push 10 items
            for i in 0..11 {
                eval.op_push_int(black_box(i));
                eval.op_push_vec();
            }

            eval.stack.clear()
        });
    });
}

fn benchmark_alternating_vecs_over_capacity(c: &mut Criterion) {
    let bump = Bump::new();
    let mut stack = Vec::with_capacity(ZAP_STACK_CAPACITY);
    let mut vecs = ManuallyDrop::new(Vec::with_capacity(100));

    let mut eval = ZapEval::new(&mut stack, &bump, &mut vecs);

    c.bench_function("alternating_vecs_over_capacity", |b| {
        b.iter(|| {
            // Create a vector with initial capacity
            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(2);
            eval.op_reg_store(); // Store in reg 2

            eval.op_push_int(1);
            eval.op_init_vec_with_initial_capacity();
            eval.op_push_int(3);
            eval.op_reg_store(); // Store in reg 3

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

            // Clear the stack for the next iteration
            eval.stack.clear();
        });
    });
}

criterion_group!(
    benches,
    benchmark_alternating_vecs_over_capacity,
    benchmark_op_add,
    benchmark_op_init_vec,
    benchmark_op_push_vec,
    benchmark_multiple_push_ops,
    benchmark_multiple_push_over_capacity
);
criterion_main!(benches);
