use criterion::{Criterion, black_box, criterion_group, criterion_main};
use zap_vm::{Instruction, opcodes};

fn create_small_bytecode() -> Vec<u8> {
    let mut bytecode = Vec::new();

    // PUSH_INT 42
    bytecode.push(opcodes::PUSH_INT);
    bytecode.extend_from_slice(&42u64.to_be_bytes());

    // ADD
    bytecode.push(opcodes::ADD);

    bytecode
}

fn create_medium_bytecode() -> Vec<u8> {
    let mut bytecode = Vec::new();

    // Multiple instructions to make a medium sized program
    for i in 0..10 {
        // PUSH_INT i
        bytecode.push(opcodes::PUSH_INT);
        bytecode.extend_from_slice(&(i as u64).to_be_bytes());

        // PUSH_BYTES (small)
        bytecode.push(opcodes::PUSH_BYTES);
        bytecode.extend_from_slice(&3u16.to_be_bytes());
        bytecode.extend_from_slice(b"abc");
    }

    // Add some additional operations
    bytecode.push(opcodes::ADD);
    bytecode.push(opcodes::BYTES_LEN);
    bytecode.push(opcodes::INIT_VEC_WITH_INITIAL_CAPACITY);

    bytecode
}

fn create_large_bytecode() -> Vec<u8> {
    let mut bytecode = Vec::new();

    // A more complex program with many instructions
    for i in 0..500 {
        // PUSH_INT i
        bytecode.push(opcodes::PUSH_INT);
        bytecode.extend_from_slice(&(i as u64).to_be_bytes());

        // PUSH_BYTES (variable size)
        let bytes = format!("string_{}", "a".repeat(i)).into_bytes();
        let len = bytes.len() as u16;
        bytecode.push(opcodes::PUSH_BYTES);
        bytecode.extend_from_slice(&len.to_be_bytes());
        bytecode.extend_from_slice(&bytes);

        // Add some operations
        if i % 5 == 0 {
            bytecode.push(opcodes::ADD);
        } else if i % 5 == 1 {
            bytecode.push(opcodes::BYTES_LEN);
        } else if i % 5 == 2 {
            bytecode.push(opcodes::BYTE_ADD);
        } else if i % 5 == 3 {
            bytecode.push(opcodes::SCRATCH_STORE);
            bytecode.push(opcodes::SCRATCH_LOAD);
        } else {
            bytecode.push(opcodes::INIT_VEC_WITH_INITIAL_CAPACITY);
            bytecode.push(opcodes::PUSH_VEC);
        }
    }

    bytecode
}

fn bench_parse_instructions(c: &mut Criterion) {
    let small_bytecode = create_small_bytecode();
    let medium_bytecode = create_medium_bytecode();
    let large_bytecode = create_large_bytecode();

    let mut group = c.benchmark_group("instruction_parsing");

    let bump = bumpalo::Bump::new();

    group.bench_function("small_program", |b| {
        b.iter(|| Instruction::from_bytes(black_box(&small_bytecode), &bump).unwrap());
    });

    group.bench_function("medium_program", |b| {
        b.iter(|| Instruction::from_bytes(black_box(&medium_bytecode), &bump).unwrap());
    });

    group.bench_function("large_program", |b| {
        b.iter(|| Instruction::from_bytes(black_box(&large_bytecode), &bump).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_parse_instructions);
criterion_main!(benches);
