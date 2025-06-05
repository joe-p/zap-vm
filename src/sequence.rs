use std::collections::HashMap;

use bumpalo::Bump;

use crate::Instruction;

pub fn eval_sequence(
    programs_bytecode: &[&[u8]],
    sequence_arena: &mut Bump,
    eval_arena: &mut Bump,
) -> Result<(), String> {
    let mut program_map = HashMap::<&[u8], &[Instruction]>::new();

    for bytecode in programs_bytecode {
        if program_map.contains_key(bytecode) {
            continue; // Skip if already parsed
        }

        let instructions = Instruction::from_bytes(bytecode, &sequence_arena)
            .map_err(|_| format!("Failed to parse bytecode"))?;

        let instructions = sequence_arena.alloc(instructions);

        program_map.insert(bytecode, instructions);
    }

    for instructions in program_map.values() {
        eval_arena.reset();
        let mut eval = crate::ZapEval::new(&eval_arena, instructions);
        eval.run()
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::opcodes::{ADD, POP, PUSH_BYTES, PUSH_INT};

    use crate::GLOBAL;
    use stats_alloc::Region;

    use super::*;

    #[test]
    fn test_eval_sequence() {
        let mut bytecode = Vec::new();

        // PUSH_BYTES "hello"
        bytecode.push(PUSH_BYTES);
        bytecode.extend_from_slice(&5u16.to_be_bytes());
        bytecode.extend_from_slice(b"hello");

        // PUSH 1337
        bytecode.push(PUSH_INT);
        bytecode.extend_from_slice(&1337u64.to_be_bytes());

        // PUSH_INT 42
        bytecode.push(PUSH_INT);
        bytecode.extend_from_slice(&42u64.to_be_bytes());

        // ADD
        bytecode.push(ADD);

        // POP
        bytecode.push(POP);

        let programs_bytecode = vec![
            bytecode.as_slice(),
            bytecode.as_slice(),
            bytecode.as_slice(),
        ];
        let programs_bytecode = programs_bytecode.as_slice();

        let mut eval_arena = Bump::with_capacity(1_000_000);
        let mut sequence_arena = Bump::with_capacity(1_000_000);

        let region = Region::new(&GLOBAL);
        let result = eval_sequence(&programs_bytecode, &mut sequence_arena, &mut eval_arena);
        let alloc_stats = region.change();

        assert!(result.is_ok(), "Expected eval_sequence to succeed");

        // Two allocations:
        // 1. program_map
        // 2. instructions program (which is then cached in the program_map)
        assert_eq!(alloc_stats.allocations, 2);
    }
}
