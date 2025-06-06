use bumpalo::Bump;

use crate::Instruction;

pub fn eval_sequence(programs: &[&[Instruction]], eval_arena: &mut Bump) -> Result<(), String> {
    for program in programs {
        eval_arena.reset();
        let mut eval = crate::ZapEval::new(&eval_arena, program);
        eval.run()
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::opcodes::{ADD, POP, PUSH_BYTES, PUSH_INT};

    use crate::GLOBAL;
    use crate::program::disassemble_bytecode;
    use stats_alloc::Region;

    use super::*;

    use bumpalo::collections::Vec as ArenaVec;

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

        let bytes_arena = Bump::with_capacity(10_000_000);
        let program_arena = Bump::with_capacity(10_000_000);

        let mut program_map = HashMap::<&[u8], &[Instruction]>::new();

        let mut programs = ArenaVec::with_capacity_in(programs_bytecode.len(), &program_arena);

        for bytecode in programs_bytecode {
            if program_map.contains_key(bytecode) {
                programs.push(program_map[bytecode]);
                continue; // Skip if already parsed
            }

            let program = disassemble_bytecode(bytecode, &bytes_arena, &program_arena)
                .map_err(|_| format!("Failed to parse bytecode"))
                .unwrap();

            let program = program_arena.alloc(program);

            program_map.insert(bytecode, program);
            programs.push(program);
        }

        let mut eval_arena = Bump::with_capacity(1_000_000);

        let region = Region::new(&GLOBAL);
        let result = eval_sequence(&programs, &mut eval_arena);
        let alloc_stats = region.change();

        assert!(result.is_ok(), "Expected eval_sequence to succeed");

        assert_eq!(alloc_stats.allocations, 0);
    }
}
