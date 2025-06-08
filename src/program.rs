extern crate alloc;

use bumpalo::Bump;
use bumpalo::collections::Vec as ArenaVec;

#[derive(Debug, Clone, Copy)]
pub enum Instruction<'bytes_arena> {
    PushInt(u64),
    PushBytes(&'bytes_arena [u8]),
    BytesLen,
    Add,
    Sub,
    Mul,
    Div,
    InitVecWithInitialCapacity,
    PushVec,
    ScratchStore,
    ScratchLoad,
    ByteAdd,
    ByteSqrt,
    Ed25519Verify,
    Branch(u16),
    GetElement,
    Pop,
    BranchZero(u16),
    BranchNonZero(u16),
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    Exit,
    Dup,
    // Function call instructions
    Call(u16),
    ReturnFunction,
    DefineFunctionSignature(u16, u16, u16), // (arg_count, local_count, return_count)
    // Frame pointer instructions
    LoadLocal(u16),
    StoreLocal(u16),
    LoadArg(u16),
}

// Opcodes for instructions
pub mod opcodes {
    pub const PUSH_INT: u8 = 0x01;
    pub const PUSH_BYTES: u8 = 0x02;
    pub const BYTES_LEN: u8 = 0x03;
    pub const ADD: u8 = 0x04;
    pub const SUB: u8 = 0x05;
    pub const MUL: u8 = 0x06;
    pub const DIV: u8 = 0x07;
    pub const INIT_VEC_WITH_INITIAL_CAPACITY: u8 = 0x08;
    pub const PUSH_VEC: u8 = 0x09;
    pub const SCRATCH_STORE: u8 = 0x0A;
    pub const SCRATCH_LOAD: u8 = 0x0B;
    pub const BYTE_ADD: u8 = 0x0C;
    pub const BYTE_SQRT: u8 = 0x0D;
    pub const ED25519_VERIFY: u8 = 0x0E;
    pub const BRANCH: u8 = 0x0F;
    pub const GET_ELEMENT: u8 = 0x10;
    pub const POP: u8 = 0x11;
    pub const BRANCH_ZERO: u8 = 0x12;
    pub const BRANCH_NON_ZERO: u8 = 0x13;
    pub const EQUAL: u8 = 0x14;
    pub const NOT_EQUAL: u8 = 0x15;
    pub const LESS_THAN: u8 = 0x16;
    pub const GREATER_THAN: u8 = 0x17;
    pub const LESS_THAN_OR_EQUAL: u8 = 0x18;
    pub const GREATER_THAN_OR_EQUAL: u8 = 0x19;
    pub const EXIT: u8 = 0x1A;
    pub const DUP: u8 = 0x1B;
    // Function call opcodes
    pub const CALL: u8 = 0x1C;
    pub const RETURN_FUNCTION: u8 = 0x1E;
    pub const DEFINE_FUNCTION_SIGNATURE: u8 = 0x22;
    // Frame pointer opcodes
    pub const LOAD_LOCAL: u8 = 0x1F;
    pub const STORE_LOCAL: u8 = 0x20;
    pub const LOAD_ARG: u8 = 0x21;
}

#[derive(Debug)]
pub enum InstructionParseError {
    InvalidOpcode(u8),
    UnexpectedEndOfBytes,
    InvalidDataLength,
}

/// Converts a byte array to a sequence of instructions
/// The bytes arena is used to allocate byte slices
/// The program arena is used to allocate the instructions
/// These are separated because the size of both are unknown and we don't want to continuously
/// re-allocate when we go over capacity.
pub fn disassemble_bytecode<'program_arena, 'bytes_arena: 'program_arena>(
    bytes: &'bytes_arena [u8],
    bytes_arena: &'bytes_arena Bump,
    program_arena: &'program_arena Bump,
) -> Result<ArenaVec<'program_arena, Instruction<'program_arena>>, InstructionParseError> {
    let mut instructions = ArenaVec::new_in(program_arena);
    let mut index = 0;

    while index < bytes.len() {
        let opcode = bytes[index];
        index += 1;

        match opcode {
            opcodes::PUSH_INT => {
                if index + 8 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut value_bytes = [0u8; 8];
                value_bytes.copy_from_slice(&bytes[index..index + 8]);
                let value = u64::from_be_bytes(value_bytes);

                instructions.push(Instruction::PushInt(value));
                index += 8;
            }
            opcodes::PUSH_BYTES => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut len_bytes = [0u8; 2];
                len_bytes.copy_from_slice(&bytes[index..index + 2]);
                let len = u16::from_be_bytes(len_bytes) as usize;
                index += 2;

                if index + len > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let data = bytes_arena.alloc(&bytes[index..index + len]);
                instructions.push(Instruction::PushBytes(data));
                index += len;
            }
            opcodes::BYTES_LEN => {
                instructions.push(Instruction::BytesLen);
            }
            opcodes::ADD => {
                instructions.push(Instruction::Add);
            }
            opcodes::SUB => {
                instructions.push(Instruction::Sub);
            }
            opcodes::MUL => {
                instructions.push(Instruction::Mul);
            }
            opcodes::DIV => {
                instructions.push(Instruction::Div);
            }
            opcodes::INIT_VEC_WITH_INITIAL_CAPACITY => {
                instructions.push(Instruction::InitVecWithInitialCapacity);
            }
            opcodes::PUSH_VEC => {
                instructions.push(Instruction::PushVec);
            }
            opcodes::SCRATCH_STORE => {
                instructions.push(Instruction::ScratchStore);
            }
            opcodes::SCRATCH_LOAD => {
                instructions.push(Instruction::ScratchLoad);
            }
            opcodes::BYTE_ADD => {
                instructions.push(Instruction::ByteAdd);
            }
            opcodes::BYTE_SQRT => {
                instructions.push(Instruction::ByteSqrt);
            }
            opcodes::ED25519_VERIFY => {
                instructions.push(Instruction::Ed25519Verify);
            }
            opcodes::BRANCH => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut branch_bytes = [0u8; 2];
                branch_bytes.copy_from_slice(&bytes[index..index + 2]);
                let branch = u16::from_be_bytes(branch_bytes);

                instructions.push(Instruction::Branch(branch));
                index += 2;
            }
            opcodes::GET_ELEMENT => {
                instructions.push(Instruction::GetElement);
            }
            opcodes::POP => {
                instructions.push(Instruction::Pop);
            }
            opcodes::BRANCH_ZERO => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut branch_bytes = [0u8; 2];
                branch_bytes.copy_from_slice(&bytes[index..index + 2]);
                let branch = u16::from_be_bytes(branch_bytes);

                instructions.push(Instruction::BranchZero(branch));
                index += 2;
            }
            opcodes::BRANCH_NON_ZERO => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut branch_bytes = [0u8; 2];
                branch_bytes.copy_from_slice(&bytes[index..index + 2]);
                let branch = u16::from_be_bytes(branch_bytes);

                instructions.push(Instruction::BranchNonZero(branch));
                index += 2;
            }
            opcodes::EQUAL => {
                instructions.push(Instruction::Equal);
            }
            opcodes::NOT_EQUAL => {
                instructions.push(Instruction::NotEqual);
            }
            opcodes::LESS_THAN => {
                instructions.push(Instruction::LessThan);
            }
            opcodes::GREATER_THAN => {
                instructions.push(Instruction::GreaterThan);
            }
            opcodes::LESS_THAN_OR_EQUAL => {
                instructions.push(Instruction::LessThanOrEqual);
            }
            opcodes::GREATER_THAN_OR_EQUAL => {
                instructions.push(Instruction::GreaterThanOrEqual);
            }
            opcodes::EXIT => {
                instructions.push(Instruction::Exit);
            }
            opcodes::DUP => {
                instructions.push(Instruction::Dup);
            }
            opcodes::CALL => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut target_bytes = [0u8; 2];
                target_bytes.copy_from_slice(&bytes[index..index + 2]);
                let target = u16::from_be_bytes(target_bytes);

                instructions.push(Instruction::Call(target));
                index += 2;
            }
            opcodes::DEFINE_FUNCTION_SIGNATURE => {
                if index + 6 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut arg_count_bytes = [0u8; 2];
                arg_count_bytes.copy_from_slice(&bytes[index..index + 2]);
                let arg_count = u16::from_be_bytes(arg_count_bytes);
                index += 2;

                let mut local_count_bytes = [0u8; 2];
                local_count_bytes.copy_from_slice(&bytes[index..index + 2]);
                let local_count = u16::from_be_bytes(local_count_bytes);
                index += 2;

                let mut return_count_bytes = [0u8; 2];
                return_count_bytes.copy_from_slice(&bytes[index..index + 2]);
                let return_count = u16::from_be_bytes(return_count_bytes);

                instructions.push(Instruction::DefineFunctionSignature(
                    arg_count,
                    local_count,
                    return_count,
                ));
                index += 2;
            }
            opcodes::RETURN_FUNCTION => {
                instructions.push(Instruction::ReturnFunction);
            }
            opcodes::LOAD_LOCAL => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut offset_bytes = [0u8; 2];
                offset_bytes.copy_from_slice(&bytes[index..index + 2]);
                let offset = u16::from_be_bytes(offset_bytes);

                instructions.push(Instruction::LoadLocal(offset));
                index += 2;
            }
            opcodes::STORE_LOCAL => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut offset_bytes = [0u8; 2];
                offset_bytes.copy_from_slice(&bytes[index..index + 2]);
                let offset = u16::from_be_bytes(offset_bytes);

                instructions.push(Instruction::StoreLocal(offset));
                index += 2;
            }
            opcodes::LOAD_ARG => {
                if index + 2 > bytes.len() {
                    return Err(InstructionParseError::UnexpectedEndOfBytes);
                }

                let mut offset_bytes = [0u8; 2];
                offset_bytes.copy_from_slice(&bytes[index..index + 2]);
                let offset = u16::from_be_bytes(offset_bytes);

                instructions.push(Instruction::LoadArg(offset));
                index += 2;
            }
            _ => return Err(InstructionParseError::InvalidOpcode(opcode)),
        }
    }

    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use super::opcodes::*;
    use super::*;

    #[test]
    fn parse_instructions_from_bytes() {
        let mut bytecode = Vec::new();

        // PUSH_INT 42
        bytecode.push(PUSH_INT);
        bytecode.extend_from_slice(&42u64.to_be_bytes());

        // PUSH_BYTES "hello"
        bytecode.push(PUSH_BYTES);
        bytecode.extend_from_slice(&5u16.to_be_bytes());
        bytecode.extend_from_slice(b"hello");

        // ADD
        bytecode.push(ADD);

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 3);

        match &result[0] {
            Instruction::PushInt(value) => assert_eq!(*value, 42),
            _ => panic!("Expected PushInt instruction"),
        }

        match &result[1] {
            Instruction::PushBytes(data) => {
                assert_eq!(data.len(), 5);
                assert_eq!(data, b"hello");
            }
            _ => panic!("Expected PushBytes instruction"),
        }

        match &result[2] {
            Instruction::Add => {}
            _ => panic!("Expected Add instruction"),
        }
    }

    #[test]
    fn parse_arithmetic_instructions() {
        let mut bytecode = Vec::new();

        // SUB
        bytecode.push(SUB);
        // MUL
        bytecode.push(MUL);
        // DIV
        bytecode.push(DIV);

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 3);

        match &result[0] {
            Instruction::Sub => {}
            _ => panic!("Expected Sub instruction"),
        }

        match &result[1] {
            Instruction::Mul => {}
            _ => panic!("Expected Mul instruction"),
        }

        match &result[2] {
            Instruction::Div => {}
            _ => panic!("Expected Div instruction"),
        }
    }

    #[test]
    fn parse_instructions_invalid_opcode() {
        let bytecode = vec![0xFF]; // Invalid opcode

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        match disassemble_bytecode(&bytecode, &bytes_arena, &program_arena) {
            Err(InstructionParseError::InvalidOpcode(opcode)) => assert_eq!(opcode, 0xFF),
            _ => panic!("Expected InvalidOpcode error"),
        }
    }

    #[test]
    fn parse_instructions_unexpected_end() {
        // PUSH_INT without enough bytes for the value
        let bytecode = vec![PUSH_INT, 0x01, 0x02]; // Missing 6 bytes

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        match disassemble_bytecode(&bytecode, &bytes_arena, &program_arena) {
            Err(InstructionParseError::UnexpectedEndOfBytes) => {}
            _ => panic!("Expected UnexpectedEndOfBytes error"),
        }
    }

    #[test]
    fn parse_branch_zero_instruction() {
        let mut bytecode = Vec::new();

        // BRANCH_ZERO with offset 1000
        bytecode.push(BRANCH_ZERO);
        bytecode.extend_from_slice(&1000u16.to_be_bytes());

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            Instruction::BranchZero(offset) => assert_eq!(*offset, 1000),
            _ => panic!("Expected BranchZero instruction"),
        }
    }

    #[test]
    fn parse_branch_non_zero_instruction() {
        let mut bytecode = Vec::new();

        // BRANCH_NON_ZERO with offset 2500
        bytecode.push(BRANCH_NON_ZERO);
        bytecode.extend_from_slice(&2500u16.to_be_bytes());

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            Instruction::BranchNonZero(offset) => assert_eq!(*offset, 2500),
            _ => panic!("Expected BranchNonZero instruction"),
        }
    }

    #[test]
    fn parse_branch_zero_unexpected_end() {
        // BRANCH_ZERO without enough bytes for the offset
        let bytecode = vec![BRANCH_ZERO, 0x01]; // Missing 1 byte

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        match disassemble_bytecode(&bytecode, &bytes_arena, &program_arena) {
            Err(InstructionParseError::UnexpectedEndOfBytes) => {}
            _ => panic!("Expected UnexpectedEndOfBytes error"),
        }
    }

    #[test]
    fn parse_branch_non_zero_unexpected_end() {
        // BRANCH_NON_ZERO without enough bytes for the offset
        let bytecode = vec![BRANCH_NON_ZERO]; // Missing 2 bytes

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        match disassemble_bytecode(&bytecode, &bytes_arena, &program_arena) {
            Err(InstructionParseError::UnexpectedEndOfBytes) => {}
            _ => panic!("Expected UnexpectedEndOfBytes error"),
        }
    }

    #[test]
    fn parse_comparison_instructions() {
        let mut bytecode = Vec::new();

        // Add all comparison instructions
        bytecode.push(EQUAL);
        bytecode.push(NOT_EQUAL);
        bytecode.push(LESS_THAN);
        bytecode.push(GREATER_THAN);
        bytecode.push(LESS_THAN_OR_EQUAL);
        bytecode.push(GREATER_THAN_OR_EQUAL);

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 6);

        match &result[0] {
            Instruction::Equal => {}
            _ => panic!("Expected Equal instruction"),
        }

        match &result[1] {
            Instruction::NotEqual => {}
            _ => panic!("Expected NotEqual instruction"),
        }

        match &result[2] {
            Instruction::LessThan => {}
            _ => panic!("Expected LessThan instruction"),
        }

        match &result[3] {
            Instruction::GreaterThan => {}
            _ => panic!("Expected GreaterThan instruction"),
        }

        match &result[4] {
            Instruction::LessThanOrEqual => {}
            _ => panic!("Expected LessThanOrEqual instruction"),
        }

        match &result[5] {
            Instruction::GreaterThanOrEqual => {}
            _ => panic!("Expected GreaterThanOrEqual instruction"),
        }
    }

    #[test]
    fn parse_return_instruction() {
        let bytecode = vec![EXIT];

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            Instruction::Exit => {}
            _ => panic!("Expected Return instruction"),
        }
    }

    #[test]
    fn parse_dup_instruction() {
        let mut bytecode = Vec::new();

        // DUP
        bytecode.push(DUP);

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 1);

        match &result[0] {
            Instruction::Dup => {}
            _ => panic!("Expected Dup instruction"),
        }
    }

    #[test]
    fn parse_function_call_instructions() {
        let mut bytecode = Vec::new();

        bytecode.push(CALL);
        bytecode.extend_from_slice(&1000u16.to_be_bytes());

        bytecode.push(CALL);
        bytecode.extend_from_slice(&1000u16.to_be_bytes());

        // DEFINE_FUNCTION_SIGNATURE with 2 args, 1 local, 1 return
        bytecode.push(DEFINE_FUNCTION_SIGNATURE);
        bytecode.extend_from_slice(&2u16.to_be_bytes()); // arg_count
        bytecode.extend_from_slice(&1u16.to_be_bytes()); // local_count  
        bytecode.extend_from_slice(&1u16.to_be_bytes()); // return_count

        // RETURN_FUNCTION
        bytecode.push(RETURN_FUNCTION);

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 4);

        match &result[0] {
            Instruction::Call(target) => assert_eq!(*target, 1000),
            _ => panic!("Expected Call instruction"),
        }

        match &result[1] {
            Instruction::Call(target) => assert_eq!(*target, 1000),
            _ => panic!("Expected Call instruction"),
        }

        match &result[2] {
            Instruction::DefineFunctionSignature(arg_count, local_count, return_count) => {
                assert_eq!(*arg_count, 2);
                assert_eq!(*local_count, 1);
                assert_eq!(*return_count, 1);
            }
            _ => panic!("Expected DefineFunctionSignature instruction"),
        }

        match &result[3] {
            Instruction::ReturnFunction => {}
            _ => panic!("Expected ReturnFunction instruction"),
        }
    }

    #[test]
    fn parse_frame_pointer_instructions() {
        let mut bytecode = Vec::new();

        // LOAD_LOCAL with offset 5
        bytecode.push(LOAD_LOCAL);
        bytecode.extend_from_slice(&5u16.to_be_bytes());

        // STORE_LOCAL with offset 10
        bytecode.push(STORE_LOCAL);
        bytecode.extend_from_slice(&10u16.to_be_bytes());

        // LOAD_ARG with offset 2
        bytecode.push(LOAD_ARG);
        bytecode.extend_from_slice(&2u16.to_be_bytes());

        let bytes_arena = Bump::new();
        let program_arena = Bump::new();

        let result = disassemble_bytecode(&bytecode, &bytes_arena, &program_arena).unwrap();
        assert_eq!(result.len(), 3);

        match &result[0] {
            Instruction::LoadLocal(offset) => assert_eq!(*offset, 5),
            _ => panic!("Expected LoadLocal instruction"),
        }

        match &result[1] {
            Instruction::StoreLocal(offset) => assert_eq!(*offset, 10),
            _ => panic!("Expected StoreLocal instruction"),
        }

        match &result[2] {
            Instruction::LoadArg(offset) => assert_eq!(*offset, 2),
            _ => panic!("Expected LoadArg instruction"),
        }
    }
}
