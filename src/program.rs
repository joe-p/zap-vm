extern crate alloc;

use alloc::vec::Vec;

pub enum Instruction {
    PushInt(u64),
    PushBytes(Vec<u8>),
    BytesLen,
    Add,
    InitVecWithInitialCapacity,
    PushVec,
    RegStore,
    RegLoad,
    ByteAdd,
    ByteSqrt,
    Ed25519Verify,
    Branch(u16),
}

// Opcodes for instructions
pub mod opcodes {
    pub const PUSH_INT: u8 = 0x01;
    pub const PUSH_BYTES: u8 = 0x02;
    pub const BYTES_LEN: u8 = 0x03;
    pub const ADD: u8 = 0x04;
    pub const INIT_VEC_WITH_INITIAL_CAPACITY: u8 = 0x05;
    pub const PUSH_VEC: u8 = 0x06;
    pub const REG_STORE: u8 = 0x07;
    pub const REG_LOAD: u8 = 0x08;
    pub const BYTE_ADD: u8 = 0x09;
    pub const BYTE_SQRT: u8 = 0x0A;
    pub const ED25519_VERIFY: u8 = 0x0B;
    pub const BRANCH: u8 = 0x0C;
}

#[derive(Debug)]
pub enum InstructionParseError {
    InvalidOpcode(u8),
    UnexpectedEndOfBytes,
    InvalidDataLength,
}

impl Instruction {
    /// Converts a byte array to a sequence of instructions
    pub fn from_bytes(bytes: &[u8]) -> Result<Vec<Instruction>, InstructionParseError> {
        let mut instructions = Vec::new();
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

                    let data = bytes[index..index + len].to_vec();
                    instructions.push(Instruction::PushBytes(data));
                    index += len;
                }
                opcodes::BYTES_LEN => {
                    instructions.push(Instruction::BytesLen);
                }
                opcodes::ADD => {
                    instructions.push(Instruction::Add);
                }
                opcodes::INIT_VEC_WITH_INITIAL_CAPACITY => {
                    instructions.push(Instruction::InitVecWithInitialCapacity);
                }
                opcodes::PUSH_VEC => {
                    instructions.push(Instruction::PushVec);
                }
                opcodes::REG_STORE => {
                    instructions.push(Instruction::RegStore);
                }
                opcodes::REG_LOAD => {
                    instructions.push(Instruction::RegLoad);
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
                _ => return Err(InstructionParseError::InvalidOpcode(opcode)),
            }
        }

        Ok(instructions)
    }
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

        let result = Instruction::from_bytes(&bytecode).unwrap();
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
    fn parse_instructions_invalid_opcode() {
        let bytecode = vec![0xFF]; // Invalid opcode

        match Instruction::from_bytes(&bytecode) {
            Err(InstructionParseError::InvalidOpcode(opcode)) => assert_eq!(opcode, 0xFF),
            _ => panic!("Expected InvalidOpcode error"),
        }
    }

    #[test]
    fn parse_instructions_unexpected_end() {
        // PUSH_INT without enough bytes for the value
        let bytecode = vec![PUSH_INT, 0x01, 0x02]; // Missing 6 bytes

        match Instruction::from_bytes(&bytecode) {
            Err(InstructionParseError::UnexpectedEndOfBytes) => {}
            _ => panic!("Expected UnexpectedEndOfBytes error"),
        }
    }
}
