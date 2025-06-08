extern crate alloc;

use alloc::collections::BTreeMap as HashMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use bumpalo::Bump;
use bumpalo::collections::Vec as ArenaVec;

use crate::program::Instruction;

#[derive(Debug, Clone, PartialEq)]
pub enum AssemblerError {
    UnknownInstruction(String),
    InvalidArgument(String),
    InvalidHexValue(String),
    MissingArgument(String),
    TooManyArguments(String),
    DuplicateLabel(String),
    UnknownLabel(String),
    InvalidNumber(String),
}

#[derive(Debug, Clone)]
pub struct ParsedInstruction {
    pub instruction: String,
    pub args: Vec<String>,
    pub line_number: usize,
}

#[derive(Debug, Clone)]
enum ParsedLine {
    Instruction(ParsedInstruction),
    Label(String),
    Empty,
}

pub struct Assembler<'bytes_arena> {
    bytes_arena: &'bytes_arena Bump,
    labels: HashMap<String, usize>,
}

impl<'bytes_arena> Assembler<'bytes_arena> {
    pub fn new(bytes_arena: &'bytes_arena Bump) -> Self {
        Self {
            bytes_arena,
            labels: HashMap::new(),
        }
    }

    /// Assemble assembly source code into Instructions
    pub fn assemble<'program_arena>(
        &mut self,
        source: &str,
        program_arena: &'program_arena Bump,
    ) -> Result<ArenaVec<'program_arena, Instruction<'bytes_arena>>, AssemblerError> {
        // First pass: parse all lines and collect labels
        let parsed_lines = self.parse_lines(source)?;
        self.collect_labels(&parsed_lines)?;

        // Second pass: convert instructions
        let mut instructions = ArenaVec::new_in(program_arena);

        for line in parsed_lines {
            if let ParsedLine::Instruction(parsed_inst) = line {
                let instruction = self.parse_instruction(parsed_inst)?;
                instructions.push(instruction);
            }
        }

        Ok(instructions)
    }

    fn parse_lines(&self, source: &str) -> Result<Vec<ParsedLine>, AssemblerError> {
        let mut parsed_lines = Vec::new();

        for (line_number, line) in source.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("//") {
                parsed_lines.push(ParsedLine::Empty);
                continue;
            }

            // Remove inline comments
            let line = if let Some(comment_pos) = line.find("//") {
                line[..comment_pos].trim()
            } else {
                line
            };

            // Skip if line becomes empty after removing comments
            if line.is_empty() {
                parsed_lines.push(ParsedLine::Empty);
                continue;
            }

            // Check for labels (lines ending with ':')
            if line.ends_with(':') {
                let label = line[..line.len() - 1].trim().to_string();
                if label.is_empty() {
                    return Err(AssemblerError::InvalidArgument("Empty label".to_string()));
                }
                parsed_lines.push(ParsedLine::Label(label));
                continue;
            }

            // Parse instruction
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                parsed_lines.push(ParsedLine::Empty);
                continue;
            }

            let instruction = parts[0].to_lowercase();
            let args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();

            parsed_lines.push(ParsedLine::Instruction(ParsedInstruction {
                instruction,
                args,
                line_number,
            }));
        }

        Ok(parsed_lines)
    }

    fn collect_labels(&mut self, parsed_lines: &[ParsedLine]) -> Result<(), AssemblerError> {
        let mut instruction_index = 0;

        for line in parsed_lines {
            match line {
                ParsedLine::Label(label) => {
                    if self.labels.contains_key(label) {
                        return Err(AssemblerError::DuplicateLabel(label.clone()));
                    }
                    self.labels.insert(label.clone(), instruction_index);
                }
                ParsedLine::Instruction(_) => {
                    instruction_index += 1;
                }
                ParsedLine::Empty => {}
            }
        }

        Ok(())
    }

    fn parse_instruction(
        &self,
        parsed_inst: ParsedInstruction,
    ) -> Result<Instruction<'bytes_arena>, AssemblerError> {
        let instruction = &parsed_inst.instruction;
        let args = &parsed_inst.args;

        match instruction.as_str() {
            // Basic stack operations
            "int" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "int requires 1 argument".to_string(),
                    ));
                }
                let value = self.parse_u64(&args[0])?;
                Ok(Instruction::PushInt(value))
            }

            "bytes" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "bytes requires 1 argument".to_string(),
                    ));
                }
                let bytes = self.parse_bytes(&args[0])?;
                Ok(Instruction::PushBytes(bytes))
            }

            "blen" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "blen takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::BytesLen)
            }

            // Arithmetic operations
            "+" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "+ takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Add)
            }

            "-" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "- takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Sub)
            }

            "*" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "* takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Mul)
            }

            "/" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "/ takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Div)
            }

            // Vector operations
            "vec_init" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "vec_init takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::InitVecWithInitialCapacity)
            }

            "vec_push" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "vec_push takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::PushVec)
            }

            "vec_get" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "vec_get takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::GetElement)
            }

            // Scratch operations
            "store" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "store takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::ScratchStore)
            }

            "load" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "load takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::ScratchLoad)
            }

            // Byte operations
            "b+" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "b+ takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::ByteAdd)
            }

            "bsqrt" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "bsqrt takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::ByteSqrt)
            }

            // Cryptographic operations
            "ed25519_verify" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "ed25519_verify takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Ed25519Verify)
            }

            // Control flow
            "b" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "b requires 1 argument".to_string(),
                    ));
                }
                let target = self.parse_branch_target(&args[0])?;
                Ok(Instruction::Branch(target))
            }

            "bz" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "bz requires 1 argument".to_string(),
                    ));
                }
                let target = self.parse_branch_target(&args[0])?;
                Ok(Instruction::BranchZero(target))
            }

            "bnz" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "bnz requires 1 argument".to_string(),
                    ));
                }
                let target = self.parse_branch_target(&args[0])?;
                Ok(Instruction::BranchNonZero(target))
            }

            // Comparison operations
            "==" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "= takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Equal)
            }

            "!=" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "!= takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::NotEqual)
            }

            "<" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "< takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::LessThan)
            }

            ">" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "> takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::GreaterThan)
            }

            "<=" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "<= takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::LessThanOrEqual)
            }

            ">=" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        ">= takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::GreaterThanOrEqual)
            }

            // Stack manipulation
            "pop" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "pop takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Pop)
            }

            "dup" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "dup takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Dup)
            }

            "return" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "return takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::Exit)
            }

            // Function operations
            "call" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "call requires 1 argument".to_string(),
                    ));
                }
                let target = self.parse_branch_target(&args[0])?;
                Ok(Instruction::Call(target))
            }

            "return_func" => {
                if !args.is_empty() {
                    return Err(AssemblerError::TooManyArguments(
                        "return_func takes no arguments".to_string(),
                    ));
                }
                Ok(Instruction::ReturnFunction)
            }

            "func_def" => {
                if args.len() != 3 {
                    return Err(AssemblerError::MissingArgument(
                        "func_def requires 3 arguments (arg_count, local_count, return_count)"
                            .to_string(),
                    ));
                }
                let arg_count = self.parse_u16(&args[0])?;
                let local_count = self.parse_u16(&args[1])?;
                let return_count = self.parse_u16(&args[2])?;
                Ok(Instruction::DefineFunctionSignature(
                    arg_count,
                    local_count,
                    return_count,
                ))
            }

            // Local variable operations
            "local_load" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "local_load requires 1 argument".to_string(),
                    ));
                }
                let offset = self.parse_u16(&args[0])?;
                Ok(Instruction::LoadLocal(offset))
            }

            "local_store" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "local_store requires 1 argument".to_string(),
                    ));
                }
                let offset = self.parse_u16(&args[0])?;
                Ok(Instruction::StoreLocal(offset))
            }

            "arg_load" => {
                if args.len() != 1 {
                    return Err(AssemblerError::MissingArgument(
                        "arg_load requires 1 argument".to_string(),
                    ));
                }
                let offset = self.parse_u16(&args[0])?;
                Ok(Instruction::LoadArg(offset))
            }

            _ => Err(AssemblerError::UnknownInstruction(instruction.clone())),
        }
    }

    fn parse_u64(&self, s: &str) -> Result<u64, AssemblerError> {
        if s.starts_with("0x") || s.starts_with("0X") {
            u64::from_str_radix(&s[2..], 16)
                .map_err(|_| AssemblerError::InvalidNumber(s.to_string()))
        } else {
            s.parse::<u64>()
                .map_err(|_| AssemblerError::InvalidNumber(s.to_string()))
        }
    }

    fn parse_u16(&self, s: &str) -> Result<u16, AssemblerError> {
        if s.starts_with("0x") || s.starts_with("0X") {
            u16::from_str_radix(&s[2..], 16)
                .map_err(|_| AssemblerError::InvalidNumber(s.to_string()))
        } else {
            s.parse::<u16>()
                .map_err(|_| AssemblerError::InvalidNumber(s.to_string()))
        }
    }

    fn parse_branch_target(&self, s: &str) -> Result<u16, AssemblerError> {
        // First try to parse as a label
        if let Some(&target) = self.labels.get(s) {
            if target > u16::MAX as usize {
                return Err(AssemblerError::InvalidArgument(format!(
                    "Label target {} is too large",
                    target
                )));
            }
            return Ok(target as u16);
        }

        // Otherwise parse as a number
        self.parse_u16(s)
    }

    fn parse_bytes(&self, s: &str) -> Result<&'bytes_arena [u8], AssemblerError> {
        if s.starts_with("0x") || s.starts_with("0X") {
            // Hex literal
            let hex_str = &s[2..];
            if hex_str.len() % 2 != 0 {
                return Err(AssemblerError::InvalidHexValue(
                    "Hex string must have even length".to_string(),
                ));
            }

            let mut bytes = Vec::new();
            for chunk in hex_str.as_bytes().chunks(2) {
                let hex_byte = core::str::from_utf8(chunk).map_err(|_| {
                    AssemblerError::InvalidHexValue("Invalid UTF-8 in hex string".to_string())
                })?;
                let byte = u8::from_str_radix(hex_byte, 16).map_err(|_| {
                    AssemblerError::InvalidHexValue(format!("Invalid hex byte: {}", hex_byte))
                })?;
                bytes.push(byte);
            }

            let arena_bytes = self.bytes_arena.alloc_slice_copy(&bytes);
            Ok(arena_bytes)
        } else {
            Err(AssemblerError::InvalidArgument(
                "Bytes must be a string literal \"...\" or hex literal 0x...".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn test_basic_instructions() {
        let bytes_arena = Bump::new();
        let program_arena = Bump::new();
        let mut assembler = Assembler::new(&bytes_arena);

        let source = r#"
            int 42
            int 10
            +
            pop
        "#;

        let instructions = assembler.assemble(source, &program_arena).unwrap();
        assert_eq!(instructions.len(), 4);

        match &instructions[0] {
            Instruction::PushInt(value) => assert_eq!(*value, 42),
            _ => panic!("Expected PushInt"),
        }

        match &instructions[1] {
            Instruction::PushInt(value) => assert_eq!(*value, 10),
            _ => panic!("Expected PushInt"),
        }

        match &instructions[2] {
            Instruction::Add => {}
            _ => panic!("Expected Add"),
        }

        match &instructions[3] {
            Instruction::Pop => {}
            _ => panic!("Expected Pop"),
        }
    }

    #[test]
    fn test_labels_and_branches() {
        let bytes_arena = Bump::new();
        let program_arena = Bump::new();
        let mut assembler = Assembler::new(&bytes_arena);

        let source = r#"
            int 0
            bz end
            int 42
        end:
            int 99
        "#;

        let instructions = assembler.assemble(source, &program_arena).unwrap();
        assert_eq!(instructions.len(), 4);

        match &instructions[1] {
            Instruction::BranchZero(target) => assert_eq!(*target, 3), // Should branch to instruction 3 (int 99)
            _ => panic!("Expected BranchZero"),
        }
    }

    #[test]
    fn test_function_call() {
        let bytes_arena = Bump::new();
        let program_arena = Bump::new();
        let mut assembler = Assembler::new(&bytes_arena);

        let source = r#"
            int 5
            int 10
            call add_function
            return
        add_function:
            func_def 2 0 1
            arg_load 0
            arg_load 1
            +
            return_func
        "#;

        let instructions = assembler.assemble(source, &program_arena).unwrap();
        assert_eq!(instructions.len(), 9);

        match &instructions[2] {
            Instruction::Call(target) => assert_eq!(*target, 4), // Should call instruction 4
            _ => panic!("Expected Call"),
        }

        match &instructions[4] {
            Instruction::DefineFunctionSignature(args, locals, returns) => {
                assert_eq!(*args, 2);
                assert_eq!(*locals, 0);
                assert_eq!(*returns, 1);
            }
            _ => panic!("Expected DefineFunctionSignature"),
        }
    }

    #[test]
    fn test_bytes_parsing() {
        let bytes_arena = Bump::new();
        let program_arena = Bump::new();
        let mut assembler = Assembler::new(&bytes_arena);

        let source = r#"
            bytes 0x48656c6c6f // "This is a comment"
        "#;

        let instructions = assembler.assemble(source, &program_arena).unwrap();
        assert_eq!(instructions.len(), 1);

        match &instructions[0] {
            Instruction::PushBytes(bytes) => assert_eq!(*bytes, b"Hello"),
            _ => panic!("Expected PushBytes"),
        }
    }

    #[test]
    fn test_comments_and_empty_lines() {
        let bytes_arena = Bump::new();
        let program_arena = Bump::new();
        let mut assembler = Assembler::new(&bytes_arena);

        let source = r#"
            // This is a comment
            int 42  // Inline comment
            
            // Empty line above
            +
        "#;

        let instructions = assembler.assemble(source, &program_arena).unwrap();
        assert_eq!(instructions.len(), 2);
    }
}
