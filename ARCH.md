## Eval Hierarchy

### One: Block

A collection of programs (at this point just bytecode) that need to be executed.

#### Responsibility

1. Disassemble bytecode to instructions
2. TODO: Use discriminate to determine which programs can be parallelized

#### Arenas & Lifetimes

##### bytes_arena

Allocates byte slices from the bytecode of programs

##### program_arena

Allocates disassembled program instructions

#### Allocations

1. Allocation for `program_map`, which is a HashMap mapping bytecode to programs to avoid the same bytecode being disassembled multiple times

### Two: Sequence

A collection of programs (now `Instruction`s) that need to be executed

#### Responsibility

Execute all programs in the sequence

#### Arenas & Lifetimes

None

#### Allocations

None

### Three: Eval

The actual evaluation

#### Responsibility

Given a program, execute it and determine the result

#### Arenas & Lifetimes

##### eval_arena

Contains allocations for everything that happens during evaluation such as new byte slices and vecs. Arena itself is long living (maybe `'static`?) and reset at the beginning of every eval

#### Allocations

None
