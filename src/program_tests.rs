extern crate alloc;

use crate::GLOBAL;
use crate::assembler::Assembler;
use crate::eval::{StackValue, ZapEval};
use bumpalo::Bump;
use stats_alloc::Region;

/// Helper function to run an assembly program and check the result
fn run_assembly_program(source: &str, expected_stack: &[StackValue]) {
    let bytes_arena = Bump::new();
    let program_arena = Bump::new();
    let eval_arena = Bump::new();

    // Assemble the source code
    let mut assembler = Assembler::new(&bytes_arena);
    let instructions = assembler
        .assemble(source, &program_arena)
        .expect("Failed to assemble program");

    // Execute the program
    let mut eval = ZapEval::new(&eval_arena, &instructions);
    let region = Region::new(GLOBAL);
    eval.run();
    let alloc_stats = region.change();

    assert_eq!(alloc_stats.allocations, 0);
    assert_eq!(alloc_stats.reallocations, 0);

    // Check the result
    assert_eq!(
        eval.stack.as_slice(),
        expected_stack,
        "Expected stack: {:?}, but got: {:?}",
        expected_stack,
        eval.stack.as_slice()
    );
}

fn rust_fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    rust_fibonacci(n - 1) + rust_fibonacci(n - 2)
}

#[test]
fn test_fibonacci_sequence() {
    let n = 6;
    let source = format!(
        r#"
        // Compute fibonacci(6) = 8
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

    let expected_stack = [StackValue::U64(rust_fibonacci(n))]; // fib(6) = 8
    run_assembly_program(&source, &expected_stack);
}

#[test]
fn test_fibonacci_edge_cases() {
    // Test fib(0) = 0
    let source_fib_0 = r#"
        int 0
        call fib_function
        return

    fib_function:
        func_def 1 0 1
        arg_load 0
        int 1
        <=
        bz recursive_case
        arg_load 0
        return_func
    recursive_case:
        arg_load 0
        int 1
        -
        call fib_function
        arg_load 0
        int 2
        -
        call fib_function
        +
        return_func
    "#;

    let expected_stack_0 = [StackValue::U64(0)];
    run_assembly_program(source_fib_0, &expected_stack_0);

    // Test fib(1) = 1
    let source_fib_1 = r#"
        int 1
        call fib_function
        return

    fib_function:
        func_def 1 0 1
        arg_load 0
        int 1
        <=
        bz recursive_case
        arg_load 0
        return_func
    recursive_case:
        arg_load 0
        int 1
        -
        call fib_function
        arg_load 0
        int 2
        -
        call fib_function
        +
        return_func
    "#;

    let expected_stack_1 = [StackValue::U64(rust_fibonacci(1))];
    run_assembly_program(source_fib_1, &expected_stack_1);
}

#[test]
fn test_simple_arithmetic_program() {
    // Test a simple program that computes (5 + 3) * 2 = 16
    let source = r#"
        int 5
        int 3
        +           // 5 + 3 = 8
        int 2
        *           // 8 * 2 = 16
    "#;

    let expected_stack = [StackValue::U64(16)];
    run_assembly_program(source, &expected_stack);
}

#[test]
fn test_factorial_program() {
    // Test factorial function: factorial(5) = 120
    let source = r#"
        int 5
        call factorial
        return

    factorial:
        func_def 1 2 1  // 1 arg, 2 locals, 1 return
        
        // Check if n <= 1 (base case)
        arg_load 0
        int 1
        <=
        bz not_base_case
        
        // Base case: return 1
        int 1
        return_func

    not_base_case:
        // Initialize: result = 1, i = 2
        int 1
        local_store 0   // result = 1
        int 2
        local_store 1   // i = 2

    loop:
        // Check if i > n
        local_load 1    // load i
        arg_load 0      // load n
        >               // i > n?
        bnz end_loop

        // result = result * i
        local_load 0    // load result
        local_load 1    // load i
        *               // result * i
        local_store 0   // result = result * i

        // i++
        local_load 1    // load i
        int 1
        +               // i + 1
        local_store 1   // i = i + 1

        // Continue loop
        b loop

    end_loop:
        // Return result
        local_load 0
        return_func
    "#;

    let expected_stack = [StackValue::U64(120)]; // 5! = 120
    run_assembly_program(source, &expected_stack);
}

#[test]
fn test_branching_program() {
    // Test a program with conditional branching
    let source = r#"
        int 7
        int 5
        >           // 7 > 5 = 1 (true)
        bz else_branch
        
        // if branch (7 > 5 is true, so this executes)
        int 100
        return       // Exit early to avoid executing else_branch
        
    else_branch:
        int 200
    "#;

    let expected_stack = [StackValue::U64(100)];
    run_assembly_program(source, &expected_stack);
}

