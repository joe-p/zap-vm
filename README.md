# Zap VM

Zap stands for Zero-cost Abstraction Platform. Zap VM intends to be a simple stack-based VM that offers high-level abstractions, like heterogeneous vectors, with a low performance overhead. In particular, there are zero allocations that occur _during_ evaluation. To accomplish this, every allocation happens in a bump allocator that is allocated on the heap once and can be reused across any number of evaluations.

## Inspiration & Aspiration: Algorand Virtual Machine

Zap VM is a VM largely inspired by the Algorand Virtual Machine and one days aspires to be fully backwards compatible with AVM bytecode. AVM compatibility, however, is a later goal and not the current focus. This sort of integration would require Zap VM to either: A) include AVM-specific implementation details or B) enable modular opcodes, like RISC-V ecalls. A more modular approach is desirable to keep Zap environment-agnostic. Integration into an Algorand client, such as `go-algorand`, is even further in the future and some serious architecture work needs to be done on both sides for that to happen.

## Why?

### Why a New VM

I am working on Zap VM mainly as a hobby project, but it was an idea conceived out of the complexities I've witnessed when trying to create easy-to-use languages for the Algorand Virtual Machine. The AVM is a simple, elegant stack machine that supports many operations, but does not have support for abstract data structures. Developers, however, want to use these data structures and thus we end up with languages like TEALScript (which I created and maintain) and Puya (Python and TypeScript), which I have done some work on. The biggest source of complexity, by far, in these implementations is support for complex data structures. This complexity, of course, also means it's one of the most common areas for bugs to occur as well.

The complexity required to implement complex data structures in the AVM also means the resultant TEAL (the AVM instruction set) is very complex, computationally expensive, and hard to audit. A VM that natively supports complex data structures in an elegant way removes all of those downsides.

### Why Rust

The AVM is currently implemented in Go, so one might logically think Go would be the best language to write a new VM in (or even better, to contribute directly to the AVM). The core problem with Go is the lack of memory control. In particular, the proposal to add memory arenas to go "is on hold indefinitely due to serious API concerns." Without a memory arena, it is near impossible to avoid allocations and related performance penalties when working with data structures like vectors (golang slices). Without even getting to vectors, the performance hit of allocations can be seen when comparing the benchmark results of uint512 math between the AVM and Zap (same exact benchmark on my M4 Pro):

AVM: ~134 ns/op (7 allocs/op)
Zap: ~17 ns/op (0 allocs/op)

To be clear, Rust is not magically faster. The go implementation can probably be improved for byte math and in many areas the performance is the same. Basics ops like `+` and `pop` are around 14ns in both and compute heavy algorithims like `ed25519_verify` perform about the same. Like all software, it's a matter of implementation and optimizations, but the memory control rust offers makes the optimizations more feasible. For this reason Zig also seems like an ideal candidate due to it's first-class support for allocators, but Rust and bumpalo so far seem to do everything that is needed (and allocators API is in nightly if it's really needed, but I am trying to stay on stable).
