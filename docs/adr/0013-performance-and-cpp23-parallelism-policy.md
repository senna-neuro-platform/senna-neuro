# ADR-0013: Performance and C++23 Parallelism Policy

- Status: Accepted
- Date: 2026-03-09

## Context

The project already fixes the architectural layers, deterministic mode, and the Python bindings
contract. Future work needs a stable policy for optimization so that performance changes do not
degrade correctness, determinism, or maintainability.

The project explicitly allows spending additional memory to reduce CPU cost, move hot-path work
from Python into C++, and use modern C++23 concurrency features where they preserve the runtime
model.

## Decision

1. Performance-sensitive execution paths must live in the C++ core. Python remains orchestration,
   configuration, reporting, and artifact control unless a Python step is demonstrably cold.
2. Cross-language APIs should prefer batch-oriented operations over per-sample or per-tick calls.
   New Python-to-C++ interfaces should minimize boundary crossings and should prefer contiguous
   buffers over Python object collections.
3. Memory may be used aggressively for speed when it improves hot-path performance without changing
   model semantics. Preferred techniques include precomputed lookup tables, dense index arrays,
   cached intermediate representations, buffer reuse, and preallocated workspaces.
4. Hot-path data structures should prefer contiguous and cache-friendly layouts. Dense arrays,
   spans, bitmaps, offset tables, and structure-of-arrays layouts are preferred over node-based
   containers and hash containers when identifiers are dense and bounded.
5. Repeated heap allocation in hot paths is prohibited unless justified by measurement. Buffers used
   per tick, per sample, or per batch must be reused or preallocated where practical.
6. Optimization must preserve the current network architecture and simulation semantics unless a
   separate ADR explicitly changes them. Event ordering, delay semantics, neuron behavior, and the
   fixed-seed deterministic mode remain normative.
7. Parallelism must be implemented in C++23 and only where semantic equivalence is clear. Preferred
   targets are embarrassingly parallel preprocessing, batch preparation, encoding, cache building,
   index rebuilds, and independent reductions.
8. Parallel execution must preserve deterministic behavior in fixed-seed mode. Parallel code must
   avoid unordered floating-point reductions, race-prone shared mutable state, and nondeterministic
   iteration order unless the result is explicitly defined as order-independent.
9. The simulation tick loop and event-delivery semantics are sequential by default. Any proposal to
   parallelize the core event-processing path requires explicit proof of semantic equivalence and
   dedicated regression coverage.
10. New performance work on the Python binding boundary should use zero-copy or near-zero-copy
    transfer where possible, such as buffer protocol support and contiguous numeric arrays.
11. Performance changes to the public Python API should extend the existing contract instead of
    breaking it. New optimized entry points may be added, but existing bindings should remain valid
    until a separate ADR approves a contract change.
12. Performance work must be evidence-driven. Changes to hot code must include either a benchmark,
    a profile-based justification, or a clearly documented reason that the change removes obvious
    asymptotic or allocation overhead.
13. Compiler-level optimization is allowed but must be policy-driven. Release builds should keep an
    optimized portable configuration, while machine-specific flags, profile-guided optimization, and
    other host-tuned settings must remain opt-in presets.
14. Concurrency primitives should use the standard C++23 toolbox by default. `std::jthread`,
    `std::stop_token`, `std::barrier`, `std::latch`, `std::span`, and standard execution policies
    are preferred over ad hoc threading infrastructure when they fit the problem.
15. Every optimization that changes a core algorithm or data structure must retain or improve test
    coverage for correctness, determinism, persistence compatibility, and the Python integration
    path.
16. Optimization work is not complete until it passes the project quality gate: formatting, lint,
    relevant tests, and the required version/changelog update policy from ADR-0001.

## Consequences

- Future changes have a consistent bias toward C++ hot-path execution and memory-for-speed
  tradeoffs.
- The repository gains an explicit rule set for cache-friendly code and bounded use of C++23
  parallelism.
- Determinism and the current network model remain protected from unsafe or premature
  parallelization.
- Optimization work becomes easier to review because it is expected to be benchmarked, compatible
  with the current API surface, and aligned with the existing ADR set.
