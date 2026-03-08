# ADR-0008: Template Usage Policy

Status: Accepted  
Date: 2026-03-07

## Context

The C++ core should follow a zero-overhead approach: reuse logic without unnecessary runtime polymorphism or extra allocations, especially on the simulation hot path.

## Decision

1. Templates are used where data-type or strategy variability affects performance or extensibility.
2. Clear aliases and concrete types are always provided for the default scenario (for example, `Coord3D`) so the API does not become harder to use.
3. Complex metaprogramming with no proven engineering benefit is not introduced.
4. Runtime polymorphism is kept for integration boundaries, not for the internal hot path where templates can express the same idea.

## Consequences

- The core gains extensibility without virtual-call overhead.
- The code stays readable through explicit aliases and a limited amount of template logic.
- New template-based elements are added only when there is clear engineering justification.
