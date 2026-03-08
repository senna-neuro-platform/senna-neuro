# ADR-0007: Quality Gates and Determinism

Status: Accepted  
Date: 2026-03-07

## Context

Minimum engineering quality criteria are needed for every change.

## Decision

1. Mandatory checks in CI: build, `ctest`, static analysis (`clang-tidy`), and Python lint (`ruff`).
2. Pull requests are additionally checked in a sanitize configuration.
3. Experiments and tests must support deterministic mode with a fixed seed.

## Consequences

- The risk of hidden regressions and nondeterministic failures is reduced.
- The engineering loop is stabilized for the next MVP steps.
