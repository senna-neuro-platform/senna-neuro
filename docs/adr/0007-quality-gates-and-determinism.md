# ADR-0007: Quality Gates and Determinism

Status: Accepted  
Date: 2026-03-07

## Context

Minimum engineering quality criteria are needed for every change.

## Decision

1. Mandatory checks in CI: build, `ctest`, static analysis (`clang-tidy`), and Python lint (`ruff`).
2. Pull requests are additionally checked in a sanitize configuration.
3. Experiments and tests must support deterministic mode with a fixed seed.
4. Before handing off a change that modifies code, build configuration, or runtime behavior, the
   local engineering loop must run `make fmt`, `make lint`, and relevant tests for the affected
   paths, unless a concrete blocker is documented in the handoff.
5. `make lint` is a production-code quality gate first. It must prioritize project code in `src/`,
   primary Python runtime code, infrastructure code, and project scripts. Tests may have their own
   checks, but they are not the primary lint scope.
6. Any code handoff must also comply with ADR-0001: version and changelog updates are part of the
   quality gate, not optional release-time housekeeping.

## Consequences

- The risk of hidden regressions and nondeterministic failures is reduced.
- The engineering loop is stabilized for the next MVP steps.
- Handoffs become stricter and more reproducible because formatting, lint, tests, version, and
  changelog updates are expected together.
