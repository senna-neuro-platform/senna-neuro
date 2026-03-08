# ADR-0003: Core Technology Stack

Status: Accepted  
Date: 2026-03-07

## Context

The MVP technology stack and the responsibility split between languages need to be fixed explicitly.

## Decision

1. The simulator core is implemented in C++23.
2. The high-level API and training pipeline are implemented in Python.
3. The C++/Python bridge is implemented with pybind11.
4. Configuration is stored in YAML.

## Consequences

- The hot path stays performant.
- Experiments and orchestration remain fast to iterate on during development.
