# ADR-0002: Architecture Baseline

Status: Accepted  
Date: 2026-03-07

## Context

The MVP needs a fixed architectural baseline so that domain logic does not get mixed with infrastructure concerns.

## Decision

1. Use a layered architecture with dependencies pointing inward.
2. `Domain` contains only business entities and does not depend on IO or infrastructure.
3. `Engine`, `Plasticity`, `IO`, `Persistence`, and `Metrics` are separate layers around the domain.
4. Execution orchestration is concentrated in `SimulationEngine`.

## Consequences

- Domain code remains reusable and testable.
- Infrastructure changes affect the core as little as possible.
