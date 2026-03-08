# ADR-0005: MVP Scope Boundaries

Status: Accepted  
Date: 2026-03-07

## Context

The MVP must have explicit scope boundaries so that implementation priorities do not drift.

## Decision

1. The MVP includes: simulation core, baseline plasticity (STDP/homeostasis/structural), baseline IO, serialization, metrics, visualizer, and training pipeline.
2. The MVP excludes: glia, neurogenesis, sleep mode, fractal topology, R-STDP, and other post-MVP extensions.
3. New mechanisms are added only after MVP criteria have been passed.

## Consequences

- The risk of schedule slip from early adoption of complex subsystems is reduced.
- The boundary between MVP and the roadmap stays explicit.
