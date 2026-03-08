# ADR-0004: Simulation Model and Time

Status: Accepted  
Date: 2026-03-07

## Context

The MVP needs one consistent computational model so that `Domain` and `Engine` are implemented compatibly.

## Decision

1. The base model is a 3D lattice with event-driven (spike-based) processing.
2. Events are delivered through a queue ordered by arrival time.
3. Virtual time is quantized (`dt = 0.5 ms`).
4. The MVP neuron model is LIF with excitatory/inhibitory types and local connectivity.

## Consequences

- The simulator is deterministic for a fixed seed and configuration.
- The architecture is ready for later expansion of plasticity rules.
