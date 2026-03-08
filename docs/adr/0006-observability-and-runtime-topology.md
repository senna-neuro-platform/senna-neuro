# ADR-0006: Observability and Runtime Topology

Status: Accepted  
Date: 2026-03-07

## Context

A standard runtime topology is needed for local runs and diagnostics.

## Decision

1. Use Docker Compose as the standard way to bring the environment up.
2. The baseline services are `simulator`, `prometheus`, `grafana`, and `visualizer`.
3. Metrics are exported in Prometheus format and charted in Grafana.
4. `simulator` and `visualizer` read only real artifacts from shared `data/artifacts`; when snapshot or trace data is absent, the services honestly switch to a waiting state without synthetic fallback.

## Consequences

- Local startup is reproducible with one command.
- Development and CI diagnostics share one observability format.
- Manual and automated acceptance are not masked by artificial data.
