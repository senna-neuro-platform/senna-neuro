# ADR-0010: Metrics Contract and Export Path

- Status: Accepted
- Date: 2026-03-07

## Context

Needs a stable and testable observability path: metric collection in the core, export to Prometheus, and visualization in Grafana.

## Decision

1. Metrics are collected in C++ through `MetricsCollector` (`src/core/metrics/metrics_collector.h`).
2. `MetricsCollector` does not perform IO; it only aggregates a snapshot and returns a `senna_*` metric map.
3. Export to Prometheus is implemented in Python (`infra/simulator/simulator_server.py`) over HTTP `/metrics`.
4. The exporter reads only a real runtime snapshot from a JSON file (`METRICS_SNAPSHOT_PATH`); when the snapshot is missing, `/metrics` is not replaced with synthetic data.
5. Tick duration is published as a histogram (`senna_tick_duration_seconds`).
6. Grafana dashboards are provisioned from the repository as JSON and maintained as three dashboards: Activity, Training, and Performance.

## Consequences

- The core stays isolated from the network and HTTP layer.
- The metrics contract is unified (`senna_*`) across C++, Prometheus, and Grafana.
- The exporter format can be checked locally and in CI without starting the full stack.
