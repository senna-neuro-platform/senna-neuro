# SENNA Neuro

Spatial-Event Neuromorphic Network Architecture.

## Summary
- Spatial-Event Neuromorphic Network Architecture: spiking SNN with 3D lattice topology, event-queued dynamics, and pluggable plasticity/observability layers.
- Core components: lattice + neighbor lookup, neural/synaptic/temporal pipelines, encoding/decoding, gRPC & WebSocket interfaces, and Prometheus/Grafana for runtime telemetry.
- Delivery model: C++23 codebase with Conan+CMake+Ninja toolchain, containerized services (core, trainer, visualizer, observability) for local runs and CI.

## Quick Start
1. Prereqs: `conan` 2.x, `cmake` ≥3.25, `ninja`, `clang-format`, `clang-tidy`.
2. Configure deps: `make configure-debug` (uses Conan with `-s build_type=Debug`; swap to `configure-release`/`configure-sanitize` as needed).
3. Build & test: `make build-debug` then `make test` (or `test-release`/`test-sanitize`).
4. Run stub stack: `make up` (uses `docker-compose.yml`) → core on 50051/8080/9090, Prometheus 9091, Grafana 3000 (anonymous viewer), visualizer 8081; stop with `make down`.

## Repo Map (short)
- `src/core/spatial`: lattice generation and neighbor index (`lattice.*`, `neighbor_index.*`).
- `src/core/main.cpp`: stub listeners for gRPC/WebSocket/metrics.
- `tests/spatial`: lattice/neighbor index unit tests registered in CTest.
- `docker/*`, `configs/*`, `visualizer/*`: compose-ready stubs for runtime + dashboards.
- `docs/SENNA-Neuro-MVP-Implementation.md`: step-by-step MVP plan.
