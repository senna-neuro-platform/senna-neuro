# SENNA Neuro

Spatial-Event Neuromorphic Network Architecture.

## Summary

SENNA Neuro is an event-driven spiking neural network MVP focused on a 3D lattice-based neuromorphic architecture for MNIST-class classification.

- C++23 core: domain model, simulation engine, plasticity, persistence, metrics.
- Python layer: pybind11 bindings, training pipeline, configuration, acceptance tooling.
- Runtime stack: Prometheus, Grafana, Three.js visualizer, MinIO artifact storage, Docker Compose orchestration.

The project goal is not just to train a classifier, but to make the network observable, reproducible, and operable as a full neuromorphic experimentation stack.

## How to use

Typical local workflow:

1. Install dependencies and project data:

```bash
make install
```

2. Build and validate the release runtime:

```bash
make build-release
ctest --preset release
```

3. Bring up the observability and runtime stack:

```bash
make up
```

4. Run training on real MNIST:

```bash
PYTHONPATH=build/release:python python3 python/train.py \
  --config configs/default.yaml \
  --dataset mnist \
  --data-root data \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --ticks 100 \
  --target-accuracy 0.85
```

5. For the full acceptance flow, evidence checklist, and DoD validation, see the acceptance runbook:

- [`docs/acceptance/README.md`](docs/acceptance/README.md)

## Current status

Already implemented in the repository:

- event-driven 3D lattice with neuron, synapse, event queue, time manager, and network builder;
- plasticity stack for STDP, homeostasis, and structural plasticity;
- encoder/decoder path and Python training pipeline for real MNIST runs;
- HDF5 state serialization and epoch artifact outbox with MinIO background upload;
- metrics collection, Prometheus exporter, Grafana dashboards, and live Three.js visualizer;
- unit, integration, lint, sanitize, CI, and acceptance runbooks for MVP closure.

## Planned

Near-term work:

- finish full MVP DoD closure on real long-run acceptance: target accuracy, sparsity, robustness, and operational evidence;
- continue performance work on training/runtime throughput and observability responsiveness;
- refine visualizer, dashboards, and acceptance automation from operator feedback.

Beyond MVP:

- richer input/output pipelines:
  - temporal, latency, rank-order, phase, and population encoding;
  - alternative output decoding beyond first-spike and winner-take-all;
  - broader dataset adapters beyond MNIST for more realistic benchmarks;
- extended learning rules:
  - reward-modulated plasticity such as R-STDP;
  - short-term plasticity, metaplasticity, and neuromodulatory control;
  - more configurable supervision and hybrid training regimes;
- deeper biological realism that was intentionally excluded from MVP:
  - additional neuron models and dendritic compartment behavior;
  - glial support processes, sleep-like regimes, and neurogenesis;
  - more expressive spatial topologies and non-uniform connectivity structures;
- performance and scaling work after profiling:
  - queue/runtime optimization on hot paths;
  - CPU parallelization where determinism and model semantics allow it;
  - GPU acceleration for the simulation and training hot paths where the data layout and execution model justify it;
  - evaluation of CUDA and other accelerator backends for larger experiments and faster iteration cycles;
  - larger experiments, faster training throughput, and better hardware utilization;
- stronger experiment operations:
  - richer artifact lineage and experiment tracking;
  - more automated acceptance, benchmarking, and regression baselines;
  - more robust deployment patterns for long-running training and batch inference;
- richer observability and operator UX:
  - more informative Grafana panels and runtime diagnostics;
  - better visualizer modes for trace analysis, slicing, and comparison between samples/runs;
  - cleaner workflows for inspecting robustness, sparsity, and structural plasticity over time.

## Dev commands

Make shortcuts:

```bash
make install
make lint
make build-debug
make build-release
make build-sanitize
make test
make up
make down
make logs
```

Build presets:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

```bash
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
```

Conan dependencies:

```bash
conan profile detect --force
conan install . --output-folder=build/conan-debug --build=missing -s build_type=Debug
```

Observability stack:

```bash
docker compose up -d
docker compose down
```

## Endpoints

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Prometheus Exporter: http://localhost:8000/metrics
- Visualizer: http://localhost:8080
- Visualizer WebSocket: ws://localhost:8080/ws
- MinIO API: http://localhost:9000
- MinIO Console: http://localhost:9001

## Visualizer

- Runtime stream:
  - `GET /lattice` returns static lattice geometry.
  - `WS /ws` streams active neurons only in the format:
    - `{ tick, neurons: [{ id, x, y, z, type, fired }], activeCount, totalNeurons }`
- UI controls:
  - pause/resume,
  - `Next Tick` frame-by-frame mode,
  - playback speed slider,
  - neuron type filters (E/I/Output),
  - `Z` layer slice slider,
  - heatmap mode (activity over recent ticks).

## Metrics and Dashboards

- C++ core metrics collector: `src/core/metrics/metrics_collector.h`.
- Prometheus exporter: `infra/simulator/simulator_server.py`.
- Exported metrics include:
  - `senna_active_neurons_ratio`
  - `senna_spikes_per_tick`
  - `senna_e_rate_hz`, `senna_i_rate_hz`, `senna_ei_balance`
  - `senna_train_accuracy`, `senna_test_accuracy`
  - `senna_synapse_count`
  - `senna_stdp_updates_total`, `senna_pruned_total`, `senna_sprouted_total`
  - `senna_tick_duration_seconds` (histogram)
- Provisioned Grafana dashboards:
  - `SENNA Activity`
  - `SENNA Training`
  - `SENNA Performance`

## Artifact Upload (MinIO, Batch/Background)

- `docker compose up -d` now starts:
  - `minio` (S3-compatible storage),
  - `minio-init` (creates bucket `senna-artifacts`),
  - `artifact-uploader` (background batch uploader).
- MinIO stores HDF5 experiment artifacts from `data/artifacts/outbox`; MNIST input is read locally from `data/MNIST/raw` by `train.py` and is not uploaded to MinIO.
- Put epoch artifacts into `data/artifacts/outbox`, for example:
  - `data/artifacts/outbox/epoch_000000001.h5`
  - `data/artifacts/outbox/epoch_000000002.h5`
- Background uploader policy is configurable via `configs/storage/artifact_uploader.env`:
  - `UPLOAD_BATCH_EPOCHS`: upload when at least N epoch-files accumulated.
  - `UPLOAD_FLUSH_INTERVAL_SEC`: force upload old pending files even if batch not full.
  - `UPLOAD_MAX_BATCH_FILES`: hard cap per upload batch.
  - `UPLOAD_MIN_FILE_AGE_SEC`: skip very new files to avoid partial uploads.
- C++ persistence can now form outbox epoch artifacts automatically via
  `core/persistence/epoch_artifact_pipeline.h` (`EpochArtifactPipeline`), writing
  `data/artifacts/outbox/epoch_XXXXXXXXX.h5` and the main experiment file in one call.

## Python Bindings + Training

- `senna_core` pybind11 module is built by CMake (`src/bindings/python_module.cpp`).
- Python integration tests are discovered by CTest when `pytest` is available.

Build + run tests:

```bash
make test
```

Run training entrypoint:

```bash
PYTHONPATH=build/debug:python python3 python/train.py --config configs/default.yaml --dataset mnist --train-limit 60000 --test-limit 10000 --epochs 5
```

Notes:
- For real MNIST in `train.py`, install `torch` and `torchvision` in your Python env.
- Real MNIST is read locally from `data/MNIST/raw`; MinIO is not used as dataset storage.
- `train.py --dataset mnist` no longer performs automatic fallback: if MNIST or `torchvision` is unavailable, the run exits with an error.
- `--dataset synthetic` remains available only for explicit smoke/dev scenarios, not for acceptance.
- `train.py` prints mid-epoch progress and refreshes the live metrics snapshot during long epochs; the visualizer receives a bootstrap trace at run start and subsequent refreshes without synthetic data.
- `train.py` writes per-epoch checkpoints to `data/artifacts/outbox/epoch_XXXXXXXXX.h5`.
- Training and robustness metrics are appended as JSONL to `data/artifacts/training/metrics.jsonl`.
- A real exporter snapshot is written to `data/artifacts/metrics/latest.json`; until this file exists, exporter `/metrics` stays unavailable instead of fabricating values.
- A real visualizer trace is written to `data/artifacts/visualizer/latest.json`; until this file exists, visualizer `/lattice` stays unavailable instead of fabricating lattice/activity frames.
- Robustness checks are executed after training:
  - `remove_neurons(0.1)` with expected drop `<5%`
  - `inject_noise(0.3)` with expected drop `<10%`
