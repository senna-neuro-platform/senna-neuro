# SENNA Neuro

**Spatial-Event Neuromorphic Network Architecture** is a spiking system on a 3D lattice (28×28×20). Stimuli hit a sensory panel, propagate through neighbors, adapt via STDP/homeostasis, and a WTA first-spike decoder in the output layer selects the class (MNIST).

## Architecture

```
   Sensory Panel (Z=0)          Processing Volume (Z=1..18)       Output Layer (Z=19)
  ┌────────────────────┐       ┌────────────────────────────┐       ┌───────────────┐
  │  28x28 = 784       │──────>│  3D lattice, density 70%   │──────>│  10 neurons   │
  │  rate-coded input  │       │  LIF neurons (80% E/20% I) │       │  WTA decoder  │
  └────────────────────┘       │  STDP + homeostasis        │       └───────────────┘
                               │  structural plasticity     │
                               └────────────────────────────┘
```

| Component         | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `spatial/`        | 3D lattice, CSR neighbors                                     |
| `neural/`         | LIF (SoA), lazy decay, homeostasis                            |
| `synaptic/`       | Weight+delay, CSR in/out, WTA output links                    |
| `temporal/`       | Lock-free MPSC queue, time quantization, parallel tick        |
| `encoding/`       | MNIST rate encoder, WTA first-spike decoder                   |
| `network/`        | Network assembly, SpikeLoop with streaming decoder            |
| `plasticity/`     | STDP, structural plasticity                                   |
| `observability/`  | Metrics/statistics                                            |
| `interfaces/`     | gRPC (trainer), WebSocket (visualizer)                        |

## Quick Start

### Prerequisites

- C++23 compiler (GCC 13+ / Clang 17+)
- CMake >= 3.25
- Conan 2.x
- Ninja
- Docker + Docker Compose (for the full stack)

### Build & Test

```bash
cd senna-neuro

make configure-debug   # Conan + CMake
make build-debug       # Build
make test              # Build+test
```

### Build Variants

| Target                  | Command                           |
|-------------------------|-----------------------------------|
| Debug                   | `make build-debug`                |
| Release                 | `make build-release`              |
| Sanitizers (ASan+UBSan) | `make build-sanitize`             |
| Tests (debug)           | `make test`                       |
| Tests (release)         | `make test-release`               |
| Tests (sanitize)        | `make test-sanitize`              |
| Single test suite       | `ctest --preset debug -R spatial` |

### Docker Compose

```bash
make up       # Start: core (50051/8080/9090), Prometheus (9091), Grafana (3000), visualizer (8081)
make down     # Stop
make logs     # Tail logs
make ps       # Service status
```

## Code Quality

```bash
make lint     # clang-format --dry-run + clang-tidy
make fmt      # Auto-format (src/ + tests/)
```

## Configuration

Main settings live in `configs/default.yaml` and load into `RuntimeConfig`.

| Section        | Parameters (key: meaning) |
| -------------- | ------------------------- |
| `simulation`   | `dt`: tick size (ms); `seed`: RNG seed |
| `lattice`      | `width/height/depth`: grid dims; `density`: fill ratio; `neighbor_radius`: neighbor search radius (voxels); `num_outputs`: output neurons; `excitatory_ratio`: E/I split |
| `lif`          | `V_rest`, `V_reset`, `tau_m`: membrane decay (ms); `t_ref`: refractory (ms); `theta_base`: initial threshold |
| `synapse`      | `w_min/w_max`: init weight bounds; `c_base`: delay per voxel (ms); `w_wta`: inhibitory WTA weight |
| `stdp`         | `A_plus/A_minus`: LTP/LTD amplitudes; `tau_plus/tau_minus` (ms); `w_max`: STDP cap |
| `homeostasis`  | `alpha`: smoothing; `target_rate`: Hz; `theta_step`: Δθ per Hz error; `theta_min/theta_max`: bounds; `interval_ticks`: apply cadence; `global_mix`: blend global vs local activity |
| `encoder`      | `max_rate`: Hz; `presentation_ms`: stimulus window; `input_value`: event amplitude |
| `decoder`      | `window_ms`: first-spike decision window |

All values feed directly into `Network`, `TimeManager`, `RateEncoder`, decoder (no hardcoded defaults).

## Project Structure

```
senna-neuro/
├── CMakeLists.txt            # Main build file
├── CMakePresets.json         # Presets: debug, release, sanitize
├── conanfile.py              # Dependencies: Boost, gRPC, spdlog, fmt, yaml-cpp, GTest
├── Makefile                  # Wrapper over CMake/Conan/Docker
├── docker-compose.yml
├── src/
│   ├── core/
│   │   ├── spatial/          # Lattice, ZonedLattice, NeighborIndex
│   │   ├── neural/           # Neuron, NeuronPool (SoA)
│   │   ├── synaptic/         # Synapse, SynapseIndex (CSR)
│   │   ├── temporal/         # EventQueue, TimeManager
│   │   ├── plasticity/       # STDP, Homeostasis, Structural
│   │   ├── encoding/         # RateEncoder, WTADecoder
│   │   ├── observability/    # MetricsCollector, PrometheusExporter
│   │   ├── network/          # NetworkBuilder, SpikeLoop
│   │   └── interfaces/       # gRPC, WebSocket servers
│   ├── trainer/              # MNIST loader, training pipeline
│   └── proto/                # Protobuf schema
├── tests/
│   ├── spatial/              # Lattice and neighbor index tests
│   ├── neural/
│   ├── synaptic/
│   ├── temporal/
│   ├── plasticity/
│   ├── encoding/
│   └── integration/
├── configs/                  # default.yaml, Prometheus, Grafana dashboards
├── docker/                   # Dockerfiles: core, trainer, visualizer
└── visualizer/               # Three.js visualizer (WebSocket)
```

## Key Parameters (MVP)

| Parameter           | Value      |
|---------------------|------------|
| Lattice             | 28×28×20   |
| Density             | 70%        |
| E/I ratio           | 80% / 20%  |
| Neighbor radius     | 2–3        |
| tau_m               | 20 ms      |
| t_ref               | 2 ms       |
| theta_base          | 1.0        |
| STDP A+/A-          | 0.01/0.012 |
| STDP tau            | 20 ms      |
| Homeostasis target  | 5 Hz       |
| WTA weight          | -5.0       |
| Encoder max_rate    | 100 Hz     |
| Presentation time   | 50 ms      |

## Data

- Download MNIST once: `make install-mnist` (stores in `data/MNIST/raw`).

## License

Apache License 2.0 - see [LICENSE](LICENSE).
