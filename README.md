# SENNA Neuro

**Spatial-Event Neuromorphic Network Architecture** is a spiking neuromorphic system
that processes information as sparse events on a 3D lattice of excitatory/inhibitory neurons.

Stimuli arrive at a sensory panel, propagate through local neighborhoods,
adapt via STDP and homeostasis, and are decoded by a compact output layer
for classification (MNIST, target accuracy >85%).

## Architecture

```
   Sensory Panel (Z=0)        Processing Volume (Z=1..18)       Output Layer (Z=19)
  ┌──────────────────┐       ┌───────────────────────────┐       ┌───────────────┐
  │  28x28 = 784     │──────>│  3D lattice, density 70%  │──────>│  10 neurons   │
  │  rate-coded input│       │  LIF neurons (80% E/20% I)│       │  WTA decoder  │
  └──────────────────┘       │  STDP + homeostasis       │       └───────────────┘
                             │  structural plasticity    │
                             └───────────────────────────┘
```

| Component            | Description                                                |
|----------------------|------------------------------------------------------------|
| `spatial/`           | 28x28x20 3D lattice, neighbor search in CSR format         |
| `neural/`            | LIF neurons, SoA pool, lazy membrane decay                 |
| `synaptic/`          | Synapses with weights and delays, CSR indices (in/out)     |
| `temporal/`          | Event queue (MPSC), virtual time manager                   |
| `plasticity/`        | STDP, threshold homeostasis, structural plasticity         |
| `encoding/`          | Rate coding (input), WTA first-spike decoder (output)      |
| `observability/`     | Prometheus metrics, statistics collector                   |
| `network/`           | Network assembly, main spike loop                          |
| `interfaces/`        | gRPC API (trainer), WebSocket API (visualizer)             |

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

# Configure (Conan + CMake)
make configure-debug

# Build
make build-debug

# Test
make test

# Or all at once (test depends on build)
make test
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

| Parameter               | Value               |
|-------------------------|---------------------|
| Lattice                 | 28x28x20            |
| Density                 | 70%                 |
| E/I neuron ratio        | 80% / 20%           |
| Neighbor radius         | 2-3 voxels          |
| V_rest, V_reset         | 0.0                 |
| tau_m                   | 20 ms               |
| Refractory period       | 2 ms                |
| Base threshold theta    | 1.0                 |
| STDP A+ / A-            | 0.01 / 0.012        |
| STDP tau                | 20 ms               |
| Homeostasis r_target    | 5 Hz                |
| WTA weight              | -5.0                |
| Rate coding max_rate    | 100 Hz              |
| Presentation time       | 50 ms               |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
