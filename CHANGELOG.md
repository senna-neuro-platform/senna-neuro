# Changelog

## `0.26.0-dev`
- Implemented rate-based input encoding: [src/core/encoding/rate_encoder.hpp](src/core/encoding/rate_encoder.hpp) / [.cpp](src/core/encoding/rate_encoder.cpp) generate Poisson spike trains from 28×28 images onto the sensory panel, configurable by max rate, presentation window, and injected event value.
- Added encoding unit tests in [tests/encoding/rate_encoder_test.cpp](tests/encoding/rate_encoder_test.cpp) covering pixel-level behavior, brightness scaling, determinism, presentation window bounds, and whole-image statistics.

## `0.25.0-dev`
- Added full network wiring in [src/core/network/network_builder.hpp](src/core/network/network_builder.hpp) and [src/core/network/network_builder.cpp](src/core/network/network_builder.cpp): assembles zoned lattice, neighbor index, neuron pool, synapse index (with WTA), event queue, and time manager into a cohesive `Network`, with helpers to inject spikes or sensory inputs.
- Implemented the simulation driver [src/core/network/spike_loop.hpp](src/core/network/spike_loop.hpp) / [.cpp](src/core/network/spike_loop.cpp): runs ticked propagation over a duration, records spike logs, and returns run statistics.
- Added integration smoke tests in [tests/integration/network_smoke_test.cpp](tests/integration/network_smoke_test.cpp) to validate construction, silence without stimuli, spike propagation from injected stimuli, sensory injection, determinism, and spike logging.

## `0.24.0-dev`
- Added temporal event infrastructure: thread-safe priority-based [EventQueue](src/core/temporal/event_queue.hpp) with batch push/drain and a ticked [TimeManager](src/core/temporal/time_manager.hpp) that delivers events, triggers spikes via `NeuronPool`, fans out through `SynapseIndex`, and advances simulation time.
- Introduced temporal unit tests in [tests/temporal/event_queue_test.cpp](tests/temporal/event_queue_test.cpp) covering queue ordering, draining windows, concurrency, and TimeManager spike propagation; registered `test_temporal` targets alongside existing suites in [CMakeLists.txt](CMakeLists.txt).

## `0.23.0-dev`
- Added synapse domain model in [src/core/synaptic/synapse.hpp](src/core/synaptic/synapse.hpp): presynaptic/postsynaptic IDs, signed effective weight, distance-proportional delay, and tunable init params (weight range, delay scale, WTA weight).
- Built a CSR-based `SynapseIndex` in [src/core/synaptic/synapse_index.hpp](src/core/synaptic/synapse_index.hpp) and [src/core/synaptic/synapse_index.cpp](src/core/synaptic/synapse_index.cpp): constructs synapses from neighbor lists with random weights and delays, exposes incoming/outgoing views, and optionally wires WTA inhibition across output neurons.

## `0.22.2-dev`
- Extended the LIF storage layer: [src/core/neural/neuron_pool.hpp](src/core/neural/neuron_pool.hpp) and [src/core/neural/neuron_pool.cpp](src/core/neural/neuron_pool.cpp) now handle lazy membrane decay, input integration, threshold check, and spike reset via `ReceiveInput`/`Fire`, keeping refractory handling consistent with MVP params.
- Added an end-to-end lattice smoke test in [tests/integration/lattice_smoke_test.cpp](tests/integration/lattice_smoke_test.cpp) and wired a `test_smoke` target in [CMakeLists.txt](CMakeLists.txt) to cover Step 1 DoD (density bounds, neighbor counts, sensory/output zones, determinism, parallel build parity).

## `0.22.1-dev`
- Introduced a LIF neuron domain model in [src/core/neural/neuron.hpp](src/core/neural/neuron.hpp): excitatory/inhibitory types, default MVP parameters, refractory checks, and a compact AoS state view.
- Added a structure-of-arrays `NeuronPool` in [src/core/neural/neuron_pool.hpp](src/core/neural/neuron_pool.hpp) and [src/core/neural/neuron_pool.cpp](src/core/neural/neuron_pool.cpp): initializes from a lattice, assigns E/I types by ratio and seed, exposes SoA field accessors plus AoS gather/scatter helpers for simulation logic.

## `0.21.3-dev`
- Added `ZonedLattice` in [src/core/spatial/lattice.hpp](src/core/spatial/lattice.hpp) and [src/core/spatial/lattice.cpp](src/core/spatial/lattice.cpp): enforces a fully populated sensory panel on Z=0, clears/places a fixed number of output neurons on the top plane, and exposes helpers to fetch sensory/output neuron IDs.
- Adjusted lint/format scopes in [Makefile](Makefile): `lint`/`fmt-check` target only `src/`, while `fmt` now reformats both `src/` and `tests/`.

## `0.21.2-dev`
- Implemented a parallel neighbor-index builder in [src/core/spatial/neighbor_index.hpp](src/core/spatial/neighbor_index.hpp) and [src/core/spatial/neighbor_index.cpp](src/core/spatial/neighbor_index.cpp): precomputes CSR neighbor lists within a configurable radius using multi-threaded voxel scans, exposes constant-time span accessors, and keeps distance values alongside neuron IDs.
- Added spatial unit tests for lattice determinism/density and neighbor symmetry/count/distance coverage in [tests/spatial/lattice_test.cpp](tests/spatial/lattice_test.cpp) and [tests/spatial/neighbor_index_test.cpp](tests/spatial/neighbor_index_test.cpp); wired them into CTest via the `test_spatial` target in [CMakeLists.txt](CMakeLists.txt).
- Fixed the Conan/CMake preset alignment: [CMakePresets.json](CMakePresets.json) now uses the flat `build/<preset>` layout, and [conanfile.py](conanfile.py) defers layout so generated toolchains land where presets expect; [Makefile](Makefile) passes `-s build_type=...` during `conan install` to keep dependency builds consistent with presets.
- Leftover runtime stubs remain unchanged; version bumped to `0.21.2-dev` in [VERSION](VERSION).

## `0.21.1-dev`
- Reset the repository for the next-generation prototype: removed the prior MVP runtime, Python training pipeline, acceptance runbooks/tests, and vendored visualizer assets to start from a minimal scaffold.
- In [CMakeLists.txt](CMakeLists.txt) and [CMakePresets.json](CMakePresets.json), rebuilt the build system around C++23 with Ninja presets for debug/release/sanitize, optional ASan/UBSan via `SENNA_ENABLE_SANITIZERS`, and distinct `senna_core`/`senna_trainer` executables linked through a shared `senna_core_lib` target.
- In [Makefile](Makefile), rewrote the developer workflow: `configure-*` presets drive Conan + CMake, `build-*`/`test-*` mirror the presets, `fmt`/`tidy` cover C++ sources, and docker helpers build/push versioned core/trainer/visualizer images and manage the compose stack.
- Added CI on GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) that installs Conan, runs `make lint`, builds the debug preset, and executes the CTest suite on Ubuntu.
- Introduced a new container stack in [docker-compose.yml](docker-compose.yml) with dedicated Dockerfiles for core, trainer, and visualizer; Prometheus/Grafana provisioning now ships default scrape config and empty placeholder dashboards in [configs/prometheus/prometheus.yml](configs/prometheus/prometheus.yml) and [configs/grafana/*](configs/grafana).
- Implemented a stub core runtime in [src/core/main.cpp](src/core/main.cpp) that opens TCP listeners on gRPC (50051), WebSocket (8080), and metrics (9090) ports, replying `200 OK` on the metrics socket and shutting down cleanly on SIGINT/SIGTERM.
- Added a trainer bootstrap in [src/trainer/main.cpp](src/trainer/main.cpp) that waits for `SENNA_CORE_HOST:SENNA_CORE_PORT` to accept connections before idling, enabling compose-start sequencing.
- Implemented a deterministic voxel lattice generator in [src/core/spatial/lattice.hpp](src/core/spatial/lattice.hpp) and [src/core/spatial/lattice.cpp](src/core/spatial/lattice.cpp): fills a 3D grid by density/seed, tracks neuron IDs, and exposes coordinate lookups.
- Replaced the heavy visualizer with a minimal WebSocket status stub in [visualizer/index.html](visualizer/index.html), [visualizer/app.js](visualizer/app.js), and [visualizer/package.json](visualizer/package.json) served via `http-server` on port 8081.
- Updated Conan metadata in [conanfile.py](conanfile.py) to declare the new dependency set (Boost 1.88, gRPC 1.78, spdlog 1.14, fmt 10.2, yaml-cpp 0.8, GTest 1.17) and generate CMake toolchains/layout.
