# Changelog

## `0.30.4-dev`
- Structural plasticity implemented: pruning of sub-threshold non-WTA synapses, sprouting for quiet neurons within a radius with configurable initial weight and delay scaling.
- RCU synapse index swaps: synapse store now uses atomic shared_ptr; spike loop and STDP worker load current snapshots each tick/batch, keeping the main loop lock-free.
- Background structural worker runs pruning/sprouting on trigger/interval ticks; integrated into spike loop; graceful stop and tests covering prune, sprout, and background execution.
- Runtime config, default YAML, and README configuration table now expose structural parameters (w_min_prune, interval_ticks, sprout_radius, sprout_weight, quiet_fraction).
- Unit and integration suites updated; all tests pass after structural additions.

## `0.29.5-dev`
- Homeostasis refactor: dedicated plasticity module with Hz target, theta_min/theta_max, interval_ticks, and background worker applying double-buffered thresholds without blocking the spike loop.
- NeuronPool now double-buffers thresholds, exposes snapshots/apply helpers, and keeps r_avg smoothing separate from threshold updates.
- TimeManager schedules homeostasis snapshots every `interval_ticks`, runs worker thread, and atomically swaps new theta buffers; Network attaches pool automatically.
- Runtime config/README updated with new homeostasis fields; defaults set to spec (alpha 0.999, target 5 Hz, step 0.001, bounds 0.1-5.0, interval 10).
- Tests updated for new API and bounds; added clamp coverage; `HomeostasisParametersAffectRate` fixed.

## `0.28.6-dev`
- STDP worker (lock-free MPSC) runs in background, supervision helper added, weight tapering/limits enforced, and expanded STDP/worker tests (all 96 tests green).
- Wired STDP into the spike loop: pre/post spikes update outgoing/incoming synapses using STDP params from config.
- Added STDP params to YAML (`stdp` block) and to `NetworkConfig`; RateEncoder/TimeManager/decoder already pull config values.
- STDP module implemented with unit tests (`test_plasticity` target).
## `0.27.9-dev`
- RuntimeConfig now fully drives network setup: YAML values populate lattice, synapse (incl. WTA), LIF, homeostasis, encoder params, decoder window, dt/seed; Network/TimeManager/encoder constructors consume these instead of hardcoded defaults; SpikeLoop seeds decoder with network seed/window.

## `0.27.8-dev`
- Stochastic WTA tie-break: event delivery order per tick is shuffled, giving a single random winner when outputs receive equal inputs; default WTA weight restored to doc value (-5) while remaining configurable.
- Streaming decoder: `SpikeLoop` can attach a `FirstSpikeDecoder`, auto-starts its 50 ms window at run start, and runs optionally on a dedicated worker thread (`RunInThread`) to match DoD Step 5.2. Decoder API now exposes `Reset(t_start)` to sync window automatically.
- Decoder integration tests updated to streaming path; TimeManager now seeds randomness from network seed for repeatable stochastic behavior.
- EventQueue remains lock-free MPSC; build fixes applied.
- Added YAML runtime config (`configs/default.yaml`) with lattice, synapse, homeostasis, encoder/decoder, and simulation seeds; introduced `RuntimeConfig` loader (yaml-cpp) and exposed `decoder_window_ms` in `NetworkConfig`.

## `0.27.7-dev`
- Homeostasis now blends per-neuron activity with global firing fraction passed from `TimeManager`, exposing configurable alpha/target/step and tests for smoothing/target alignment and global influence.
- Spike loop delivers tick events per-neuron in parallel (thread pool via `std::execution::par`), then derives run stats/global activity for homeostasis to align with DoD parallel update requirements.
- Decoder gains a 50 ms decision window: `FirstSpikeDecoder` tracks start time, ignores spikes after expiry, and exposes `ResultWithTimeout`; unit and integration tests updated. WTA inhibition default strengthened (`w_wta=-12`) and integration test enforces a single output winner for similar inputs.
- EventQueue reworked to lock-free MPSC producer path (atomic pending stack) with single-consumer heap ordering to match DoD concurrency requirements; boundary and ordering semantics preserved.

## `0.27.6-dev`
- Added homeostasis updates in `TimeManager`/`NeuronPool` and unit tests for LIF decay/refractory/threshold, E/I ratio, and synapse index properties (distance-based delay, weight bounds/signs, WTA params).
- Tightened decoder/WTA integration: earliest output still wins, decoder stays empty without output spikes, and WTA weight increased in test to enforce a single winner.
- Added event-queue boundary test for < `t_end` draining behavior.

## `0.27.5-dev`
- Added unit tests for neuron pool LIF dynamics and synapse index (delay-distance, weight bounds, sign, WTA parameters).
- Decoder/output integration tightened: earliest output wins even under WTA, decoder stays empty with no output spikes, outputs verified to have incoming volume synapses.
- Added first-spike decoding integration: network smoke test now feeds output spikes through the decoder to ensure the earliest firing output neuron is selected.
- Added integration checks for output-layer WTA: output neurons validated for count, top-plane placement, and inhibitory zero-delay WTA links with configured weight across all outputs.
- Introduced a decoding module with a first-spike decoder (`src/core/decoding/first_spike_decoder.*`) and documentation (`docs/decoding.md`); added unit tests in `tests/decoding/first_spike_decoder_test.cpp` and wired them into CMake.
- Exposed output-layer metadata in `Network` (`output_ids()`) to support decoders and diagnostics, and added integration checks for output count, top-plane placement, and WTA fan-out.
- Strengthened `RateEncoder` tests with brightness proportionality and distinct-target assertions plus helper utilities for averaged counts.
- Wired the input encoder into the network: `Network` now owns a `RateEncoder` and exposes `EncodeImage(...)` to enqueue sensory spikes directly into the event queue, keeping encoder `dt`/seed aligned with the simulation config.
- Added an integration smoke test in `tests/integration/network_smoke_test.cpp` that encodes an image and verifies spike activity after propagation.

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
