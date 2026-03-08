# Changelog

## `0.16.17-dev`
- Added a short `How to use` section to `README.md` with the basic local workflow (`install`, `build-release`, `make up`, training run) and an explicit link to the acceptance runbook in `docs/acceptance/README.md`.
- Translated the repository documentation to English: root `README.md`, the acceptance runbook, ADRs, and `CHANGELOG.md`.
- Added a GPU acceleration direction to the `Beyond MVP` roadmap in `README.md`: potential CUDA and other accelerator backend work is now explicitly recorded for faster simulation, training, and larger post-MVP runs.
- Expanded the `Beyond MVP` block in `README.md`: the post-MVP roadmap is now described in more detail across IO, learning rules, biological modeling, performance, observability, and experiment operations.
- Added a short high-level summary to `README.md` for SENNA Neuro: what the project is, which MVP parts are already implemented in the repository, and what is planned next.

## `0.16.12-dev`
- In `visualizer/index.html`, the visualizer scene was brightened further: the background and fog became lighter, ambient/hemisphere/fill lighting was increased, and voxel material was made less matte so the network no longer sinks into a dark background.
- In `visualizer/index.html`, palette contrast was increased again: the base colors of neuron types are brighter, and the heatmap now uses a more luminous scale with clear transitions between cool, green, and warm activity zones.
- In `visualizer/index.html`, the scene was tuned for better contrast: the background and lighting were made more distinct, and the base colors of excitatory, inhibitory, and output neurons were separated more strongly so voxels do not disappear into the background.
- In `visualizer/index.html`, the heatmap palette was rebuilt around the `navy -> cyan -> lime -> amber -> orange` scale, so activity levels are easier to distinguish and no longer collapse into a single shade.
- In `visualizer/index.html`, fallback playback no longer depends on WebSocket state: when the live queue is empty, the visualizer continues cycling through the latest real trace snapshot, so animation no longer freezes on a static cube.
- In `visualizer/index.html`, the visual response to activity was strengthened: base voxels became smaller and darker, while active neurons now pulse much more visibly in both size and brightness.
- In `visualizer/index.html`, the dependency on an external CDN was removed: `three.module.js` and `OrbitControls.js` are now loaded locally from `visualizer/vendor`, so the visualizer no longer stalls on the startup screen when `unpkg.com` is unavailable.
- In `visualizer/index.html`, background HTTP polling of `/trace` was added for offline/fallback mode, so the tab can pick up the real trace after backend startup without requiring a manual reload in edge cases.
- In `visualizer/server.js`, no-cache responses and a new HTTP `/trace` endpoint were added so the current full visualizer trace, including lattice and frame payload, is available without depending on the WebSocket stream.
- In `visualizer/index.html`, the visualizer was switched to bootstrap and fallback from `/trace`: the page loads the latest real trace over HTTP, can replay it when WebSocket degrades, and no longer gets stuck in `offline / Trace waiting / Buffered frames 0` if the backend trace is already ready.

## `0.16.7-dev`
- In `docs/acceptance/scripts/run_acceptance.sh`, the interactive `continue` pause was removed: after `make up` and `Observation memo`, the training run now starts immediately.
- In `docs/acceptance/README.md`, runbook was synchronized: references to `--no-observe-pause` and manual stopping before training starts were removed.
- In `Makefile`, `make up` was returned to a network-independent `docker compose up -d --force-recreate --no-build`, and a separate `make up-build` target was added for explicit runtime image rebuilds; this removes Docker Hub/TLS failures while preserving a controlled rebuild path when needed.
- In `python/train.py`, live snapshot and visualizer trace writing was changed from `tmp + replace()` to in-place updates with `fsync`, because Docker bind-mounted `simulator` and `visualizer` containers otherwise kept seeing the old bootstrap inode and missed live metric/trace updates during training.
- In `Makefile`, `make up` was changed to `docker compose up -d --build --force-recreate` so acceptance and manual runtime always bring up current `simulator` and `artifact-uploader` containers instead of keeping stale exporter code after repository changes.
- In `python/senna/training.py`, a mid-epoch progress callback was added for `train_epoch` and `evaluate`, so long-running MNIST epochs no longer look like a hung process with no stdout and no metrics.
- In `python/train.py`, `training_bootstrap`, periodic `progress ...` lines, live snapshots for `data/artifacts/metrics/latest.json`, and bootstrap/refresh trace output for `data/artifacts/visualizer/latest.json` were added so Grafana and the visualizer come alive before the first epoch finishes.
- In `docs/acceptance/README.md` and `README.md`, the new observable runtime contract was documented: long training runs must emit progress logs, mid-epoch exporter snapshots, and bootstrap/live visualizer traces.
- In `docker-compose.yml`, MinIO startup was made robust: an HTTP healthcheck was added, `minio-init` waits for `service_healthy` and `mc ready`, and `artifact-uploader` no longer starts in a race before MinIO is ready.
- In `docs/acceptance/scripts/run_acceptance.sh`, an early preflight for real MNIST was added: it checks host Python modules `torch` and `torchvision`, the presence of `data/MNIST/raw/*`, the MinIO health endpoint, and explicitly reminds that MinIO is used only for artifacts, not for dataset input.
- In `python/senna/training.py`, `python/train.py`, and `python/tests/test_integration.py`, the MNIST error contract was clarified: instead of a vague error, the training run now reports either missing `torch`/`torchvision` or missing local raw files, and this is covered by pytest tests.
- In `docs/acceptance/README.md`, `README.md`, and ADR-0011, documentation was synchronized: real MNIST is read locally from `data/MNIST/raw`, it requires `torch` and `torchvision`, and MinIO stores only epoch/state artifacts.
- In `docs/acceptance/scripts/run_acceptance.sh`, the acceptance runtime was moved to release builds: instead of `build/debug` and `make test`, it now uses `make build-release`, `ctest --preset release`, and `PYTHONPATH=build/release:python` for training and inference validation.
- In `docs/acceptance/scripts/run_acceptance.sh`, a required pause after `print_observe_stack_memo` was added: the script waited for the operator to open Grafana and the visualizer before continuing the training run after entering `continue`; a `--no-observe-pause` flag was added for non-interactive mode.
- In `docs/acceptance/README.md`, instructions were updated for the release runtime and the interactive observation pause before training.

## `0.16.0-dev`
- In `src/bindings/python_module.cpp`, `python/senna/training.py`, and `python/train.py`, export of the real visualizer trace was added: pybind returns lattice and per-tick activity, and the training run writes `data/artifacts/visualizer/latest.json` together with checkpoints and metric snapshots.
- In `visualizer/server.js`, `visualizer/index.html`, and `docker-compose.yml`, the visualizer was switched to reading only a real trace from `data/artifacts`; until the artifact appears, `/lattice` is not replaced with synthetic data, the UI waits honestly for the trace, and Docker mounts expose the shared artifact volume.
- In `python/senna/training.py`, robustness gate semantics were fixed: `prune_pass` and `noise_pass` can no longer pass when baseline accuracy is zero.
- In `docs/acceptance/scripts/run_acceptance.sh`, `docs/acceptance/README.md`, `README.md`, and ADR-0006, a new acceptance contract was fixed: exporter and visualizer use only real artifacts, orchestration clears stale traces, and waits for `/metrics` and `/lattice` after training.

## `0.15.6-dev`
- In `python/train.py`, the implicit fallback to synthetic data for `--dataset mnist` was disabled; the training run now requires real MNIST and writes a fresh exporter snapshot to `data/artifacts/metrics/latest.json`.
- In `infra/simulator/simulator_server.py`, the synthetic metrics fallback was removed: the exporter reads only a real snapshot, `/metrics` returns `503` until it exists, and `/health` explicitly reports `snapshot_ready`.
- In `src/core/metrics/metrics_collector.h`, `senna_max_active_neurons_ratio` was added, and `docs/acceptance/scripts/check_dod_metrics.py` was updated to validate maximum sparsity across all `epoch_end` records with a required `dataset_mode=mnist`.
- In `docs/acceptance/scripts/run_acceptance.sh` and `docs/acceptance/README.md`, steps 15 and 16 were synchronized: acceptance now works only with `mnist`, clears stale exporter snapshots, checks real `/metrics`, and uses correct Grafana metric names and WebSocket-check CLI flags.
- `README.md`, ADR-0010, and exporter/metrics tests were updated for the new contract without synthetic metrics.

## `0.15.5-dev`
- In `docs/acceptance/README.md`, the absolute local workspace path was removed; launch commands were rewritten in a neutral form without local environment details.
- In `.clang-tidy`, a pragmatic full-project check profile was fixed (`clang-analyzer`, `bugprone`, `performance`, and selected `modernize`/`readability`) for `src/` and `tests/`, suitable for a stable quality gate without noisy warnings.
- Added `scripts/run_clang_tidy.py`: a sequential `clang-tidy` runner across all translation units from `build/debug/compile_commands.json` with `--warnings-as-errors=*`.
- In `Makefile`, `make lint` was switched to a full run of `clang-format` + `scripts/run_clang_tidy.py` + `ruff check`, instead of checking only `src/main.cpp`.
- In `CMakeLists.txt`, the sanitize run for the Python integration test was fixed: when `SENNA_ENABLE_SANITIZERS=ON`, `pytest` automatically receives `LD_PRELOAD=<libasan.so>`, `ASAN_OPTIONS=detect_leaks=0`, and `UBSAN_OPTIONS=print_stacktrace=1`.
- In `.github/workflows/ci.yml`, `clang-tidy` was switched to the new full-project runner, and sanitize configure/build/test now runs on both `push` and `pull_request`.
- In `docs/acceptance/README.md`, closure scenarios were expanded: separate procedures for MNIST training, required evidence artifacts, manual Grafana/Visualizer checks, and exact commands for DoD 13 and 14.
- In `docs/acceptance/README.md`, an explicit `How to run the training run` section was added with two scenarios: through `run_acceptance.sh` in training-only mode and directly through `python/train.py`.
- The runbook now fixes the required training-run output artifacts: `metrics.jsonl`, `epoch_XXXXXXXXX.h5`, and `final_state.h5`.
- In `docs/acceptance/README.md`, intermediate observation notes were added between acceptance steps, including where to inspect Docker state, Grafana dashboards, the exporter, and the visualizer.
- In `docs/acceptance/scripts/run_acceptance.sh`, automatic `Observation memo` blocks were added after `make up`/health checks and after the training run, with live observation commands such as `docker compose ps`, `make logs`, `tail metrics.jsonl`, and exporter probes.
- Added the final MVP acceptance runbook at `docs/acceptance/README.md`, with a step-by-step scenario from environment bring-up through DoD gates.
- Added the orchestration script `docs/acceptance/scripts/run_acceptance.sh` for automated build/test/lint/sanitize execution, Docker health checks, the training run, and DoD validation.
- Added the script `docs/acceptance/scripts/check_dod_metrics.py` to validate numeric DoD gates in `metrics.jsonl` (`accuracy`, `active_ratio`, `prune_drop`, `noise_drop`).
- Added the script `docs/acceptance/scripts/check_inference_pipeline.py` to validate the path `state + sample -> prediction [0..9]` through Python bindings.
- Added the script `docs/acceptance/scripts/check_ws_sparsity.py` to validate visualizer frame sparsity over WebSocket (`activeCount/totalNeurons < 5%`) without external dependencies.
- In `src/bindings/python_module.cpp`, training loop was strengthened: `supervise()` now applies deterministic sensor-input weight updates toward the correct or incorrect output with a clamp on `stdp.w_max`.
- In the bindings, `encoder.max_rate` is now applied to input spike density and `decoder.W_wta` is applied to lateral inhibition (WTA) through injected inhibitory events.
- In `python/senna/training.py`, helper functions `evaluate_from_state` and `robustness_report` were added for reproducible evaluation of saved states and robustness checks.
- `python/train.py` was expanded to the full scenario: epoch checkpoints `epoch_XXXXXXXXX.h5`, early stop on `target_accuracy`, JSONL logging (`data/artifacts/training/metrics.jsonl`), diagnostic hints, and post-training checks `remove_neurons(0.1)` and `inject_noise(0.3)`.
- `configs/default.yaml` was updated with fixed hyperparameters, including `training.target_accuracy`, `training.learning_rate`, `encoder.max_rate`, and an updated `w_init_range`.
- Python integration tests in `python/tests/test_integration.py` were extended with a smoke check for `robustness_report`.
- Added ADR-0012 `docs/adr/0012-training-target-and-robustness-gates.md` to fix target quality gates.

## `0.14.0-dev`
- Added the pybind11 module `senna_core` in `src/bindings/python_module.cpp` with contract: `create_network`, `load_sample`, `step`, `get_prediction`, `get_metrics`, `save_state`, `load_state`, `inject_noise`, `remove_neurons`, `supervise`.
- Implemented strict YAML validation for `configs/default.yaml` in the C++ bindings through `yaml-cpp` (required sections and parameter ranges), without duplicating validation in the Python layer.
- Added a unified `configs/default.yaml` with all hyperparameter sections: `lattice`, `neuron`, `synapse`, `stdp`, `homeostasis`, `structural`, `encoder`, `decoder`, `training`.
- In `CMakeLists.txt`, the `senna_core` Python module build and `python/tests/test_integration.py` integration run through `pytest` were added to CTest.
- Added the Python training pipeline in `python/senna/training.py` and `python/train.py` (MNIST through `torchvision` with fallback to a synthetic dataset, using the `load_sample -> step -> predict -> supervise` loop).
- Added Python integration tests in `python/tests/test_integration.py` for the full API cycle, supervision, and save/load.
- In CI (`.github/workflows/ci.yml`), `pytest` installation was added for the Python integration test run.
- Added ADR-0011 `docs/adr/0011-python-bindings-and-training-contract.md` to fix the bindings and training-pipeline contract.

## `0.13.0-dev`
- Implemented the visualizer WebSocket server in `visualizer/server.js`: `/ws` endpoint, frame stream `{tick, neurons:[...]}` only for active neurons, `/lattice` for full lattice geometry, and `/health` for service checks.
- In `visualizer/server.js`, deterministic generation of the 3D lattice and the wave-pattern activity trace was added (interference fronts) with a sparsity cap for active neurons (<5% of all neurons per frame).
- Completely reworked `visualizer/index.html`: lattice rendering through `Three.js InstancedMesh`, type-based color coding (E/I/Output), spike flashes decaying over 3-5 frames, and an orbit camera.
- Added visualizer UI controls: pause/resume, frame-by-frame `Next Tick`, playback-speed slider, type filters, `Z`-layer slicing, and heatmap mode.
- Added a client-side WebSocket frame queue with auto-reconnect, enabling stable real-time mode and controlled frame-by-frame playback without losing visual continuity.
- Added runtime visualizer documentation to `README.md` (HTTP + WebSocket endpoints and supported modes/controls).

## `0.12.0-dev`
- Added `MetricsCollector` in `src/core/metrics/metrics_collector.h`: collects metrics from `SimulationEngine` events (spikes/tick), computes the active-neuron ratio, `spikes_per_tick`, average E/I rates, `ei_balance`, and STDP/structural-plasticity counters.
- Added snapshot export to a metric map (`as_metric_map`) in `MetricsCollector` with Prometheus-compatible names (`senna_*`) for later transfer into the Python exporter.
- Added GTest `tests/test_metrics.cpp`: validates metrics after 100 deterministic ticks and checks exported keys/values.
- In `CMakeLists.txt`, `test_metrics` was connected and a CTest `test_prometheus_exporter_format` was added to validate the Prometheus output format of the Python exporter.
- Rewrote `infra/simulator/simulator_server.py` into a Prometheus exporter with the full metric set (`active ratio`, `spikes/tick`, `E/I`, `train/test accuracy`, `synapse count`, `pruned/sprouted`, `tick_duration_seconds` histogram, `stdp_updates_total`).
- Added Python test `infra/simulator/test_simulator_server.py` for Prometheus payload validity and JSON snapshot loading.
- In `docker-compose.yml`, added `METRICS_SNAPSHOT_PATH` and the volume `./data/artifacts:/artifacts` for the `simulator` service so runtime metric snapshots are read automatically.
- Added three provisioned Grafana dashboard JSON files: `SENNA Activity`, `SENNA Training`, and `SENNA Performance`; removed the placeholder dashboard.
- Added metrics/dashboard documentation and the exporter endpoint `http://localhost:8000/metrics` to `README.md`.

## `0.11.3-dev`
- In `EpochArtifactPipeline` (`src/core/persistence/epoch_artifact_pipeline.h`), the epoch outbox filename format was extended to 9 digits: `data/artifacts/outbox/epoch_XXXXXXXXX.h5`.
- In `tests/test_persistence.cpp`, an explicit check for the new outbox filename format (`epoch_000000002.h5`) was added.
- Updated examples and the outbox-path description in `README.md` for the 9-digit epoch index.
- Added `EpochArtifactPipeline` in `src/core/persistence/epoch_artifact_pipeline.h`: one call writes epoch data into the main experiment HDF5 and automatically creates an outbox file `data/artifacts/outbox/epoch_XXXXXX.h5` for the background uploader.
- In `EpochArtifactPipeline`, an atomic outbox write pattern (`.tmp` -> rename) was implemented so the uploader never sees a partially written epoch artifact.
- Added support for storing a `/state` snapshot in the outbox through integration with `StateSerializer`, so recovery is possible directly from an epoch file.
- Added test `EpochArtifactPipelineTest.WritesEpochFileToOutboxAutomatically` in `tests/test_persistence.cpp` to validate outbox file generation and correct reading of the saved `/state`.
- Added documentation to `README.md` for automatic epoch-file generation from C++ persistence (`EpochArtifactPipeline`) so the MinIO uploader works out of the box.
- Added `minio`, `minio-init`, and `artifact-uploader` services to `docker-compose.yml` for S3-compatible artifact storage and background uploads.
- Added a containerized uploader (`infra/artifact-uploader/Dockerfile`, `infra/artifact-uploader/uploader.py`) with a batched-upload policy: threshold by number of epochs (`UPLOAD_BATCH_EPOCHS`) plus forced flush by timer (`UPLOAD_FLUSH_INTERVAL_SEC`), with a batch-size limit.
- Added `configs/storage/artifact_uploader.env` for uploader configuration (S3 endpoint/credentials/bucket/prefix and batched background-upload parameters).
- Added MinIO endpoints and artifact-flow documentation through `data/artifacts/outbox` with the background batch uploader to `README.md`.
- Added the Persistence module in `src/core/persistence/hdf5_writer.h`: read/write support for `spike_trace`, `snapshot` (neurons + synapses), and `metrics` in HDF5 with epoch-based grouping.
- Added `StateSerializer` in `src/core/persistence/state_serializer.h`: save/load for the full simulation state (neurons, synapses, pending events, `elapsed`, `dt`, `rng_state`) and runtime-structure restoration.
- In `Neuron` (`src/core/domain/neuron.h`), added a serializable `NeuronSnapshot` and `snapshot()/from_snapshot()/restore_from_snapshot()` APIs for round-trip state recovery.
- In `EventQueue` (`src/core/engine/event_queue.h`), added `snapshot()/restore()` for delayed-event serialization.
- Added GTest tests in `tests/test_persistence.cpp`: bitwise `spike_trace` round-trip, snapshot round-trip, metric round-trip, and deterministic simulation continuation after `save/load`.
- Connected `test_persistence` in `CMakeLists.txt` with `HDF5::HDF5` linkage and registered it in CTest through `gtest_discover_tests`.

## `0.10.0-dev`
- Implemented `StructuralPlasticity` in `src/core/plasticity/structural_plasticity.h`: pruning weak connections below `w_min`, sprouting new connections for quiet neurons (`r_avg < r_target * quiet_ratio`), and periodic execution every `N` ticks.
- In `StructuralPlasticity`, added the `prune + sprout + rebuild_indices` cycle with per-step `pruned/sprouted` metrics and cumulative counters.
- Sprouting uses `Lattice::neighbors` within a configured radius, filters out existing connections, and creates new synapses with `sprout_weight`.
- Added GTest tests in `tests/test_structural_plasticity.cpp`: removing a weak synapse, preserving a strong one, creating new inputs for a quiet neuron, keeping connection counts stable after prune+sprout, and interval-based execution every `N` ticks.
- Connected `test_structural_plasticity` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.9.0-dev`
- Implemented `Homeostasis` in `src/core/plasticity/homeostasis.h`: EMA-based `r_avg`, threshold adjustment toward `r_target`, clamping within `[theta_min, theta_max]`, and updates over a window of `N` ticks.
- In `Neuron` (`src/core/domain/neuron.h`), added homeostasis control methods: `set_average_rate`, `set_threshold`, and `adjust_threshold`.
- In `SimulationEngine`, extended the observer API with subscriptions for spikes and tick completion (`set_*`/`add_*` observers), enabling slow learning loops without embedding their logic into `tick()`.
- In `Network` (`src/core/engine/network_builder.h`), added proxy methods for observer subscriptions and non-const access to `neurons` and `synapses` for plasticity rules.
- Added GTest tests in `tests/test_homeostasis.cpp`: threshold increase for a hyperactive neuron, threshold decrease for a silent neuron, enforcement of `theta` bounds, and `r_avg` convergence to the target rate in a long run.
- Connected `test_homeostasis` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.8.0-dev`
- Added the plasticity interface `IPlasticityRule` in `src/core/plasticity/iplasticity_rule.h` with `on_pre_spike` and `on_post_spike` hooks.
- Implemented `STDPRule` in `src/core/plasticity/stdp.h`: causal and anti-causal weight updates over an exponential window, soft potentiation limiting, and hard clamping at `w_max`.
- Added `Supervisor` in `src/core/plasticity/supervisor.h` to emit a corrective teacher spike to the correct output neuron on classification error.
- In `SimulationEngine`, added a spike observer hook (`set_spike_observer`) so plasticity can integrate through event subscriptions.
- In `SynapseStore`, added non-const synapse access (`at`/`synapses`) for weight modification by plasticity rules.
- Added GTest tests in `tests/test_stdp.cpp`: causal and anti-causal pairs, effect decay at large `delta_t`, `w_max` enforcement, and weight growth toward the correct output under supervision.
- Connected `test_stdp` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.7.2-dev`
- In `Makefile`, added the `data-mnist` target: downloads MNIST into `data/MNIST/raw` idempotently with existing-file checks, and wires this step into `make install`.
- In `.gitignore`, added `data/` for locally downloaded datasets kept outside git.
- Renamed the IO interface headers to remove `_`: `src/core/io/i_encoder.h -> src/core/io/iencoder.h` and `src/core/io/i_decoder.h -> src/core/io/idecoder.h`; include references were updated accordingly.
- Added IO interfaces `IEncoder` and `IDecoder` in `src/core/io/iencoder.h` and `src/core/io/idecoder.h`.
- Implemented `RateEncoder` in `src/core/io/rate_encoder.h`: encodes `MNIST 28x28` into a `SpikeEvent` stream using `rate = pixel/255 * max_rate` and spike probability `rate * dt / 1000`.
- Implemented `FirstSpikeDecoder` in `src/core/io/first_spike_decoder.h`: decodes from the first output spike and generates lateral inhibition (WTA) for the remaining output neurons.
- In `SimulationEngine`, added export of spikes emitted during the current tick (`emitted_events_last_tick`) for collecting output activity in the end-to-end pipeline.
- In `Network`, added access to last-tick spikes (`emitted_spikes_last_tick`) for decoder integration.
- Added GTest tests in `tests/test_io.cpp`: `RateEncoder` checks (black/medium/white), `FirstSpikeDecoder` checks (first spike, tie-break, WTA), and the full `encode -> simulate -> decode` path.
- Connected `test_io` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.6.0-dev`
- Implemented `NetworkBuilder` and the aggregate `Network` in `src/core/engine/network_builder.h`: builds `Lattice -> SynapseStore -> EventQueue -> TimeManager -> SimulationEngine` with a deterministic seed.
- Added `inject_spike(NeuronId, Time)`, `tick()`, and `simulate(duration_ms)` to `Network` for the first end-to-end wave propagation through the network.
- In `SimulationEngine`, added the `emitted_last_tick` counter to record the number of generated spikes per tick.
- In `Lattice`, added non-const access to the neuron vector for `SimulationEngine` integration inside the aggregate `Network`.
- Added integration GTest tests in `tests/test_network_builder.cpp`: silence without a stimulus, wave propagation from a single stimulus, `1 vs 10` stimulus comparison, and deterministic trace behavior.
- Connected `test_network_builder` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.5.0-dev`
- Implemented `Engine: EventQueue` in `src/core/engine/event_queue.h`: a `std::priority_queue`-based event queue with the minimum `arrival` at the top, plus `push` and `drain_tick([t_start, t_end))` methods.
- Implemented `Engine: TimeManager` in `src/core/engine/time_manager.h`: stores virtual time, the `dt` step (default `0.5 ms`), and provides `advance`, `elapsed`, and `reset` methods.
- Implemented `Engine: SimulationEngine` in `src/core/engine/simulation_engine.h`: `tick()` delivers events to neurons, processes emitted spikes, and schedules new events across outgoing synapses (`arrival = spike_time + delay`, `value = weight * sign`).
- Added GTest tests in `tests/test_event_queue.cpp`: extraction order by time, tick-interval quantization, `A->B->C` propagation with delays, and an empty-tick check.
- Connected `test_event_queue` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.4.0-dev`
- Implemented `Domain: Lattice` in `src/core/domain/lattice.h`: lattice configuration, voxel storage (`NeuronId` or empty), flat arrays of `Neuron` and `NeighborInfo`.
- Added deterministic lattice generation: the sensor layer `Z=0` is fully populated, the processing volume `Z=1..D-2` is populated by density, and the output layer `Z=D-1` contains exactly 10 neurons distributed evenly.
- Fixed neuron-type placement rules: sensor and output layers are fully `Excitatory`, while the internal volume uses the `80/20` (`Excitatory`/`Inhibitory`) split.
- Added neighbor lookup `neighbors(NeuronId, radius)` with naive cube traversal and precomputation for the baseline radius in CSR form (`offsets + data`).
- Added GTest tests in `tests/test_lattice.cpp`: dimensions/density, layer correctness, neighbor checks (center and corner), and deterministic generation.
- Connected `test_lattice` in `CMakeLists.txt` and registered it in CTest through `gtest_discover_tests`.

## `0.3.2-dev`
- `Makefile` was moved to project Conan profiles (`build/conan/profiles/host|build`) with auto-generation from the local `g++` version, removing warnings from `conan profile detect`.
- In CI (`.github/workflows/ci.yml`), a step was added to prepare the same Conan profiles and add the `conancenter` remote only when needed, so warnings like `Remote ... already exists` no longer appear.
- For Conan commands in both `Makefile` and GitHub Actions, suppression of the known upstream deprecated warning (`core:skip_warnings=["deprecated"]`) was added for the `hdf5` recipe.
- Implemented `Domain: Synapse` in `src/core/domain/synapse.h`: `Synapse` (`pre_id`, `post_id`, `weight`, `delay`, `sign`) and `SynapseStore` with flat-array storage.
- In `SynapseStore`, added `outgoing` and `incoming` indices, `add`, `connect`, `connect_random`, `rebuild_indices`, and delay/sign calculation from distance and presynaptic neuron type.
- Added a GTest suite in `tests/test_synapse.cpp`: distance-based delay, E/I sign, random-weight range, index correctness, and a `~300k` synapse scale check.
- Connected `test_synapse` in CMake and ran it through `ctest`.

## `0.2.2-dev`
- Fixed ADR-0009: the C++ testing standard is GoogleTest and test-case registration in CTest through `gtest_discover_tests`.
- Updated the ADR index: ADR-0009 was added.
- Converted `tests/test_types.cpp` and `tests/test_neuron.cpp` to GoogleTest (`TEST`, `ASSERT_*`, `EXPECT_*`) instead of manual checks and a custom `main`.
- In `CMakeLists.txt`, tests are wired through `find_package(GTest)` and registered in `CTest` with `gtest_discover_tests`, so every case is visible as a separate test.
- Implemented `Domain: Neuron` in `src/core/domain/neuron.h`: LIF neuron state, parameters (`V_rest`, `V_reset`, `tau_m`, `t_ref`, `theta_base`), and `receive_input(Time, Weight) -> std::optional<SpikeEvent>`.
- In `receive_input`, added analytical membrane-potential decay, refractory-window checks, spike generation with state reset, and deterministic internal-state updates.
- Added tests in `tests/test_neuron.cpp`: decay, firing and reset, refractory period, E/I spike sign, and determinism.
- Connected `test_neuron` in CMake and ran it through `ctest`.

## `0.1.4-dev`
- In CI (`.github/workflows/ci.yml`), Conan install now runs with `-s compiler.cppstd=gnu23`.
- In `Makefile`, Conan flag `-s compiler.cppstd=gnu23` was fixed for `install`, `build-release`, and `build-sanitize`.
- Fixed ADR-0008 for template usage policy in the C++ core.
- Added justified template usage in `Domain`: generic `Coord3<T>` (with alias `Coord3D`) and `ArrivalEarlier<EventT>`.
- Added tests for template scenarios (`Coord3<uint16_t>`/`Coord3<int>` and the templated event comparator).
- Fixed the baseline project ADR set (architecture, stack, simulation model, MVP boundaries, observability, quality gates).
- Added the ADR index: `docs/adr/README.md`.
- Fixed ADR-0001 with versioning rules (`A.B.C-dev`) and changelog maintenance policy.
- Implemented Domain in `src/core/domain/types.h`: baseline types (`NeuronId`, `SynapseId`, `Time`, `Voltage`, `Weight`), `NeuronType`, `Coord3D::distance()`, and `SpikeEvent`.
- Added the `test_types` unit test covering distance, `SpikeEvent` ordering, and neuron-type distinction; the test is wired into `ctest`.

## `0.0.5-dev`
- Baseline project infrastructure: CMake/Ninja, Conan, CI, Docker Compose.
- Placeholder `simulator`/`prometheus`/`grafana`/`visualizer` services and the initial bootstrap `README`.
- Added `Makefile` commands: `install`, `lint`, `build-debug`, `build-release`, `build-sanitize`, `test`.
- Added container control commands: `make up`, `make down`, `make logs`.
- Added a block with quick `make` commands to `README`.
- Resolved the conflict of duplicate CMake presets during repeated `make build-*` calls through Conan.
- `build-debug`, `build-release`, `build-sanitize`, `test`, and `lint` now run stably inside the same working copy.
- Conan commands in `Makefile` now disable generation of `CMakeUserPresets.json` to avoid preset conflicts across repeated builds.
