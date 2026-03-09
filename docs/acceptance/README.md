# Acceptance Runbook

Goal: validate the MVP DoD from environment bring-up through the final PASS/FAIL decision.

## Scope

- Step-by-step execution scenario.
- Automation for the key checks.
- Explicit PASS/FAIL criteria.

## Files

- `docs/acceptance/scripts/run_acceptance.sh`
- `docs/acceptance/scripts/run_e2e_smoke.sh`
- `docs/acceptance/scripts/check_e2e_smoke.py`
- `docs/acceptance/scripts/check_dod_metrics.py`
- `docs/acceptance/scripts/check_inference_pipeline.py`
- `docs/acceptance/scripts/check_ws_sparsity.py`

## Quick start

```bash
cd senna-neuro
chmod +x docs/acceptance/scripts/run_acceptance.sh
docs/acceptance/scripts/run_acceptance.sh
```

For a faster deployment-to-training validation, use the smoke path:

```bash
cd senna-neuro
chmod +x docs/acceptance/scripts/run_e2e_smoke.sh
docs/acceptance/scripts/run_e2e_smoke.sh
```

The smoke path uses a very small dataset slice by default (`synthetic`, `epochs=5`, `train-limit=8`, `test-limit=4`, `ticks=16`) and ends with a JSON verdict containing key metrics, endpoint checks, artifact counts, and uploader progress.

The script will execute:

1. `make install`, `make build-release`, `ctest --preset release`
2. `make lint`
3. `make build-sanitize` + `ctest --preset sanitize`
4. `make up` + endpoint health checks
5. a full training run on real MNIST 60k/10k without synthetic fallback
6. DoD metric checks (`accuracy`, `robustness`, `max_active_ratio`)
7. inference pipeline validation (`MNIST image -> class 0..9`)
8. WebSocket sparsity validation (`activeCount/totalNeurons < 0.05`) on the real visualizer trace
9. mid-epoch progress in stdout, live metrics snapshots for Grafana, and bootstrap/live trace output for the visualizer before the first epoch finishes

## Preconditions

The following must be available:

1. `python3`, `conan`, `cmake`, `ninja`, `clang-tidy`, `docker compose`, `ruff`, `pytest`
2. `g++` with `libasan.so`
3. the 4 MNIST raw files in `data/MNIST/raw` (`make install` downloads them automatically)
4. `torch` and `torchvision` available to the host Python runtime; the recommended project-local install location is `.python-packages/`
5. local access to ports `3000`, `8000`, and `8080`
6. MinIO is used in this scenario only for uploading artifacts from `data/artifacts/outbox`; the training run reads MNIST locally from `data/MNIST/raw`, not from S3/MinIO

Baseline preparation:

```bash
make install
make build-release
ctest --preset release
```

`make install` now also populates `.python-packages/` with the required host-side Python packages (`numpy`, `pytest`, `ruff`, `torch`, `torchvision`). `docs/acceptance/scripts/run_acceptance.sh` and `docs/acceptance/scripts/run_e2e_smoke.sh` prepend `.python-packages` automatically, so the preflight import check, the training process, and the post-run validation scripts all use the same project-local host packages.

## Observation notes between steps

1. After `make up`, check container status with `docker compose ps`, logs with `make logs`, and verify that `http://localhost:3000`, `http://localhost:8080/health`, and `http://localhost:8000/health` are reachable.
2. After `make up`, verify MinIO via `http://localhost:9000/minio/health/live` and `http://localhost:9001`; it is needed only for background artifact upload, not for reading MNIST.
3. During the training run, follow `tail -f data/artifacts/training/metrics.jsonl`; in parallel, open `SENNA Training` and `SENNA Activity` in Grafana.
4. Right after the training run starts, check stdout: it must show `training_bootstrap`, `progress ...`, and `live_trace_refreshed ...`; this is the main sign that a long-running epoch is not stuck.
5. After the training run starts, check `data/artifacts/metrics/latest.json` and `http://localhost:8000/metrics`: the exporter must publish real metrics already in the middle of the epoch from the live snapshot, not only at `epoch_end`.
6. After the training run starts, check `data/artifacts/visualizer/latest.json` and `http://localhost:8080/lattice`: the visualizer must receive a bootstrap trace at the beginning of the run and then update without synthetic data.
7. After the training run starts, check `docker compose logs -f artifact-uploader`: the uploader should pick up `epoch_XXXXXXXXX.h5` and `final_state.h5` from `data/artifacts/outbox` and send them to MinIO in batches.
8. After the DoD metric checks, compare `eval_accuracy`, `senna_max_active_neurons_ratio`, `prune_drop`, and `noise_drop` in JSONL with the Grafana graphs to confirm that the telemetry matches.
9. After the WebSocket sparsity check, open `http://localhost:8080`, enable heatmap and frame-by-frame `Next Tick` mode, and visually verify the wave pattern and sparsity on the real trace.

## How to run the training run

### Lightweight E2E smoke

This path validates the full operational chain with a tiny workload:

1. release build for the host Python runtime
2. `docker compose` bring-up for simulator, Prometheus, Grafana, visualizer, MinIO, and artifact uploader
3. small training run on a synthetic dataset
4. checkpoint and final state export into the outbox
5. live metrics snapshot for the exporter and trace snapshot for the visualizer
6. uploader state growth after HDF5 artifacts are flushed to MinIO
7. state-load inference validation from the saved `final_state.h5`
8. final `PASS` or `FAIL` verdict with key metrics

Run it with:

```bash
make e2e-smoke
```

Or directly:

```bash
docs/acceptance/scripts/run_e2e_smoke.sh \
  --epochs 5 \
  --train-limit 8 \
  --test-limit 4 \
  --ticks 16
```

Smoke artifacts are written into:

1. `data/artifacts/e2e-smoke/<run-id>/metrics.jsonl`
2. `data/artifacts/e2e-smoke/<run-id>/train.log`
3. `data/artifacts/e2e-smoke/<run-id>/verdict.json`
4. `data/artifacts/outbox/<run-id>/epoch_*.h5`
5. `data/artifacts/outbox/<run-id>/final_state.h5`

`data/artifacts/metrics/latest.json`, `data/artifacts/visualizer/latest.json`, and `data/artifacts/uploader_state.json` remain the shared live contract for exporter, visualizer, and uploader checks.

Option A: through the orchestrator (recommended)

```bash
docs/acceptance/scripts/run_acceptance.sh \
  --skip-build \
  --skip-lint \
  --skip-sanitize \
  --skip-docker \
  --skip-ws-sparsity \
  --dataset mnist \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

Option B: directly through `python/train.py`

```bash
make install
make build-release
PYTHONPATH=.python-packages:build/release:python python3 python/train.py \
  --config configs/default.yaml \
  --dataset mnist \
  --data-root data \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --ticks 100 \
  --target-accuracy 0.85 \
  --progress-every 50 \
  --live-trace-every 250 \
  --checkpoint-dir data/artifacts/outbox \
  --state-out data/artifacts/outbox/final_state.h5 \
  --metrics-out data/artifacts/training/metrics.jsonl \
  --metrics-snapshot-path data/artifacts/metrics/latest.json \
  --visualizer-trace-path data/artifacts/visualizer/latest.json
```

After the training run starts, verify:

1. `data/artifacts/training/metrics.jsonl` (epoch and robustness records)
2. `data/artifacts/outbox/epoch_XXXXXXXXX.h5` (checkpoint for each epoch)
3. `data/artifacts/outbox/final_state.h5` (final state)
4. `data/artifacts/metrics/latest.json` (real live snapshot for exporter/Grafana, also refreshed mid-epoch)
5. `data/artifacts/visualizer/latest.json` (real bootstrap/live lattice and per-tick trace for the visualizer/WebSocket)

## Closing dev steps

Steps is closed only after a full training run on real MNIST and the resulting artifacts are recorded.

1. Prepare data and the release build:

```bash
make install
make build-release
ctest --preset release
```

2. Run the baseline training:

```bash
PYTHONPATH=.python-packages:build/release:python python3 python/train.py \
  --config configs/default.yaml \
  --dataset mnist \
  --data-root data \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --ticks 100 \
  --target-accuracy 0.85 \
  --checkpoint-dir data/artifacts/outbox \
  --state-out data/artifacts/outbox/final_state.h5 \
  --metrics-out data/artifacts/training/metrics.jsonl \
  --metrics-snapshot-path data/artifacts/metrics/latest.json \
  --visualizer-trace-path data/artifacts/visualizer/latest.json
```

3. While the run is in progress, observe:
   - `tail -f data/artifacts/training/metrics.jsonl`
   - `ls data/artifacts/outbox/epoch_*.h5`
   - `cat data/artifacts/metrics/latest.json`
   - `cat data/artifacts/visualizer/latest.json`
   - `docker compose logs -f artifact-uploader`
   - `curl -fsS http://localhost:8000/metrics`
   - `curl -fsS http://localhost:8080/lattice`

4. After completion, record the evidence:
   - `data/artifacts/training/metrics.jsonl`
   - `data/artifacts/outbox/final_state.h5`
   - at least one `epoch_XXXXXXXXX.h5`
   - `data/artifacts/metrics/latest.json`
   - `data/artifacts/visualizer/latest.json`

5. Check DoD items:

```bash
python3 docs/acceptance/scripts/check_dod_metrics.py \
  --metrics-path data/artifacts/training/metrics.jsonl \
  --require-dataset mnist \
  --target-accuracy 0.85 \
  --max-active-ratio 0.05 \
  --max-prune-drop 0.05 \
  --max-noise-drop 0.10

python3 docs/acceptance/scripts/check_inference_pipeline.py \
  --state-path data/artifacts/outbox/final_state.h5 \
  --data-root data \
  --dataset mnist
```

Considered closed if:

1. the inference pipeline returns a class in `0..9`
2. `eval_accuracy >= 0.85`
3. `senna_max_active_neurons_ratio <= 0.05`
4. `prune_drop <= 0.05`
5. `noise_drop <= 0.10`
6. epoch checkpoints, `final_state.h5`, `data/artifacts/metrics/latest.json`, and `data/artifacts/visualizer/latest.json` are created without errors

## Closing steps

Closes the operational and quality-gate requirements on top of final steps.

1. Full automated run:

```bash
docs/acceptance/scripts/run_acceptance.sh \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

The script prints `Observation memo` after `make up`, but the training run starts immediately without an interactive pause.

2. Bring up the runtime separately if manual inspection is needed:

```bash
make up
docker compose ps
curl -fsS http://localhost:9000/minio/health/live
curl -fsS http://localhost:3000/api/health
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8000/health
```

3. After the training run, verify the exporter:
   - `cat data/artifacts/metrics/latest.json`
   - `curl -fsS http://localhost:8000/metrics`
   - before the snapshot appears, the exporter must not return synthetic or fabricated metrics

4. After the training run, verify the MinIO/upload path:
   - `docker compose logs -f artifact-uploader`
   - `data/artifacts/outbox/epoch_XXXXXXXXX.h5` and `final_state.h5` must be uploaded into bucket `senna-artifacts`
   - MinIO is not part of MNIST input; the dataset remains local in `data/MNIST/raw`

5. After the training run, verify the visualizer trace:
   - `cat data/artifacts/visualizer/latest.json`
   - `curl -fsS http://localhost:8080/lattice`
   - before the trace appears, the visualizer must not replace lattice geometry or WebSocket frames with synthetic data

6. Verify Grafana:
   - `http://localhost:3000`
   - dashboards `SENNA Training`, `SENNA Activity`, `SENNA Performance`
   - metrics `senna_test_accuracy`, `senna_active_neurons_ratio`, `senna_spikes_per_tick`

7. Verify the visualizer:
   - `http://localhost:8080`
   - `Next Tick` mode
   - heatmap
   - neuron type filters

8. Verify WebSocket sparsity:

```bash
python3 docs/acceptance/scripts/check_ws_sparsity.py \
  --ws-url ws://localhost:8080/ws \
  --max-ratio 0.05
```

9. For DoD item 6, collect manual evidence for interference patterns:
   - screenshot or video from the visualizer for several classes
   - metrics or a correlation report if external analysis is used
   - record that the observed pattern does not collapse into uniform noise

Step is considered closed if:

1. the acceptance orchestration finishes without FAIL
2. the Docker stack comes up with one command, `make up`
3. MinIO and `artifact-uploader` are available, epoch/state artifacts are uploaded into the bucket, and the dataset remains local
4. Grafana, exporter, and the visualizer are available; the exporter does not fabricate metrics with synthetic fallback, and the visualizer does not fabricate lattice geometry or frames with a synthetic trace
5. the WebSocket sparsity check passes
6. all quality gates 13 and 14 are green

## Example with parameters

```bash
docs/acceptance/scripts/run_acceptance.sh \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

## Useful flags

- `--skip-build`
- `--skip-lint`
- `--skip-sanitize`
- `--skip-docker`
- `--skip-training`
- `--skip-ws-sparsity`
- `--dataset mnist`
- `--max-active-ratio`, `--max-prune-drop`, `--max-noise-drop`
- `--metrics-path <path>`
- `--metrics-snapshot-path <path>`
- `--visualizer-trace-path <path>`
- `--state-path <path>`
- `--config <path>`, `--checkpoint-dir <path>`, `--data-root <path>`
- `PYTHON_BIN=python3.12 .../run_acceptance.sh` (if a different Python executable is needed)
- for manual host-side launches, use `PYTHONPATH=.python-packages:build/release:python`

## DoD 13: clang-tidy clean

Command to close it:

```bash
make lint
```

What is checked:

1. `clang-format` for `src/` and `tests/`
2. `scripts/run_clang_tidy.py --build-dir build/debug`
3. `ruff check .`

Important details:

1. `clang-tidy` runs over all `.cpp` files from `build/debug/compile_commands.json`, not only `src/main.cpp`
2. every warning is treated as an error through `--warnings-as-errors=*`
3. the same full-project run is executed in GitHub Actions

## DoD 14: ASan/UBSan clean

Command to close it:

```bash
make build-sanitize
ctest --preset sanitize
```

What is important here:

1. the sanitize run includes C++ GTest/CTest and the Python integration test
2. for the pybind11 module `senna_core`, `CTest` automatically injects `LD_PRELOAD=<libasan.so>`
3. `ASAN_OPTIONS=detect_leaks=0` is intentional: CPython as the host process produces leak noise that is not related to leaks inside `senna_core`
4. `UBSAN_OPTIONS=print_stacktrace=1` is kept for undefined-behavior diagnostics

If `ctest --preset sanitize` is green, step is closed.

## Mapping to DoD items

1. Pipeline `MNIST -> class`: `check_inference_pipeline.py`
2. Accuracy `>85%`: `check_dod_metrics.py` (`eval_accuracy`)
3. Sparsity `<5%`: `check_ws_sparsity.py` + `senna_max_active_neurons_ratio` across all `epoch_end`
4. `remove_neurons(0.1)` loss `<5%`: `check_dod_metrics.py`
5. `inject_noise(0.3)` loss `<10%`: `check_dod_metrics.py`
6. Interference patterns (visual + correlation): manual validation
7. Grafana dashboards: health checks + manual review
8. 3D visualizer: health checks + manual review
9. Docker Compose with one command: `make up`
10. CI green: verify in GitHub Actions
11. Determinism: run training twice with the same seed and compare artifacts
12. HDF5 reproducibility: covered by `make test` (`test_persistence`)
13. clang-tidy without warnings: `make lint`
14. ASan/UBSan clean: `make build-sanitize` + `ctest --preset sanitize`

## Important note for item 6

Quantitative correlation of class patterns (0-9) requires separate export of
voxel activity maps per label. The current runbook automates only the
infrastructure and the baseline DoD gates; item 6 is closed through manual or
additional export-based analysis.
