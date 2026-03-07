# SENNA Neuro

Spatial-Event Neuromorphic Network Architecture.

## Dev commands

Make shortcuts:

```bash
make install
make lint
make build-debug
make build-release
make build-sanityze
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
- MinIO API: http://localhost:9000
- MinIO Console: http://localhost:9001

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
