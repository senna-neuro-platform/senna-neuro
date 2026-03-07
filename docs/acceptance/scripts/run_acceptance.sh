#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/docs/acceptance/scripts"

EPOCHS=5
TRAIN_LIMIT=60000
TEST_LIMIT=10000
TICKS=100
TARGET_ACCURACY=0.85
MAX_ACTIVE_RATIO=0.05
MAX_PRUNE_DROP=0.05
MAX_NOISE_DROP=0.10
DATASET="mnist"
DATA_ROOT="data"
CONFIG_PATH="configs/default.yaml"
CHECKPOINT_DIR="data/artifacts/outbox"
STATE_PATH="data/artifacts/outbox/final_state.h5"
METRICS_PATH="data/artifacts/training/metrics.jsonl"
METRICS_SNAPSHOT_PATH="data/artifacts/metrics/latest.json"
VISUALIZER_TRACE_PATH="data/artifacts/visualizer/latest.json"
WS_URL="ws://localhost:8080/ws"
LATTICE_URL="http://localhost:8080/lattice"
SIMULATOR_HEALTH_URL="http://localhost:8000/health"
VISUALIZER_HEALTH_URL="http://localhost:8080/health"
PROMETHEUS_HEALTH_URL="http://localhost:9090/-/healthy"
GRAFANA_HEALTH_URL="http://localhost:3000/api/health"
MINIO_HEALTH_URL="http://localhost:9000/minio/health/live"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_PYTHONPATH="build/release:python"

SKIP_BUILD=false
SKIP_LINT=false
SKIP_SANITIZE=false
SKIP_DOCKER=false
SKIP_TRAINING=false
SKIP_WS_SPARSITY=false

print_usage() {
  cat <<'EOF'
Usage: docs/acceptance/scripts/run_acceptance.sh [options]

Options:
  --epochs <int>
  --train-limit <int>
  --test-limit <int>
  --ticks <int>
  --target-accuracy <float>
  --max-active-ratio <float>
  --max-prune-drop <float>
  --max-noise-drop <float>
  --dataset <mnist>
  --data-root <path>
  --config <path>
  --checkpoint-dir <path>
  --state-path <path>
  --metrics-path <path>
  --metrics-snapshot-path <path>
  --visualizer-trace-path <path>
  --ws-url <url>
  --lattice-url <url>
  --skip-build
  --skip-lint
  --skip-sanitize
  --skip-docker
  --skip-training
  --skip-ws-sparsity
  --help
EOF
}

log() {
  printf '\n[%s] %s\n' "" "$*"
}

run_cmd() {
  printf '+ %s\n' "$*"
  "$@"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[FAIL] missing required command: $1"
    exit 1
  fi
}

require_python_module() {
  local module_expr="$1"
  local install_hint="$2"

  if ! "${PYTHON_BIN}" -c "${module_expr}" >/dev/null 2>&1; then
    echo "[FAIL] missing required Python modules in current host env"
    echo "[INFO] install with: ${install_hint}"
    echo "[INFO] real MNIST for acceptance is read locally from ${DATA_ROOT}/MNIST/raw by host Python; MinIO is used only for artifact upload from data/artifacts/outbox"
    exit 1
  fi
}

require_mnist_raw_files() {
  local data_root="$1"
  local raw_dir="${data_root%/}/MNIST/raw"
  local missing=()
  local file=""

  for file in \
    train-images-idx3-ubyte \
    train-labels-idx1-ubyte \
    t10k-images-idx3-ubyte \
    t10k-labels-idx1-ubyte; do
    if [[ ! -f "${raw_dir}/${file}" ]]; then
      missing+=("${file}")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[FAIL] missing MNIST raw files under ${raw_dir}"
    printf '[INFO] missing files: %s\n' "${missing[*]}"
    echo "[INFO] run 'make install' to download them locally; MinIO is not used for MNIST input"
    exit 1
  fi
}

wait_http_ok() {
  local name="$1"
  local url="$2"
  local attempts=60
  local sleep_sec=2

  for ((i = 1; i <= attempts; i += 1)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[PASS] ${name}: ${url}"
      return 0
    fi
    sleep "${sleep_sec}"
  done

  echo "[FAIL] ${name} is not reachable: ${url}"
  return 1
}

print_observe_stack_memo() {
  cat <<EOF
[OBSERVE] Docker status: docker compose ps
[OBSERVE] Docker logs: make logs
[OBSERVE] Grafana: http://localhost:3000 (admin/admin), dashboards: SENNA Activity / SENNA Training / SENNA Performance
[OBSERVE] Visualizer: http://localhost:8080 (WS: ${WS_URL})
[OBSERVE] Exporter health: http://localhost:8000/health
[OBSERVE] MinIO live: ${MINIO_HEALTH_URL} and console: http://localhost:9001
[OBSERVE] MinIO receives epoch/state artifacts from data/artifacts/outbox; MNIST input stays local in ${DATA_ROOT}/MNIST/raw
[OBSERVE] Exporter metrics become available after current training run writes ${METRICS_SNAPSHOT_PATH}
EOF
}

print_observe_training_memo() {
  cat <<'EOF'
[OBSERVE] Training metrics tail: tail -f data/artifacts/training/metrics.jsonl
[OBSERVE] Epoch checkpoints: ls -1 data/artifacts/outbox/epoch_*.h5 | tail
[OBSERVE] Exporter snapshot: cat data/artifacts/metrics/latest.json
[OBSERVE] Visualizer trace: cat data/artifacts/visualizer/latest.json
[OBSERVE] MinIO uploader logs: docker compose logs -f artifact-uploader
[OBSERVE] Prometheus probe: curl -fsS http://localhost:8000/metrics | rg 'senna_(train|test|active|spikes)'
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --train-limit)
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --test-limit)
      TEST_LIMIT="$2"
      shift 2
      ;;
    --ticks)
      TICKS="$2"
      shift 2
      ;;
    --target-accuracy)
      TARGET_ACCURACY="$2"
      shift 2
      ;;
    --max-active-ratio)
      MAX_ACTIVE_RATIO="$2"
      shift 2
      ;;
    --max-prune-drop)
      MAX_PRUNE_DROP="$2"
      shift 2
      ;;
    --max-noise-drop)
      MAX_NOISE_DROP="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --state-path)
      STATE_PATH="$2"
      shift 2
      ;;
    --metrics-path)
      METRICS_PATH="$2"
      shift 2
      ;;
    --metrics-snapshot-path)
      METRICS_SNAPSHOT_PATH="$2"
      shift 2
      ;;
    --visualizer-trace-path)
      VISUALIZER_TRACE_PATH="$2"
      shift 2
      ;;
    --ws-url)
      WS_URL="$2"
      shift 2
      ;;
    --lattice-url)
      LATTICE_URL="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-lint)
      SKIP_LINT=true
      shift
      ;;
    --skip-sanitize)
      SKIP_SANITIZE=true
      shift
      ;;
    --skip-docker)
      SKIP_DOCKER=true
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    --skip-ws-sparsity)
      SKIP_WS_SPARSITY=true
      shift
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "[FAIL] unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

require_cmd make
require_cmd cmake
require_cmd ctest
require_cmd curl
require_cmd "${PYTHON_BIN}"

if [[ "${DATASET}" != "mnist" ]]; then
  echo "[FAIL] acceptance run requires --dataset mnist"
  exit 1
fi

cd "${ROOT_DIR}"
log "root=${ROOT_DIR}"

if [[ "${SKIP_TRAINING}" == false ]]; then
  require_python_module \
    "import torch, torchvision" \
    "${PYTHON_BIN} -m pip install torch torchvision"
fi

mkdir -p \
  "$(dirname "${STATE_PATH}")" \
  "$(dirname "${METRICS_PATH}")" \
  "$(dirname "${METRICS_SNAPSHOT_PATH}")" \
  "$(dirname "${VISUALIZER_TRACE_PATH}")" \
  "${CHECKPOINT_DIR}"

if [[ "${SKIP_TRAINING}" == false ]]; then
  rm -f "${METRICS_PATH}" "${METRICS_SNAPSHOT_PATH}" "${VISUALIZER_TRACE_PATH}"
fi

if [[ "${SKIP_BUILD}" == false ]]; then
  log "Build + test gates"
  run_cmd make install
  run_cmd make build-release
  run_cmd ctest --preset release
fi

if [[ "${SKIP_LINT}" == false ]]; then
  log "Lint gate"
  run_cmd make lint
fi

if [[ "${SKIP_SANITIZE}" == false ]]; then
  log "Sanitize gate"
  run_cmd make build-sanitize
  run_cmd ctest --preset sanitize
fi

if [[ "${SKIP_TRAINING}" == false ]]; then
  require_mnist_raw_files "${DATA_ROOT}"
fi

if [[ "${SKIP_DOCKER}" == false ]]; then
  log "Docker stack bring-up + health checks"
  run_cmd make up
  wait_http_ok "minio health" "${MINIO_HEALTH_URL}"
  wait_http_ok "simulator health" "${SIMULATOR_HEALTH_URL}"
  wait_http_ok "visualizer health" "${VISUALIZER_HEALTH_URL}"
  wait_http_ok "prometheus health" "${PROMETHEUS_HEALTH_URL}"
  wait_http_ok "grafana health" "${GRAFANA_HEALTH_URL}"
  log "Observation memo (Docker/Grafana/Visualizer)"
  print_observe_stack_memo
fi
if [[ "${SKIP_TRAINING}" == false ]]; then
  log "Training run (Step 15/16 prerequisites)"
  run_cmd env \
    PYTHONPATH="${RUNTIME_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" python/train.py \
    --config "${CONFIG_PATH}" \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --epochs "${EPOCHS}" \
    --train-limit "${TRAIN_LIMIT}" \
    --test-limit "${TEST_LIMIT}" \
    --ticks "${TICKS}" \
    --target-accuracy "${TARGET_ACCURACY}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --state-out "${STATE_PATH}" \
    --metrics-out "${METRICS_PATH}" \
    --metrics-snapshot-path "${METRICS_SNAPSHOT_PATH}" \
    --visualizer-trace-path "${VISUALIZER_TRACE_PATH}"
  if [[ "${SKIP_DOCKER}" == false ]]; then
    wait_http_ok "exporter metrics" "http://localhost:8000/metrics"
    wait_http_ok "visualizer lattice" "${LATTICE_URL}"
  fi
  log "Observation memo (training telemetry)"
  print_observe_training_memo
fi

log "DoD numeric gates from metrics"
run_cmd "${PYTHON_BIN}" "${SCRIPTS_DIR}/check_dod_metrics.py" \
  --metrics-path "${METRICS_PATH}" \
  --target-accuracy "${TARGET_ACCURACY}" \
  --max-active-ratio "${MAX_ACTIVE_RATIO}" \
  --max-prune-drop "${MAX_PRUNE_DROP}" \
  --max-noise-drop "${MAX_NOISE_DROP}"

log "Inference pipeline check (state -> prediction)"
run_cmd env \
  PYTHONPATH="${RUNTIME_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPTS_DIR}/check_inference_pipeline.py" \
  --state-path "${STATE_PATH}" \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --metrics-path "${METRICS_PATH}" \
  --data-root "${DATA_ROOT}" \
  --ticks "${TICKS}"

if [[ "${SKIP_WS_SPARSITY}" == false ]]; then
  log "WebSocket sparsity gate"
  run_cmd "${PYTHON_BIN}" "${SCRIPTS_DIR}/check_ws_sparsity.py" \
    --ws-url "${WS_URL}" \
    --lattice-url "${LATTICE_URL}" \
    --frames 40 \
    --max-ratio "${MAX_ACTIVE_RATIO}"
fi

log "Automatic acceptance checks finished: PASS."
cat <<'EOF'
Manual DoD checks that remain:
1. Interference patterns by class: visual distinction + correlation gap (intra vs inter).
2. Grafana dashboards: open and verify activity/training/performance panels.
3. Visualizer UX: lattice + waves + heatmap controls.
4. CI status: verify latest GitHub Actions run is green.
EOF
