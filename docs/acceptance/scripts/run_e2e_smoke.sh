#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/docs/acceptance/scripts"
RUN_TAG="e2e_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_PYTHONPATH=".python-packages:build/release:python${PYTHONPATH:+:${PYTHONPATH}}"

DATASET="synthetic"
DATA_ROOT="data"
CONFIG_PATH="configs/default.yaml"
EPOCHS=5
TRAIN_LIMIT=8
TEST_LIMIT=4
TICKS=16
PROGRESS_EVERY=2
LIVE_TRACE_EVERY=4
WAIT_UPLOAD_TIMEOUT_SEC=90
KEEP_UP=false
SKIP_BUILD=false
SKIP_DOCKER_BUILD=false
SKIP_DOCKER=false

METRICS_SNAPSHOT_PATH="data/artifacts/metrics/latest.json"
VISUALIZER_TRACE_PATH="data/artifacts/visualizer/latest.json"
UPLOADER_STATE_PATH="data/artifacts/uploader_state.json"
RUN_DIR=""
CHECKPOINT_DIR=""
STATE_PATH=""
METRICS_PATH=""
TRAIN_LOG_PATH=""
VERDICT_PATH=""
UPLOADER_LOG_PATH=""

SIMULATOR_HEALTH_URL="http://localhost:8000/health"
SIMULATOR_METRICS_URL="http://localhost:8000/metrics"
VISUALIZER_HEALTH_URL="http://localhost:8080/health"
VISUALIZER_LATTICE_URL="http://localhost:8080/lattice"
PROMETHEUS_HEALTH_URL="http://localhost:9090/-/healthy"
GRAFANA_HEALTH_URL="http://localhost:3000/api/health"
MINIO_HEALTH_URL="http://localhost:9000/minio/health/live"

print_usage() {
  cat <<'EOF'
Usage: docs/acceptance/scripts/run_e2e_smoke.sh [options]

Options:
  --run-tag <name>
  --dataset <synthetic|mnist>
  --data-root <path>
  --config <path>
  --epochs <int>
  --train-limit <int>
  --test-limit <int>
  --ticks <int>
  --progress-every <int>
  --live-trace-every <int>
  --wait-upload-timeout-sec <int>
  --keep-up
  --skip-build
  --skip-docker-build
  --skip-docker
  --help
EOF
}

log() {
  printf '\n[%s] %s\n' "e2e-smoke" "$*"
}

run_cmd() {
  printf '+ %s\n' "$*"
  "$@"
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

uploaded_count() {
  local path="$1"
  "${PYTHON_BIN}" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit(0)

signatures = payload.get("uploaded_signatures", [])
print(len(signatures) if isinstance(signatures, list) else 0)
PY
}

wait_for_uploaded_delta() {
  local before="$1"
  local expected_delta="$2"
  local timeout_sec="$3"
  local started_at
  started_at="$(date +%s)"

  while true; do
    local current_count
    current_count="$(uploaded_count "${UPLOADER_STATE_PATH}")"
    if (( current_count - before >= expected_delta )); then
      echo "[PASS] uploader delta reached: before=${before} after=${current_count} delta=$((current_count - before))"
      return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - started_at >= timeout_sec )); then
      echo "[FAIL] uploader delta timeout: before=${before} after=${current_count} expected_delta=${expected_delta}"
      return 1
    fi

    sleep 2
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-tag)
      RUN_TAG="$2"
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
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --live-trace-every)
      LIVE_TRACE_EVERY="$2"
      shift 2
      ;;
    --wait-upload-timeout-sec)
      WAIT_UPLOAD_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --keep-up)
      KEEP_UP=true
      shift
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-docker-build)
      SKIP_DOCKER_BUILD=true
      shift
      ;;
    --skip-docker)
      SKIP_DOCKER=true
      shift
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

RUN_DIR="data/artifacts/e2e-smoke/${RUN_TAG}"
CHECKPOINT_DIR="data/artifacts/outbox/${RUN_TAG}"
STATE_PATH="${CHECKPOINT_DIR}/final_state.h5"
METRICS_PATH="${RUN_DIR}/metrics.jsonl"
TRAIN_LOG_PATH="${RUN_DIR}/train.log"
VERDICT_PATH="${RUN_DIR}/verdict.json"
UPLOADER_LOG_PATH="${RUN_DIR}/artifact_uploader.log"
EXPECTED_UPLOADED_DELTA=$((EPOCHS + 1))

cleanup() {
  if [[ "${KEEP_UP}" == "false" && "${SKIP_DOCKER}" == "false" ]]; then
    log "Stopping docker compose stack"
    run_cmd make down
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}"
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

UPLOADED_BEFORE="$(uploaded_count "${UPLOADER_STATE_PATH}")"

if [[ "${SKIP_BUILD}" == "false" ]]; then
  log "Building release runtime"
  run_cmd make build-release
fi

if [[ "${SKIP_DOCKER}" == "false" ]]; then
  log "Bringing up runtime stack"
  if [[ "${SKIP_DOCKER_BUILD}" == "false" ]]; then
    run_cmd make up-build
  else
    run_cmd make up
  fi

  wait_http_ok "simulator health" "${SIMULATOR_HEALTH_URL}"
  wait_http_ok "visualizer health" "${VISUALIZER_HEALTH_URL}"
  wait_http_ok "prometheus health" "${PROMETHEUS_HEALTH_URL}"
  wait_http_ok "grafana health" "${GRAFANA_HEALTH_URL}"
  wait_http_ok "minio health" "${MINIO_HEALTH_URL}"
fi

log "Running small end-to-end training pass"
run_cmd env PYTHONPATH="${RUNTIME_PYTHONPATH}" "${PYTHON_BIN}" python/train.py \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --train-limit "${TRAIN_LIMIT}" \
  --test-limit "${TEST_LIMIT}" \
  --ticks "${TICKS}" \
  --no-early-stop \
  --skip-robustness \
  --progress-every "${PROGRESS_EVERY}" \
  --live-trace-every "${LIVE_TRACE_EVERY}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --state-out "${STATE_PATH}" \
  --metrics-out "${METRICS_PATH}" \
  --metrics-snapshot-path "${METRICS_SNAPSHOT_PATH}" \
  --visualizer-trace-path "${VISUALIZER_TRACE_PATH}" | tee "${TRAIN_LOG_PATH}"

log "Waiting for uploader to persist new HDF5 artifacts"
wait_for_uploaded_delta "${UPLOADED_BEFORE}" "${EXPECTED_UPLOADED_DELTA}" "${WAIT_UPLOAD_TIMEOUT_SEC}"

if [[ "${SKIP_DOCKER}" == "false" ]]; then
  log "Checking uploader logs for this run tag"
  run_cmd docker compose logs --no-color artifact-uploader > "${UPLOADER_LOG_PATH}"
  if ! grep -q "${RUN_TAG}" "${UPLOADER_LOG_PATH}"; then
    echo "[FAIL] artifact-uploader logs do not mention ${RUN_TAG}"
    exit 1
  fi
fi

log "Collecting final verdict"
run_cmd env PYTHONPATH="${RUNTIME_PYTHONPATH}" "${PYTHON_BIN}" "${SCRIPTS_DIR}/check_e2e_smoke.py" \
  --run-id "${RUN_TAG}" \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --expected-epochs "${EPOCHS}" \
  --expected-train-limit "${TRAIN_LIMIT}" \
  --expected-test-limit "${TEST_LIMIT}" \
  --expected-ticks "${TICKS}" \
  --metrics-path "${METRICS_PATH}" \
  --state-path "${STATE_PATH}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --metrics-snapshot-path "${METRICS_SNAPSHOT_PATH}" \
  --visualizer-trace-path "${VISUALIZER_TRACE_PATH}" \
  --uploader-state-path "${UPLOADER_STATE_PATH}" \
  --uploaded-before "${UPLOADED_BEFORE}" \
  --expected-uploaded-delta "${EXPECTED_UPLOADED_DELTA}" \
  --simulator-health-url "${SIMULATOR_HEALTH_URL}" \
  --simulator-metrics-url "${SIMULATOR_METRICS_URL}" \
  --visualizer-health-url "${VISUALIZER_HEALTH_URL}" \
  --visualizer-lattice-url "${VISUALIZER_LATTICE_URL}" \
  --prometheus-health-url "${PROMETHEUS_HEALTH_URL}" \
  --grafana-health-url "${GRAFANA_HEALTH_URL}" \
  --minio-health-url "${MINIO_HEALTH_URL}" \
  --verdict-out "${VERDICT_PATH}"

log "Smoke artifacts"
printf '%s\n' \
  "run_dir=${RUN_DIR}" \
  "checkpoint_dir=${CHECKPOINT_DIR}" \
  "metrics_path=${METRICS_PATH}" \
  "state_path=${STATE_PATH}" \
  "verdict_path=${VERDICT_PATH}" \
  "train_log_path=${TRAIN_LOG_PATH}" \
  "uploader_log_path=${UPLOADER_LOG_PATH}"
