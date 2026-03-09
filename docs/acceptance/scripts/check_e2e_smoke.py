#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT_DIR / "python"
BUILD_RELEASE_DIR = ROOT_DIR / "build" / "release"

if str(BUILD_RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_RELEASE_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from senna.training import (  # noqa: E402
    TrainingPipeline,
    iter_mnist_samples,
    make_synthetic_digit_samples,
)


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            events.append(payload)

    if not events:
        raise ValueError(f"metrics file is empty: {path}")
    return events


def http_fetch(url: str, *, expect_json: bool) -> dict[str, Any] | str:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json" if expect_json else "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        payload = response.read().decode("utf-8")

    if not expect_json:
        return payload

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return data


def read_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def select_sample(
    *, dataset_mode: str, data_root: str, sample_index: int, synthetic_seed: int
) -> tuple[list[int], int]:
    if dataset_mode == "mnist":
        iterator = iter_mnist_samples(
            root=data_root,
            train=False,
            limit=sample_index + 1,
            download=False,
        )
        for index, sample in enumerate(iterator):
            if index == sample_index:
                return sample.image, sample.label
        raise RuntimeError("MNIST sample index is out of range")

    samples = make_synthetic_digit_samples(sample_index + 1, seed=synthetic_seed)
    sample = samples[sample_index]
    return sample.image, sample.label


def run_inference_check(
    *,
    state_path: Path,
    config_path: str,
    dataset_mode: str,
    data_root: str,
    sample_index: int,
    ticks: int,
    synthetic_seed: int,
) -> dict[str, Any]:
    image, label = select_sample(
        dataset_mode=dataset_mode,
        data_root=data_root,
        sample_index=sample_index,
        synthetic_seed=synthetic_seed,
    )

    pipeline = TrainingPipeline.load_state(str(state_path), config_path=config_path)
    pipeline.handle.set_eval_mode(True)
    pipeline.handle.load_sample(image, label, False)
    pipeline.handle.step(ticks)
    prediction = int(pipeline.handle.get_prediction())
    metrics = dict(pipeline.handle.get_metrics())

    if prediction < -1 or prediction > 9:
        raise RuntimeError(f"prediction is outside class range [-1..9]: {prediction}")

    return {
        "label": label,
        "prediction": prediction,
        "prediction_ready": prediction >= 0,
        "ticks": ticks,
        "senna_active_neurons_ratio": float(
            metrics.get("senna_active_neurons_ratio", 0.0)
        ),
        "senna_spikes_per_tick": float(metrics.get("senna_spikes_per_tick", 0.0)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the lightweight deployment-to-training E2E smoke path."
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "synthetic"),
        default="synthetic",
        help="Dataset mode used for the smoke run",
    )
    parser.add_argument("--data-root", default="data", help="Dataset root")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index for state-load inference validation",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Synthetic seed for state-load inference validation",
    )
    parser.add_argument(
        "--expected-epochs", type=int, required=True, help="Expected epoch count"
    )
    parser.add_argument(
        "--expected-train-limit",
        type=int,
        required=True,
        help="Expected train_limit value",
    )
    parser.add_argument(
        "--expected-test-limit",
        type=int,
        required=True,
        help="Expected test_limit value",
    )
    parser.add_argument(
        "--expected-ticks", type=int, required=True, help="Expected ticks value"
    )
    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to metrics JSONL produced by python/train.py",
    )
    parser.add_argument(
        "--state-path",
        required=True,
        help="Path to final_state.h5 produced by the smoke run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory with epoch_*.h5 checkpoints",
    )
    parser.add_argument(
        "--metrics-snapshot-path",
        default="data/artifacts/metrics/latest.json",
        help="Path to exporter snapshot JSON",
    )
    parser.add_argument(
        "--visualizer-trace-path",
        default="data/artifacts/visualizer/latest.json",
        help="Path to visualizer trace JSON",
    )
    parser.add_argument(
        "--uploader-state-path",
        default="data/artifacts/uploader_state.json",
        help="Path to uploader state JSON",
    )
    parser.add_argument(
        "--uploaded-before",
        type=int,
        default=0,
        help="Uploaded signature count captured before the smoke run",
    )
    parser.add_argument(
        "--expected-uploaded-delta",
        type=int,
        default=0,
        help="Minimum expected uploaded signature growth after the smoke run",
    )
    parser.add_argument(
        "--simulator-health-url",
        default="http://localhost:8000/health",
        help="Simulator health URL",
    )
    parser.add_argument(
        "--simulator-metrics-url",
        default="http://localhost:8000/metrics",
        help="Simulator metrics URL",
    )
    parser.add_argument(
        "--visualizer-health-url",
        default="http://localhost:8080/health",
        help="Visualizer health URL",
    )
    parser.add_argument(
        "--visualizer-lattice-url",
        default="http://localhost:8080/lattice",
        help="Visualizer lattice URL",
    )
    parser.add_argument(
        "--prometheus-health-url",
        default="http://localhost:9090/-/healthy",
        help="Prometheus health URL",
    )
    parser.add_argument(
        "--grafana-health-url",
        default="http://localhost:3000/api/health",
        help="Grafana health URL",
    )
    parser.add_argument(
        "--minio-health-url",
        default="http://localhost:9000/minio/health/live",
        help="MinIO health URL",
    )
    parser.add_argument(
        "--verdict-out",
        default="",
        help="Optional path to JSON summary output",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    state_path = Path(args.state_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_snapshot_path = Path(args.metrics_snapshot_path)
    visualizer_trace_path = Path(args.visualizer_trace_path)
    uploader_state_path = Path(args.uploader_state_path)

    failures: list[str] = []

    try:
        events = load_jsonl(metrics_path)
    except (FileNotFoundError, ValueError) as exc:
        return fail(str(exc))

    epoch_events = [event for event in events if event.get("event") == "epoch_end"]
    if len(epoch_events) != args.expected_epochs:
        failures.append(
            f"epoch_end count {len(epoch_events)} != expected {args.expected_epochs}"
        )

    last_epoch = epoch_events[-1] if epoch_events else {}
    for index, event in enumerate(epoch_events, start=1):
        if event.get("dataset_mode") != args.dataset:
            failures.append(
                f"epoch_end #{index} dataset_mode={event.get('dataset_mode')!r} "
                f"!= expected {args.dataset!r}"
            )
        if event.get("train_limit") != args.expected_train_limit:
            failures.append(
                f"epoch_end #{index} train_limit={event.get('train_limit')!r} "
                f"!= expected {args.expected_train_limit}"
            )
        if event.get("test_limit") != args.expected_test_limit:
            failures.append(
                f"epoch_end #{index} test_limit={event.get('test_limit')!r} "
                f"!= expected {args.expected_test_limit}"
            )
        if event.get("ticks") != args.expected_ticks:
            failures.append(
                f"epoch_end #{index} ticks={event.get('ticks')!r} "
                f"!= expected {args.expected_ticks}"
            )

    checkpoint_files = sorted(checkpoint_dir.glob("epoch_*.h5"))
    if len(checkpoint_files) != args.expected_epochs:
        failures.append(
            f"checkpoint count {len(checkpoint_files)} != expected {args.expected_epochs}"
        )
    if not state_path.exists():
        failures.append(f"state file not found: {state_path}")

    if epoch_events:
        expected_checkpoint = str(checkpoint_files[-1]) if checkpoint_files else ""
        if last_epoch.get("checkpoint_path") != expected_checkpoint:
            failures.append(
                "last epoch checkpoint_path does not match the last checkpoint file"
            )

    try:
        metrics_snapshot = load_json(metrics_snapshot_path)
    except (FileNotFoundError, ValueError) as exc:
        failures.append(str(exc))
        metrics_snapshot = {}

    try:
        visualizer_trace = load_json(visualizer_trace_path)
    except (FileNotFoundError, ValueError) as exc:
        failures.append(str(exc))
        visualizer_trace = {}

    if metrics_snapshot:
        if metrics_snapshot.get("dataset_mode") != args.dataset:
            failures.append(
                f"metrics snapshot dataset_mode={metrics_snapshot.get('dataset_mode')!r} "
                f"!= expected {args.dataset!r}"
            )
        if int(metrics_snapshot.get("epoch", 0)) != args.expected_epochs:
            failures.append(
                f"metrics snapshot epoch={metrics_snapshot.get('epoch')!r} "
                f"!= expected {args.expected_epochs}"
            )

    trace_frames = visualizer_trace.get("frames")
    trace_lattice = visualizer_trace.get("lattice")
    if not isinstance(trace_frames, list) or not trace_frames:
        failures.append("visualizer trace must contain a non-empty frames list")
    if (
        not isinstance(trace_lattice, dict)
        or int(trace_lattice.get("neuronCount", 0)) <= 0
    ):
        failures.append("visualizer trace must contain lattice.neuronCount > 0")

    uploader_count = args.uploaded_before
    uploaded_delta = 0
    if uploader_state_path.exists():
        try:
            uploader_state = load_json(uploader_state_path)
            signatures = uploader_state.get("uploaded_signatures", [])
            if isinstance(signatures, list):
                uploader_count = len(signatures)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))
    uploaded_delta = uploader_count - args.uploaded_before
    if uploaded_delta < args.expected_uploaded_delta:
        failures.append(
            f"uploaded signature delta {uploaded_delta} < expected {args.expected_uploaded_delta}"
        )

    endpoint_summary: dict[str, Any] = {}
    try:
        simulator_health = http_fetch(args.simulator_health_url, expect_json=True)
        endpoint_summary["simulator_health"] = simulator_health
        if not bool(simulator_health.get("snapshot_ready")):
            failures.append("simulator health reports snapshot_ready=false")
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read simulator health: {exc}")

    try:
        simulator_metrics = str(
            http_fetch(args.simulator_metrics_url, expect_json=False)
        )
        endpoint_summary["simulator_metrics_ready"] = True
        for metric_name in (
            "senna_test_accuracy",
            "senna_active_neurons_ratio",
            "senna_spikes_per_tick",
        ):
            if metric_name not in simulator_metrics:
                failures.append(f"simulator metrics payload is missing {metric_name}")
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read simulator metrics: {exc}")

    try:
        visualizer_health = http_fetch(args.visualizer_health_url, expect_json=True)
        endpoint_summary["visualizer_health"] = visualizer_health
        if not bool(visualizer_health.get("traceReady")):
            failures.append("visualizer health reports traceReady=false")
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read visualizer health: {exc}")

    try:
        lattice_payload = http_fetch(args.visualizer_lattice_url, expect_json=True)
        endpoint_summary["visualizer_lattice_neuron_count"] = int(
            lattice_payload.get("neuronCount", 0)
        )
        if endpoint_summary["visualizer_lattice_neuron_count"] <= 0:
            failures.append("visualizer lattice neuronCount must be > 0")
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read visualizer lattice: {exc}")

    try:
        http_fetch(args.prometheus_health_url, expect_json=False)
        endpoint_summary["prometheus_health"] = "ok"
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read prometheus health: {exc}")

    try:
        grafana_health = http_fetch(args.grafana_health_url, expect_json=True)
        endpoint_summary["grafana_health"] = grafana_health
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read grafana health: {exc}")

    try:
        http_fetch(args.minio_health_url, expect_json=False)
        endpoint_summary["minio_health"] = "ok"
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        failures.append(f"cannot read minio health: {exc}")

    try:
        inference = run_inference_check(
            state_path=state_path,
            config_path=args.config,
            dataset_mode=args.dataset,
            data_root=args.data_root,
            sample_index=args.sample_index,
            ticks=args.expected_ticks,
            synthetic_seed=args.synthetic_seed,
        )
    except Exception as exc:  # pragma: no cover - integration boundary
        failures.append(f"inference validation failed: {exc}")
        inference = {}

    train_metrics = last_epoch.get("train", {}) if isinstance(last_epoch, dict) else {}
    eval_metrics = last_epoch.get("eval", {}) if isinstance(last_epoch, dict) else {}

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "dataset_mode": args.dataset,
        "verdict": "PASS" if not failures else "FAIL",
        "epochs": {
            "expected": args.expected_epochs,
            "observed": len(epoch_events),
        },
        "artifacts": {
            "metrics_path": str(metrics_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_count": len(checkpoint_files),
            "state_path": str(state_path),
            "metrics_snapshot_path": str(metrics_snapshot_path),
            "visualizer_trace_path": str(visualizer_trace_path),
        },
        "uploads": {
            "uploaded_before": args.uploaded_before,
            "uploaded_after": uploader_count,
            "uploaded_delta": uploaded_delta,
            "expected_delta": args.expected_uploaded_delta,
        },
        "metrics": {
            "train_accuracy": read_float(train_metrics, "epoch_accuracy"),
            "eval_accuracy": read_float(eval_metrics, "eval_accuracy"),
            "spikes_per_tick": read_float(eval_metrics, "senna_spikes_per_tick"),
            "active_neurons_ratio": read_float(
                eval_metrics, "senna_active_neurons_ratio"
            ),
            "synapse_count": read_float(eval_metrics, "senna_synapse_count"),
        },
        "trace": {
            "frame_count": len(trace_frames) if isinstance(trace_frames, list) else 0,
            "neuron_count": int(trace_lattice.get("neuronCount", 0))
            if isinstance(trace_lattice, dict)
            else 0,
        },
        "inference": inference,
        "endpoints": endpoint_summary,
        "failures": failures,
    }

    if args.verdict_out:
        write_json(Path(args.verdict_out), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print("[FAIL] E2E smoke path failed.")
        return 1

    print("[PASS] E2E smoke path passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
