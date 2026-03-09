from __future__ import annotations

import json
import math
import os
import resource
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))
METRICS_SNAPSHOT_PATH = Path(
    os.getenv("METRICS_SNAPSHOT_PATH", "/artifacts/metrics/latest.json")
)


@dataclass(frozen=True)
class MetricsSnapshot:
    active_neurons_ratio: float
    spikes_per_tick: float
    ei_balance: float
    train_accuracy: float
    test_accuracy: float
    synapse_count: float
    pruned_total: float
    sprouted_total: float
    stdp_updates_total: float
    tick_duration_seconds: float
    e_rate_hz: float
    i_rate_hz: float
    training_progress_ratio: float
    training_samples_per_sec: float
    training_eta_seconds: float


class TickDurationHistogram:
    def __init__(self) -> None:
        self._buckets = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, math.inf]
        self._counts = [0 for _ in self._buckets]
        self._sum = 0.0
        self._total = 0

    def observe(self, value: float) -> None:
        clean_value = max(0.0, float(value))
        self._sum += clean_value
        self._total += 1
        for index, upper in enumerate(self._buckets):
            if clean_value <= upper:
                self._counts[index] += 1

    def render(self, metric_name: str) -> list[str]:
        lines: list[str] = []
        for upper, count in zip(self._buckets, self._counts):
            le = "+Inf" if math.isinf(upper) else f"{upper:.6g}"
            lines.append(f'{metric_name}_bucket{{le="{le}"}} {count}')

        lines.append(f"{metric_name}_sum {self._sum:.9f}")
        lines.append(f"{metric_name}_count {self._total}")
        return lines


class ExporterState:
    def __init__(self) -> None:
        self.histogram = TickDurationHistogram()
        self.started_at = time.time()

    def observe(self, snapshot: MetricsSnapshot) -> None:
        self.histogram.observe(snapshot.tick_duration_seconds)


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def _read_float(payload: dict[str, Any], default: float, *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return default


def load_snapshot_from_file(path: Path) -> MetricsSnapshot | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    active_ratio = _read_float(
        payload,
        0.0,
        "active_neurons_ratio",
        "senna_active_neurons_ratio",
    )
    spikes_per_tick = _read_float(
        payload, 0.0, "spikes_per_tick", "senna_spikes_per_tick"
    )
    e_rate_hz = _read_float(payload, 0.0, "e_rate_hz", "senna_e_rate_hz")
    i_rate_hz = _read_float(payload, 0.0, "i_rate_hz", "senna_i_rate_hz")
    ei_balance = _read_float(payload, 0.0, "ei_balance", "senna_ei_balance")
    if ei_balance <= 0.0 and i_rate_hz > 0.0:
        ei_balance = e_rate_hz / i_rate_hz

    return MetricsSnapshot(
        active_neurons_ratio=max(0.0, active_ratio),
        spikes_per_tick=max(0.0, spikes_per_tick),
        ei_balance=max(0.0, ei_balance),
        train_accuracy=_clamp_0_1(
            _read_float(payload, 0.0, "train_accuracy", "senna_train_accuracy")
        ),
        test_accuracy=_clamp_0_1(
            _read_float(payload, 0.0, "test_accuracy", "senna_test_accuracy")
        ),
        synapse_count=max(
            0.0, _read_float(payload, 0.0, "synapse_count", "senna_synapse_count")
        ),
        pruned_total=max(
            0.0, _read_float(payload, 0.0, "pruned_total", "senna_pruned_total")
        ),
        sprouted_total=max(
            0.0, _read_float(payload, 0.0, "sprouted_total", "senna_sprouted_total")
        ),
        stdp_updates_total=max(
            0.0,
            _read_float(payload, 0.0, "stdp_updates_total", "senna_stdp_updates_total"),
        ),
        tick_duration_seconds=max(
            0.0,
            _read_float(
                payload,
                0.0,
                "tick_duration_seconds",
                "senna_tick_duration_seconds",
            ),
        ),
        e_rate_hz=max(0.0, e_rate_hz),
        i_rate_hz=max(0.0, i_rate_hz),
        training_progress_ratio=max(
            0.0,
            min(
                1.0,
                _read_float(
                    payload,
                    0.0,
                    "training_progress_ratio",
                    "senna_training_progress_ratio",
                ),
            ),
        ),
        training_samples_per_sec=max(
            0.0,
            _read_float(
                payload,
                0.0,
                "training_samples_per_sec",
                "senna_training_samples_per_sec",
            ),
        ),
        training_eta_seconds=max(
            0.0,
            _read_float(
                payload, 0.0, "training_eta_seconds", "senna_training_eta_seconds"
            ),
        ),
    )


def read_snapshot(snapshot_path: Path | None = None) -> MetricsSnapshot | None:
    path = snapshot_path if snapshot_path is not None else METRICS_SNAPSHOT_PATH
    return load_snapshot_from_file(path)


def _format_metric(name: str, value: float) -> str:
    return f"{name} {float(value):.9f}"


def render_metrics_payload(snapshot: MetricsSnapshot, state: ExporterState) -> str:
    uptime = max(0.0, time.time() - state.started_at)
    memory_bytes = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0

    lines = [
        "# HELP senna_active_neurons_ratio Ratio of active neurons.",
        "# TYPE senna_active_neurons_ratio gauge",
        _format_metric("senna_active_neurons_ratio", snapshot.active_neurons_ratio),
        "# HELP senna_spikes_per_tick Average spikes per simulation tick.",
        "# TYPE senna_spikes_per_tick gauge",
        _format_metric("senna_spikes_per_tick", snapshot.spikes_per_tick),
        "# HELP senna_e_rate_hz Average firing rate of excitatory neurons (Hz).",
        "# TYPE senna_e_rate_hz gauge",
        _format_metric("senna_e_rate_hz", snapshot.e_rate_hz),
        "# HELP senna_i_rate_hz Average firing rate of inhibitory neurons (Hz).",
        "# TYPE senna_i_rate_hz gauge",
        _format_metric("senna_i_rate_hz", snapshot.i_rate_hz),
        "# HELP senna_ei_balance Excitatory-to-inhibitory rate ratio.",
        "# TYPE senna_ei_balance gauge",
        _format_metric("senna_ei_balance", snapshot.ei_balance),
        "# HELP senna_train_accuracy Training accuracy [0..1].",
        "# TYPE senna_train_accuracy gauge",
        _format_metric("senna_train_accuracy", snapshot.train_accuracy),
        "# HELP senna_test_accuracy Test accuracy [0..1].",
        "# TYPE senna_test_accuracy gauge",
        _format_metric("senna_test_accuracy", snapshot.test_accuracy),
        "# HELP senna_synapse_count Number of active synapses.",
        "# TYPE senna_synapse_count gauge",
        _format_metric("senna_synapse_count", snapshot.synapse_count),
        "# HELP senna_stdp_updates_total Cumulative number of STDP updates.",
        "# TYPE senna_stdp_updates_total counter",
        _format_metric("senna_stdp_updates_total", snapshot.stdp_updates_total),
        "# HELP senna_pruned_total Cumulative number of pruned synapses.",
        "# TYPE senna_pruned_total counter",
        _format_metric("senna_pruned_total", snapshot.pruned_total),
        "# HELP senna_sprouted_total Cumulative number of sprouted synapses.",
        "# TYPE senna_sprouted_total counter",
        _format_metric("senna_sprouted_total", snapshot.sprouted_total),
        "# HELP senna_tick_duration_seconds Simulation tick duration histogram.",
        "# TYPE senna_tick_duration_seconds histogram",
        *state.histogram.render("senna_tick_duration_seconds"),
        "# HELP senna_training_progress_ratio Estimated full-training progress [0..1].",
        "# TYPE senna_training_progress_ratio gauge",
        _format_metric(
            "senna_training_progress_ratio", snapshot.training_progress_ratio
        ),
        "# HELP senna_training_samples_per_sec Overall training throughput in samples per second.",
        "# TYPE senna_training_samples_per_sec gauge",
        _format_metric(
            "senna_training_samples_per_sec", snapshot.training_samples_per_sec
        ),
        "# HELP senna_training_eta_seconds Estimated seconds remaining until training completion.",
        "# TYPE senna_training_eta_seconds gauge",
        _format_metric("senna_training_eta_seconds", snapshot.training_eta_seconds),
        "# HELP senna_exporter_uptime_seconds Exporter uptime in seconds.",
        "# TYPE senna_exporter_uptime_seconds gauge",
        _format_metric("senna_exporter_uptime_seconds", uptime),
        "# HELP senna_exporter_memory_bytes Exporter memory usage in bytes.",
        "# TYPE senna_exporter_memory_bytes gauge",
        _format_metric("senna_exporter_memory_bytes", memory_bytes),
        "",
    ]

    return "\n".join(lines)


EXPORTER_STATE = ExporterState()


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/metrics":
            snapshot = read_snapshot()
            if snapshot is None:
                payload = f"# snapshot_not_ready path={METRICS_SNAPSHOT_PATH}\n".encode(
                    "utf-8"
                )
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            EXPORTER_STATE.observe(snapshot)
            payload = render_metrics_payload(snapshot, EXPORTER_STATE).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/health":
            snapshot_ready = read_snapshot() is not None
            payload = json.dumps(
                {
                    "status": "ok" if snapshot_ready else "waiting_for_snapshot",
                    "snapshot_ready": snapshot_ready,
                    "snapshot_path": str(METRICS_SNAPSHOT_PATH),
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def run_server() -> None:
    server = HTTPServer((HOST, PORT), MetricsHandler)
    print(f"Prometheus exporter running on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
