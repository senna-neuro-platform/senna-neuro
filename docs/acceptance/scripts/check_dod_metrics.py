#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
                raise ValueError(f"invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no} must contain a JSON object")
            events.append(payload)

    if not events:
        raise ValueError(f"metrics file is empty: {path}")
    return events


def last_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == event_name:
            return event
    return None


def collect_events(
    events: list[dict[str, Any]], event_name: str
) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == event_name]


def read_float(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool):
        return None
    if isinstance(current, int | float):
        return float(current)
    return None


def first_defined_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DoD numeric gates from training metrics JSONL."
    )
    parser.add_argument(
        "--metrics-path",
        default="data/artifacts/training/metrics.jsonl",
        help="Path to train.py metrics JSONL",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.85,
        help="Required eval_accuracy threshold",
    )
    parser.add_argument(
        "--max-active-ratio",
        type=float,
        default=0.05,
        help="Maximum allowed active neurons ratio",
    )
    parser.add_argument(
        "--max-prune-drop",
        type=float,
        default=0.05,
        help="Maximum allowed drop after remove_neurons(0.1)",
    )
    parser.add_argument(
        "--max-noise-drop",
        type=float,
        default=0.10,
        help="Maximum allowed drop after inject_noise(0.3)",
    )
    parser.add_argument(
        "--require-dataset",
        default="mnist",
        help="Require this dataset_mode in all epoch_end events (default: mnist).",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    try:
        events = load_jsonl(metrics_path)
    except (FileNotFoundError, ValueError) as exc:
        return fail(str(exc))

    epoch_events = collect_events(events, "epoch_end")
    if not epoch_events:
        return fail("metrics JSONL does not contain event='epoch_end'")
    epoch_event = epoch_events[-1]

    robustness_event = last_event(events, "robustness")
    if robustness_event is None:
        return fail("metrics JSONL does not contain event='robustness'")

    if args.require_dataset:
        for index, event in enumerate(epoch_events, start=1):
            dataset_mode = event.get("dataset_mode")
            if dataset_mode != args.require_dataset:
                return fail(
                    f"epoch_end #{index} has dataset_mode={dataset_mode!r}, expected {args.require_dataset!r}"
                )

    eval_accuracy = read_float(epoch_event, "eval", "eval_accuracy")
    if eval_accuracy is None:
        return fail("missing eval.eval_accuracy in epoch_end event")

    active_ratios: list[float] = []
    for event in epoch_events:
        active_ratio = first_defined_float(
            read_float(event, "eval", "senna_max_active_neurons_ratio"),
            read_float(event, "eval", "max_active_neurons_ratio"),
            read_float(event, "eval", "senna_active_neurons_ratio"),
            read_float(event, "eval", "active_neurons_ratio"),
            read_float(event, "eval", "active_ratio"),
        )
        if active_ratio is None:
            return fail(
                "missing eval.senna_max_active_neurons_ratio in epoch_end event"
            )
        active_ratios.append(active_ratio)

    max_active_ratio = max(active_ratios)

    prune_drop = read_float(robustness_event, "metrics", "prune_drop")
    noise_drop = read_float(robustness_event, "metrics", "noise_drop")
    if prune_drop is None:
        return fail("missing metrics.prune_drop in robustness event")
    if noise_drop is None:
        return fail("missing metrics.noise_drop in robustness event")

    failures: list[str] = []
    if eval_accuracy < args.target_accuracy:
        failures.append(
            f"eval_accuracy {eval_accuracy:.4f} < target {args.target_accuracy:.4f}"
        )
    if max_active_ratio > args.max_active_ratio:
        failures.append(
            f"max_active_ratio {max_active_ratio:.4f} > max {args.max_active_ratio:.4f}"
        )
    if prune_drop >= args.max_prune_drop:
        failures.append(f"prune_drop {prune_drop:.4f} >= max {args.max_prune_drop:.4f}")
    if noise_drop >= args.max_noise_drop:
        failures.append(f"noise_drop {noise_drop:.4f} >= max {args.max_noise_drop:.4f}")

    print(f"metrics_path={metrics_path}")
    print(
        "summary "
        f"eval_accuracy={eval_accuracy:.4f} "
        f"max_active_ratio={max_active_ratio:.4f} "
        f"prune_drop={prune_drop:.4f} "
        f"noise_drop={noise_drop:.4f}"
    )

    if failures:
        print("[FAIL] Numeric DoD gates failed:")
        for issue in failures:
            print(f"  - {issue}")
        return 1

    print("[PASS] Numeric DoD gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
