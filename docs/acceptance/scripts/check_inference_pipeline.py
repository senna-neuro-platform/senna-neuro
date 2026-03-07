#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT_DIR / "python"
BUILD_DEBUG_DIR = ROOT_DIR / "build" / "debug"

if str(BUILD_DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DEBUG_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from senna.training import (  # noqa: E402
    TrainingPipeline,
    iter_mnist_samples,
    make_synthetic_digit_samples,
)


def resolve_dataset_mode(dataset: str, metrics_path: Path | None) -> str:
    if dataset in {"mnist", "synthetic"}:
        return dataset

    if metrics_path and metrics_path.exists():
        try:
            last_line = ""
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last_line = line
            if last_line:
                payload = json.loads(last_line)
                if isinstance(payload, dict):
                    mode = payload.get("dataset_mode")
                    if mode in {"mnist", "synthetic"}:
                        return mode
        except (OSError, json.JSONDecodeError):
            pass

    return "mnist"


def select_sample(
    *,
    dataset_mode: str,
    data_root: str,
    sample_index: int,
    download: bool,
    seed: int,
) -> tuple[list[int], int]:
    if sample_index < 0:
        raise ValueError("sample-index must be >= 0")

    if dataset_mode == "mnist":
        iterator = iter_mnist_samples(
            root=data_root,
            train=False,
            limit=sample_index + 1,
            download=download,
        )
        for idx, sample in enumerate(iterator):
            if idx == sample_index:
                return sample.image, sample.label
        raise RuntimeError("MNIST sample index is out of range")

    samples = make_synthetic_digit_samples(sample_index + 1, seed=seed)
    sample = samples[sample_index]
    return sample.image, sample.label


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Step 16 inference pipeline from saved state."
    )
    parser.add_argument(
        "--state-path",
        default="data/artifacts/outbox/final_state.h5",
        help="Path to saved network state",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "mnist", "synthetic"),
        default="auto",
        help="Sample source for inference check",
    )
    parser.add_argument(
        "--metrics-path",
        default="data/artifacts/training/metrics.jsonl",
        help="Metrics JSONL path (used when --dataset=auto)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index in selected dataset split",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=100,
        help="Ticks to run after load_sample",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download MNIST if needed",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=42,
        help="Seed for synthetic dataset mode",
    )
    args = parser.parse_args()

    state_path = Path(args.state_path)
    if not state_path.exists():
        return fail(f"state file not found: {state_path}")

    metrics_path = Path(args.metrics_path)
    dataset_mode = resolve_dataset_mode(args.dataset, metrics_path)

    try:
        image, label = select_sample(
            dataset_mode=dataset_mode,
            data_root=args.data_root,
            sample_index=args.sample_index,
            download=args.download,
            seed=args.synthetic_seed,
        )
    except Exception as exc:
        return fail(f"cannot load sample ({dataset_mode}): {exc}")

    try:
        pipeline = TrainingPipeline.load_state(
            str(state_path),
            config_path=args.config,
        )
    except Exception as exc:
        return fail(f"cannot load state into bindings: {exc}")

    pipeline.handle.set_eval_mode(True)
    pipeline.handle.load_sample(image, label, False)
    pipeline.handle.step(args.ticks)
    prediction = int(pipeline.handle.get_prediction())
    metrics: dict[str, Any] = dict(pipeline.handle.get_metrics())

    summary = {
        "dataset_mode": dataset_mode,
        "sample_index": args.sample_index,
        "label": label,
        "prediction": prediction,
        "ticks": args.ticks,
        "senna_active_neurons_ratio": float(
            metrics.get("senna_active_neurons_ratio", 0.0)
        ),
        "senna_spikes_per_tick": float(metrics.get("senna_spikes_per_tick", 0.0)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if prediction < 0 or prediction > 9:
        return fail("prediction is outside class range [0..9]")

    print("[PASS] Inference pipeline check passed (image -> class 0..9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
