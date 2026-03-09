from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import senna_core

from .config import resolve_config_path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in local env
    np = None

try:
    from torchvision import datasets  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency in local env
    datasets = None

MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)

DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class Sample:
    image: list[int]
    label: int


@dataclass(frozen=True)
class ProgressUpdate:
    stage: str
    completed: int
    expected_total: int | None
    accuracy: float
    metrics: dict[str, float]


def iter_sample_batches(
    samples: Iterable["Sample"], batch_size: int
) -> Iterator[list["Sample"]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def as_fast_batch(batch: list["Sample"]) -> tuple[Any, Any] | None:
    if np is None or not batch:
        return None

    images = np.asarray([sample.image for sample in batch], dtype=np.uint8)
    labels = np.fromiter(
        (sample.label for sample in batch), dtype=np.int32, count=len(batch)
    )
    return images, labels


class TrainingPipeline:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = str(resolve_config_path(config_path))
        self.handle = senna_core.create_network(self.config_path)

    def train_epoch(
        self,
        samples: Iterable[Sample],
        ticks_per_sample: int = 100,
        *,
        expected_total: int | None = None,
        progress_every: int = 0,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> dict[str, float]:
        total = 0
        correct = 0
        batch_size = progress_every if progress_every > 0 else DEFAULT_BATCH_SIZE

        self.handle.set_eval_mode(False)
        for batch in iter_sample_batches(samples, batch_size):
            fast_batch = (
                as_fast_batch(batch)
                if hasattr(self.handle, "batch_train_array")
                else None
            )
            if fast_batch is not None:
                images, labels = fast_batch
                batch_result = dict(
                    self.handle.batch_train_array(images, labels, ticks_per_sample)
                )
                total += int(batch_result.get("completed", len(batch)))
                correct += int(batch_result.get("correct", 0))
                batch_metrics = dict(batch_result)
            elif hasattr(self.handle, "batch_train"):
                batch_result = dict(
                    self.handle.batch_train(
                        [sample.image for sample in batch],
                        [sample.label for sample in batch],
                        ticks_per_sample,
                    )
                )
                total += int(batch_result.get("completed", len(batch)))
                correct += int(batch_result.get("correct", 0))
                batch_metrics = dict(batch_result)
            else:
                for sample in batch:
                    self.handle.load_sample(sample.image, sample.label, True)
                    self.handle.step(ticks_per_sample)
                    prediction = self.handle.get_prediction()
                    if prediction != sample.label:
                        self.handle.supervise(sample.label)
                        prediction = self.handle.get_prediction()
                    total += 1
                    if prediction == sample.label:
                        correct += 1
                batch_metrics = dict(self.handle.get_metrics())

            if progress_callback is not None and progress_every > 0:
                progress_callback(
                    ProgressUpdate(
                        stage="train",
                        completed=total,
                        expected_total=expected_total,
                        accuracy=(correct / total) if total else 0.0,
                        metrics=batch_metrics,
                    )
                )

        accuracy = (correct / total) if total else 0.0
        metrics = dict(self.handle.get_metrics())
        metrics["epoch_accuracy"] = accuracy
        metrics["epoch_samples"] = float(total)
        if progress_callback is not None and total > 0:
            progress_callback(
                ProgressUpdate(
                    stage="train",
                    completed=total,
                    expected_total=expected_total,
                    accuracy=accuracy,
                    metrics=dict(metrics),
                )
            )
        return metrics

    def evaluate(
        self,
        samples: Iterable[Sample],
        ticks_per_sample: int = 100,
        *,
        expected_total: int | None = None,
        progress_every: int = 0,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> dict[str, float]:
        total = 0
        correct = 0
        batch_size = progress_every if progress_every > 0 else DEFAULT_BATCH_SIZE

        self.handle.set_eval_mode(True)
        for batch in iter_sample_batches(samples, batch_size):
            fast_batch = (
                as_fast_batch(batch)
                if hasattr(self.handle, "batch_evaluate_array")
                else None
            )
            if fast_batch is not None:
                images, labels = fast_batch
                batch_result = dict(
                    self.handle.batch_evaluate_array(images, labels, ticks_per_sample)
                )
                total += int(batch_result.get("completed", len(batch)))
                correct += int(batch_result.get("correct", 0))
                batch_metrics = dict(batch_result)
            elif hasattr(self.handle, "batch_evaluate"):
                batch_result = dict(
                    self.handle.batch_evaluate(
                        [sample.image for sample in batch],
                        [sample.label for sample in batch],
                        ticks_per_sample,
                    )
                )
                total += int(batch_result.get("completed", len(batch)))
                correct += int(batch_result.get("correct", 0))
                batch_metrics = dict(batch_result)
            else:
                for sample in batch:
                    self.handle.load_sample(sample.image, sample.label, False)
                    self.handle.step(ticks_per_sample)
                    prediction = self.handle.get_prediction()
                    total += 1
                    if prediction == sample.label:
                        correct += 1
                batch_metrics = dict(self.handle.get_metrics())

            if progress_callback is not None and progress_every > 0:
                progress_callback(
                    ProgressUpdate(
                        stage="eval",
                        completed=total,
                        expected_total=expected_total,
                        accuracy=(correct / total) if total else 0.0,
                        metrics=batch_metrics,
                    )
                )

        accuracy = (correct / total) if total else 0.0
        metrics = dict(self.handle.get_metrics())
        metrics["eval_accuracy"] = accuracy
        metrics["eval_samples"] = float(total)
        if progress_callback is not None and total > 0:
            progress_callback(
                ProgressUpdate(
                    stage="eval",
                    completed=total,
                    expected_total=expected_total,
                    accuracy=accuracy,
                    metrics=dict(metrics),
                )
            )
        return metrics

    def capture_trace(
        self, sample: Sample, ticks_per_sample: int = 100
    ) -> dict[str, object]:
        self.handle.set_eval_mode(True)
        self.handle.load_sample(sample.image, sample.label, False)
        frames = list(self.handle.step_with_trace(ticks_per_sample))
        return {
            "label": sample.label,
            "prediction": int(self.handle.get_prediction()),
            "ticks": ticks_per_sample,
            "frames": frames,
            "lattice": dict(self.handle.get_lattice()),
        }

    def save_state(self, path: str) -> None:
        senna_core.save_state(self.handle, path)

    @classmethod
    def load_state(
        cls, path: str, config_path: str | None = None
    ) -> "TrainingPipeline":
        pipeline = cls(config_path)
        load_config_path = pipeline.config_path if config_path is not None else ""
        pipeline.handle = senna_core.load_state(path, load_config_path)
        return pipeline


def evaluate_from_state(
    *,
    state_path: str,
    config_path: str | None,
    samples: Iterable[Sample],
    ticks_per_sample: int,
    mutate: Callable[[object], None] | None = None,
) -> dict[str, float]:
    pipeline = TrainingPipeline.load_state(state_path, config_path=config_path)
    if mutate is not None:
        mutate(pipeline.handle)
    return pipeline.evaluate(samples, ticks_per_sample=ticks_per_sample)


def capture_trace_from_state(
    *,
    state_path: str,
    config_path: str | None,
    sample: Sample,
    ticks_per_sample: int,
) -> dict[str, object]:
    pipeline = TrainingPipeline.load_state(state_path, config_path=config_path)
    return pipeline.capture_trace(sample, ticks_per_sample=ticks_per_sample)


def robustness_report(
    *,
    state_path: str,
    config_path: str | None,
    sample_factory: Callable[[], Iterable[Sample]],
    ticks_per_sample: int,
    remove_fraction: float = 0.1,
    noise_sigma: float = 0.3,
) -> dict[str, float]:
    baseline = evaluate_from_state(
        state_path=state_path,
        config_path=config_path,
        samples=sample_factory(),
        ticks_per_sample=ticks_per_sample,
    )
    baseline_acc = baseline.get("eval_accuracy", 0.0)

    pruned = evaluate_from_state(
        state_path=state_path,
        config_path=config_path,
        samples=sample_factory(),
        ticks_per_sample=ticks_per_sample,
        mutate=lambda handle: handle.remove_neurons(remove_fraction),
    )
    pruned_acc = pruned.get("eval_accuracy", 0.0)

    noised = evaluate_from_state(
        state_path=state_path,
        config_path=config_path,
        samples=sample_factory(),
        ticks_per_sample=ticks_per_sample,
        mutate=lambda handle: handle.inject_noise(noise_sigma),
    )
    noised_acc = noised.get("eval_accuracy", 0.0)

    prune_drop = max(0.0, baseline_acc - pruned_acc)
    noise_drop = max(0.0, baseline_acc - noised_acc)
    has_meaningful_baseline = baseline_acc > 0.0
    return {
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "noise_accuracy": noised_acc,
        "prune_drop": prune_drop,
        "noise_drop": noise_drop,
        "prune_pass": 1.0 if has_meaningful_baseline and prune_drop < 0.05 else 0.0,
        "noise_pass": 1.0 if has_meaningful_baseline and noise_drop < 0.10 else 0.0,
    }


def iter_mnist_samples(
    *,
    root: str | Path = "data",
    train: bool,
    limit: int | None = None,
    download: bool = False,
) -> Iterator[Sample]:
    if datasets is None:
        raise RuntimeError(
            "Real MNIST requires host Python packages torch and torchvision. "
            "Install them in the current environment before running train.py. "
            "MinIO is not used for MNIST input."
        )

    raw_dir = Path(root) / "MNIST" / "raw"
    if not download:
        missing_files = [
            name for name in MNIST_RAW_FILES if not (raw_dir / name).exists()
        ]
        if missing_files:
            missing_list = ", ".join(missing_files)
            raise RuntimeError(
                f"MNIST raw files are missing under {raw_dir}: {missing_list}. "
                "Run `make install` to download them locally or rerun train.py with --download."
            )

    dataset = datasets.MNIST(root=str(root), train=train, download=download)
    max_items = len(dataset) if limit is None else min(limit, len(dataset))

    for index in range(max_items):
        image, label = dataset[index]
        pixels = [int(value) for value in image.getdata()]
        yield Sample(image=pixels, label=int(label))


def make_synthetic_digit_samples(count: int, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []

    for idx in range(count):
        label = idx % 10
        image = [0] * (28 * 28)

        base_row = 2 + (label * 2) % 20
        base_col = 2 + (label * 3) % 20
        for row in range(base_row, min(base_row + 6, 28)):
            for col in range(base_col, min(base_col + 6, 28)):
                value = 120 + ((label * 13 + row + col) % 120)
                image[row * 28 + col] = value

        noise_points = 24
        for _ in range(noise_points):
            noise_index = rng.randrange(28 * 28)
            image[noise_index] = max(image[noise_index], rng.randrange(0, 60))

        samples.append(Sample(image=image, label=label))

    return samples
