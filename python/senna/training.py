from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import senna_core

from .config import resolve_config_path

try:
    from torchvision import datasets  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency in local env
    datasets = None


@dataclass(frozen=True)
class Sample:
    image: list[int]
    label: int


class TrainingPipeline:
    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = str(resolve_config_path(config_path))
        self.handle = senna_core.create_network(self.config_path)

    def train_epoch(
        self, samples: Iterable[Sample], ticks_per_sample: int = 100
    ) -> dict[str, float]:
        total = 0
        correct = 0

        self.handle.set_eval_mode(False)
        for sample in samples:
            self.handle.load_sample(sample.image, sample.label, True)
            self.handle.step(ticks_per_sample)
            prediction = self.handle.get_prediction()
            if prediction != sample.label:
                self.handle.supervise(sample.label)
                prediction = self.handle.get_prediction()
            total += 1
            if prediction == sample.label:
                correct += 1

        accuracy = (correct / total) if total else 0.0
        metrics = dict(self.handle.get_metrics())
        metrics["epoch_accuracy"] = accuracy
        metrics["epoch_samples"] = float(total)
        return metrics

    def evaluate(
        self, samples: Iterable[Sample], ticks_per_sample: int = 100
    ) -> dict[str, float]:
        total = 0
        correct = 0

        self.handle.set_eval_mode(True)
        for sample in samples:
            self.handle.load_sample(sample.image, sample.label, False)
            self.handle.step(ticks_per_sample)
            prediction = self.handle.get_prediction()
            total += 1
            if prediction == sample.label:
                correct += 1

        accuracy = (correct / total) if total else 0.0
        metrics = dict(self.handle.get_metrics())
        metrics["eval_accuracy"] = accuracy
        metrics["eval_samples"] = float(total)
        return metrics

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


def iter_mnist_samples(
    *,
    root: str | Path = "data",
    train: bool,
    limit: int | None = None,
    download: bool = False,
) -> Iterator[Sample]:
    if datasets is None:
        raise RuntimeError(
            "torchvision is not installed. Install it to use real MNIST in train.py."
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
