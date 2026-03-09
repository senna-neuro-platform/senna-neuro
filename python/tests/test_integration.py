from __future__ import annotations

from pathlib import Path

import pytest
import senna_core

import senna.training as training_module
from senna.training import (
    TrainingPipeline,
    Sample,
    make_synthetic_digit_samples,
    robustness_report,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in local env
    np = None


def test_bindings_full_cycle(tmp_path: Path) -> None:
    pipeline = TrainingPipeline(config_path="configs/default.yaml")
    samples = make_synthetic_digit_samples(12, seed=7)

    metrics = pipeline.train_epoch(samples, ticks_per_sample=32)
    assert metrics["epoch_samples"] == 12.0
    assert 0.0 <= metrics["epoch_accuracy"] <= 1.0

    prediction = pipeline.handle.get_prediction()
    assert -1 <= prediction <= 9

    exported = pipeline.handle.get_metrics()
    assert "senna_spikes_per_tick" in exported
    assert "senna_train_accuracy" in exported

    state_path = tmp_path / "state.h5"
    pipeline.save_state(str(state_path))
    assert state_path.exists()

    restored = TrainingPipeline.load_state(
        str(state_path), config_path="configs/default.yaml"
    )
    restored_samples = make_synthetic_digit_samples(4, seed=17)
    restored_metrics = restored.evaluate(restored_samples, ticks_per_sample=24)
    assert restored_metrics["eval_samples"] == 4.0


def test_module_level_api(tmp_path: Path) -> None:
    handle = senna_core.create_network("configs/default.yaml")
    image = [0] * (28 * 28)
    for i in range(0, 28 * 28, 29):
        image[i] = 200

    senna_core.load_sample(handle, image, 3)
    senna_core.step(handle, 40)

    prediction = senna_core.get_prediction(handle)
    assert -1 <= prediction <= 9
    if prediction != 3:
        senna_core.supervise(handle, 3)
        assert senna_core.get_prediction(handle) == 3

    metrics = senna_core.get_metrics(handle)
    assert "senna_active_neurons_ratio" in metrics

    state_path = tmp_path / "module_state.h5"
    senna_core.save_state(handle, str(state_path))
    assert state_path.exists()

    loaded = senna_core.load_state(str(state_path), "configs/default.yaml")
    senna_core.inject_noise(loaded, 0.1)
    senna_core.remove_neurons(loaded, 0.01)
    senna_core.step(loaded, 10)


def test_module_level_trace_export() -> None:
    handle = senna_core.create_network("configs/default.yaml")
    image = [0] * (28 * 28)
    for i in range(0, 28 * 28, 17):
        image[i] = 255

    senna_core.load_sample(handle, image, 5, False)
    frames = senna_core.step_with_trace(handle, 12)
    lattice = senna_core.get_lattice(handle)

    assert len(frames) == 12
    assert lattice["neuronCount"] == len(lattice["neurons"])
    assert lattice["width"] >= 28
    assert lattice["depth"] >= 2
    assert any(frame["totalNeurons"] == lattice["neuronCount"] for frame in frames)


def test_module_level_batch_api() -> None:
    handle = senna_core.create_network("configs/default.yaml")
    images = []
    labels = []

    for label in range(4):
        image = [0] * (28 * 28)
        for offset in range(label, 28 * 28, 31):
            image[offset] = 180 + (label * 10)
        images.append(image)
        labels.append(label)

    train_result = senna_core.batch_train(handle, images, labels, 24)
    assert train_result["completed"] == 4
    assert 0.0 <= train_result["batch_accuracy"] <= 1.0
    assert "senna_train_accuracy" in train_result

    eval_result = senna_core.batch_evaluate(handle, images, labels, 24)
    assert eval_result["completed"] == 4
    assert 0.0 <= eval_result["batch_accuracy"] <= 1.0
    assert "senna_test_accuracy" in eval_result


def test_module_level_batch_array_api() -> None:
    if np is None:
        pytest.skip("numpy is not available")

    handle = senna_core.create_network("configs/default.yaml")
    images = np.zeros((4, 28 * 28), dtype=np.uint8)
    labels = np.arange(4, dtype=np.int32)

    for label in range(4):
        for offset in range(label, 28 * 28, 31):
            images[label, offset] = 180 + (label * 10)

    train_result = senna_core.batch_train_array(handle, images, labels, 24)
    assert train_result["completed"] == 4
    assert 0.0 <= train_result["batch_accuracy"] <= 1.0
    assert "senna_train_accuracy" in train_result

    eval_result = senna_core.batch_evaluate_array(handle, images, labels, 24)
    assert eval_result["completed"] == 4
    assert 0.0 <= eval_result["batch_accuracy"] <= 1.0
    assert "senna_test_accuracy" in eval_result


def test_robustness_report_smoke(tmp_path: Path) -> None:
    pipeline = TrainingPipeline(config_path="configs/default.yaml")
    train_samples = make_synthetic_digit_samples(8, seed=23)
    pipeline.train_epoch(train_samples, ticks_per_sample=20)

    state_path = tmp_path / "robust_state.h5"
    pipeline.save_state(str(state_path))

    report = robustness_report(
        state_path=str(state_path),
        config_path="configs/default.yaml",
        sample_factory=lambda: make_synthetic_digit_samples(6, seed=31),
        ticks_per_sample=20,
        remove_fraction=0.1,
        noise_sigma=0.3,
    )

    assert 0.0 <= report["baseline_accuracy"] <= 1.0
    assert 0.0 <= report["pruned_accuracy"] <= 1.0
    assert 0.0 <= report["noise_accuracy"] <= 1.0
    assert report["prune_drop"] >= 0.0
    assert report["noise_drop"] >= 0.0


def test_robustness_report_requires_positive_baseline(monkeypatch) -> None:
    responses = iter(
        [
            {"eval_accuracy": 0.0},
            {"eval_accuracy": 0.0},
            {"eval_accuracy": 0.0},
        ]
    )

    def fake_evaluate_from_state(**_: object) -> dict[str, float]:
        return next(responses)

    monkeypatch.setattr("senna.training.evaluate_from_state", fake_evaluate_from_state)

    report = robustness_report(
        state_path="unused.h5",
        config_path="configs/default.yaml",
        sample_factory=lambda: [Sample(image=[0] * (28 * 28), label=0)],
        ticks_per_sample=8,
    )

    assert report["baseline_accuracy"] == 0.0
    assert report["prune_pass"] == 0.0
    assert report["noise_pass"] == 0.0


def test_iter_mnist_samples_requires_torch_and_torchvision(monkeypatch) -> None:
    monkeypatch.setattr(training_module, "datasets", None)

    with pytest.raises(RuntimeError, match="torch and torchvision"):
        next(training_module.iter_mnist_samples(root="data", train=True, limit=1))


def test_iter_mnist_samples_requires_local_raw_files(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(training_module, "datasets", object())

    with pytest.raises(RuntimeError, match="MNIST raw files are missing"):
        next(training_module.iter_mnist_samples(root=tmp_path, train=True, limit=1))
