from __future__ import annotations

from pathlib import Path

import senna_core

from senna.training import (
    TrainingPipeline,
    make_synthetic_digit_samples,
    robustness_report,
)


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
