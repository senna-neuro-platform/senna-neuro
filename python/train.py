from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from senna.training import (
    ProgressUpdate,
    TrainingPipeline,
    capture_trace_from_state,
    iter_mnist_samples,
    make_synthetic_digit_samples,
    robustness_report,
)


def append_metrics_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def resolve_dataset_mode(args: argparse.Namespace) -> Literal["mnist", "synthetic"]:
    if args.dataset == "synthetic":
        return "synthetic"

    train_probe = iter_mnist_samples(
        root=args.data_root,
        train=True,
        limit=1,
        download=args.download,
    )
    next(iter(train_probe))

    test_probe = iter_mnist_samples(
        root=args.data_root,
        train=False,
        limit=1,
        download=args.download,
    )
    next(iter(test_probe))
    return "mnist"


def make_samples(
    *,
    dataset_mode: Literal["mnist", "synthetic"],
    data_root: str,
    train: bool,
    limit: int | None,
    epoch_seed: int,
    download: bool,
) -> Iterable:
    if dataset_mode == "mnist":
        return iter_mnist_samples(
            root=data_root,
            train=train,
            limit=limit,
            download=download,
        )

    fallback_limit = limit if limit is not None else (2000 if train else 1000)
    seed_shift = 0 if train else 100_000
    return make_synthetic_digit_samples(fallback_limit, seed=epoch_seed + seed_shift)


def diagnostics_for_step15(
    eval_metrics: dict[str, float], prediction_history: list[int]
) -> list[str]:
    diagnostics: list[str] = []
    spikes_per_tick = eval_metrics.get("senna_spikes_per_tick", 0.0)
    active_ratio = eval_metrics.get("senna_active_neurons_ratio", 0.0)
    eval_accuracy = eval_metrics.get("eval_accuracy", 0.0)
    ei_balance = eval_metrics.get("senna_ei_balance", 0.0)

    if spikes_per_tick <= 0.1:
        diagnostics.append(
            "Сеть молчит: попробуйте снизить theta_base, увеличить w_init и max_rate."
        )
    if active_ratio > 0.5:
        diagnostics.append(
            "Сеть эпилептична: увеличьте theta_base, долю inhibitory и модуль W_wta."
        )
    if eval_accuracy < 0.3:
        diagnostics.append(
            "Точность низкая: проверьте A_plus/A_minus, tau_plus/tau_minus и распределение весов."
        )
    if len(prediction_history) >= 32:
        recent = prediction_history[-32:]
        dominant = max(recent.count(cls) for cls in set(recent))
        if dominant >= 30:
            diagnostics.append(
                "Один выход доминирует: ослабьте W_wta и усилите homeostasis (eta_homeo)."
            )
    if ei_balance <= 0.05:
        diagnostics.append(
            "E/I дисбаланс: проверьте excitatory_ratio и параметры homeostasis."
        )

    return diagnostics


def metric_float(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value) if isinstance(value, int | float) else default


def write_metrics_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def first_sample(
    *,
    dataset_mode: Literal["mnist", "synthetic"],
    data_root: str,
    train: bool,
    epoch_seed: int,
    download: bool,
):
    iterator = iter(
        make_samples(
            dataset_mode=dataset_mode,
            data_root=data_root,
            train=train,
            limit=1,
            epoch_seed=epoch_seed,
            download=download,
        )
    )
    sample = next(iterator, None)
    if sample is None:
        raise RuntimeError("No samples available for visualizer trace export")
    return sample


def exporter_snapshot_payload(
    *,
    snapshot_source: str,
    dataset_mode: Literal["mnist", "synthetic"],
    epoch: int,
    train_limit: int,
    test_limit: int,
    ticks: int,
    checkpoint_path: Path | None,
    runtime_metrics: dict[str, float],
    train_accuracy: float,
    test_accuracy: float,
    progress_stage: str | None = None,
    progress_completed: int | None = None,
    progress_total: int | None = None,
    training_progress_ratio: float | None = None,
    training_samples_per_sec: float | None = None,
    training_eta_seconds: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_source": snapshot_source,
        "dataset_mode": dataset_mode,
        "epoch": epoch,
        "train_limit": train_limit,
        "test_limit": test_limit,
        "ticks": ticks,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "active_neurons_ratio": metric_float(
            runtime_metrics, "senna_active_neurons_ratio"
        ),
        "max_active_neurons_ratio": metric_float(
            runtime_metrics,
            "senna_max_active_neurons_ratio",
            metric_float(runtime_metrics, "senna_active_neurons_ratio"),
        ),
        "mean_active_neurons_ratio": metric_float(
            runtime_metrics, "senna_mean_active_neurons_ratio"
        ),
        "spikes_per_tick": metric_float(runtime_metrics, "senna_spikes_per_tick"),
        "mean_spikes_per_tick": metric_float(
            runtime_metrics, "senna_mean_spikes_per_tick"
        ),
        "ticks_total": metric_float(runtime_metrics, "senna_ticks_total"),
        "spikes_total": metric_float(runtime_metrics, "senna_spikes_total"),
        "e_rate_hz": metric_float(runtime_metrics, "senna_e_rate_hz"),
        "i_rate_hz": metric_float(runtime_metrics, "senna_i_rate_hz"),
        "ei_balance": metric_float(runtime_metrics, "senna_ei_balance"),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "synapse_count": metric_float(runtime_metrics, "senna_synapse_count"),
        "pruned_total": metric_float(runtime_metrics, "senna_pruned_total"),
        "sprouted_total": metric_float(runtime_metrics, "senna_sprouted_total"),
        "stdp_updates_total": metric_float(runtime_metrics, "senna_stdp_updates_total"),
        "tick_duration_seconds": metric_float(
            runtime_metrics, "senna_tick_duration_seconds"
        ),
    }

    if progress_stage is not None:
        payload["progress_stage"] = progress_stage
    if progress_completed is not None:
        payload["progress_completed"] = progress_completed
    if progress_total is not None:
        payload["progress_total"] = progress_total
    if training_progress_ratio is not None:
        payload["training_progress_ratio"] = max(
            0.0, min(1.0, float(training_progress_ratio))
        )
    if training_samples_per_sec is not None:
        payload["training_samples_per_sec"] = max(0.0, float(training_samples_per_sec))
    if training_eta_seconds is not None:
        payload["training_eta_seconds"] = max(0.0, float(training_eta_seconds))

    return payload


def visualizer_trace_payload(
    *,
    trace_source: str,
    dataset_mode: Literal["mnist", "synthetic"],
    epoch: int,
    checkpoint_path: Path | None,
    sample,
    trace: dict[str, object],
) -> dict[str, object]:
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "trace_source": trace_source,
        "dataset_mode": dataset_mode,
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "label": sample.label,
        "prediction": trace["prediction"],
        "ticks": trace["ticks"],
        "lattice": trace["lattice"],
        "frames": trace["frames"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SENNA Neuro training entrypoint")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--dataset",
        choices=("mnist", "synthetic"),
        default="mnist",
        help="Dataset source",
    )
    parser.add_argument(
        "--data-root", default="data", help="Root directory for datasets"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download MNIST if files are missing",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=60000,
        help="Number of train samples per epoch",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=10000,
        help="Number of test samples per epoch",
    )
    parser.add_argument("--ticks", type=int, default=100, help="Ticks per sample")
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.85,
        help="Target eval accuracy for early stop",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stop after reaching target accuracy",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/artifacts/outbox",
        help="Directory for per-epoch checkpoints",
    )
    parser.add_argument(
        "--state-out",
        default="data/artifacts/outbox/final_state.h5",
        help="Path to state file",
    )
    parser.add_argument(
        "--metrics-out",
        default="data/artifacts/training/metrics.jsonl",
        help="JSONL path for epoch and robustness metrics",
    )
    parser.add_argument(
        "--metrics-snapshot-path",
        default="data/artifacts/metrics/latest.json",
        help="JSON snapshot path consumed by Prometheus exporter",
    )
    parser.add_argument(
        "--visualizer-trace-path",
        default="data/artifacts/visualizer/latest.json",
        help="JSON trace path consumed by visualizer websocket server",
    )
    parser.add_argument(
        "--robust-remove-fraction",
        type=float,
        default=0.1,
        help="Fraction of neurons removed for robustness check",
    )
    parser.add_argument(
        "--robust-noise-sigma",
        type=float,
        default=0.3,
        help="Threshold noise sigma for robustness check",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip post-training robustness checks",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress and refresh live metrics every N samples per stage",
    )
    parser.add_argument(
        "--live-trace-every",
        type=int,
        default=250,
        help="Refresh visualizer trace every N train samples (0 disables mid-epoch trace refresh)",
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    try:
        dataset_mode = resolve_dataset_mode(args)
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)
    metrics_snapshot_path = Path(args.metrics_snapshot_path)
    visualizer_trace_path = Path(args.visualizer_trace_path)
    live_dir = checkpoint_dir.parent / "_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_state_path = live_dir / "progress_state.h5"
    prediction_history: list[int] = []
    last_train_accuracy = 0.0
    last_eval_accuracy = 0.0
    last_checkpoint_path: Path | None = None
    run_started_at = time.perf_counter()
    total_work_items = max(1, args.epochs * (args.train_limit + args.test_limit))
    work_items_per_epoch = args.train_limit + args.test_limit

    def compute_training_progress(
        *,
        epoch_index: int,
        stage: Literal["train", "eval"] | None = None,
        completed: int = 0,
    ) -> tuple[float, float, float]:
        if stage is None:
            overall_completed = min(
                total_work_items, epoch_index * work_items_per_epoch
            )
        else:
            completed_before_epoch = (epoch_index - 1) * work_items_per_epoch
            completed_before_stage = completed_before_epoch + (
                args.train_limit if stage == "eval" else 0
            )
            overall_completed = min(
                total_work_items, completed_before_stage + completed
            )

        total_elapsed = max(0.001, time.perf_counter() - run_started_at)
        training_samples_per_sec = (
            overall_completed / total_elapsed if overall_completed > 0 else 0.0
        )
        training_eta_seconds = (
            max(0.0, (total_work_items - overall_completed) / training_samples_per_sec)
            if training_samples_per_sec > 0.0
            else 0.0
        )
        training_progress_ratio = overall_completed / float(total_work_items)
        return (
            training_progress_ratio,
            training_samples_per_sec,
            training_eta_seconds,
        )

    visualizer_sample = first_sample(
        dataset_mode=dataset_mode,
        data_root=args.data_root,
        train=False,
        epoch_seed=7_000,
        download=False,
    )

    def write_live_snapshot(
        *,
        snapshot_source: str,
        epoch: int,
        runtime_metrics: dict[str, float],
        train_accuracy: float,
        test_accuracy: float,
        progress_stage: str | None = None,
        progress_completed: int | None = None,
        progress_total: int | None = None,
        training_progress_ratio: float | None = None,
        training_samples_per_sec: float | None = None,
        training_eta_seconds: float | None = None,
    ) -> None:
        write_metrics_snapshot(
            metrics_snapshot_path,
            exporter_snapshot_payload(
                snapshot_source=snapshot_source,
                dataset_mode=dataset_mode,
                epoch=epoch,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
                ticks=args.ticks,
                checkpoint_path=None,
                runtime_metrics=runtime_metrics,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                progress_stage=progress_stage,
                progress_completed=progress_completed,
                progress_total=progress_total,
                training_progress_ratio=training_progress_ratio,
                training_samples_per_sec=training_samples_per_sec,
                training_eta_seconds=training_eta_seconds,
            ),
        )

    def write_live_trace(*, epoch: int, trace_source: str) -> None:
        pipeline.save_state(str(live_state_path))
        trace = capture_trace_from_state(
            state_path=str(live_state_path),
            config_path=args.config,
            sample=visualizer_sample,
            ticks_per_sample=args.ticks,
        )
        write_metrics_snapshot(
            visualizer_trace_path,
            visualizer_trace_payload(
                trace_source=trace_source,
                dataset_mode=dataset_mode,
                epoch=epoch,
                checkpoint_path=None,
                sample=visualizer_sample,
                trace=trace,
            ),
        )

    write_live_snapshot(
        snapshot_source="training_bootstrap",
        epoch=0,
        runtime_metrics=dict(pipeline.handle.get_metrics()),
        train_accuracy=0.0,
        test_accuracy=0.0,
        progress_stage="bootstrap",
        progress_completed=0,
        progress_total=work_items_per_epoch,
        training_progress_ratio=0.0,
        training_samples_per_sec=0.0,
        training_eta_seconds=0.0,
    )
    write_live_trace(epoch=0, trace_source="training_bootstrap")
    print(
        f"training_bootstrap metrics_snapshot={metrics_snapshot_path} "
        f"visualizer_trace={visualizer_trace_path}",
        flush=True,
    )

    for epoch in range(args.epochs):
        epoch_index = epoch + 1
        train_samples = make_samples(
            dataset_mode=dataset_mode,
            data_root=args.data_root,
            train=True,
            limit=args.train_limit,
            epoch_seed=42 + epoch,
            download=args.download and epoch == 0,
        )
        eval_samples = make_samples(
            dataset_mode=dataset_mode,
            data_root=args.data_root,
            train=False,
            limit=args.test_limit,
            epoch_seed=142 + epoch,
            download=False,
        )

        train_stage_started_at = time.perf_counter()
        last_train_progress_print = 0
        last_live_trace_emit = 0

        def handle_progress(update: ProgressUpdate) -> None:
            nonlocal last_train_accuracy
            nonlocal last_train_progress_print
            nonlocal last_live_trace_emit
            nonlocal last_eval_accuracy

            stage_elapsed = max(0.001, time.perf_counter() - train_stage_started_at)
            stage_samples_per_sec = (
                update.completed / stage_elapsed if update.completed > 0 else 0.0
            )
            stage_eta_seconds = None
            if update.expected_total is not None and stage_samples_per_sec > 0.0:
                stage_eta_seconds = max(
                    0.0,
                    (update.expected_total - update.completed) / stage_samples_per_sec,
                )

            (
                training_progress_ratio,
                training_samples_per_sec,
                training_eta_seconds,
            ) = compute_training_progress(
                epoch_index=epoch_index,
                stage=update.stage,
                completed=update.completed,
            )

            should_print = (
                args.progress_every > 0
                and update.completed >= last_train_progress_print + args.progress_every
            ) or (
                update.expected_total is not None
                and update.completed == update.expected_total
            )

            if should_print:
                eta_text = (
                    f" eta_sec={stage_eta_seconds:.1f}"
                    if stage_eta_seconds is not None
                    else ""
                )
                total_text = (
                    str(update.expected_total)
                    if update.expected_total is not None
                    else "?"
                )
                print(
                    f"progress stage={update.stage} epoch={epoch_index} "
                    f"samples={update.completed}/{total_text} "
                    f"accuracy={update.accuracy:.4f} "
                    f"spikes_per_tick={metric_float(update.metrics, 'senna_spikes_per_tick'):.4f} "
                    f"active_ratio={metric_float(update.metrics, 'senna_active_neurons_ratio'):.4f} "
                    f"samples_per_sec={stage_samples_per_sec:.2f}{eta_text}",
                    flush=True,
                )
                last_train_progress_print = update.completed

            if update.stage == "train":
                last_train_accuracy = update.accuracy

            write_live_snapshot(
                snapshot_source=f"{update.stage}_progress",
                epoch=epoch_index,
                runtime_metrics=update.metrics,
                train_accuracy=last_train_accuracy,
                test_accuracy=last_eval_accuracy,
                progress_stage=update.stage,
                progress_completed=update.completed,
                progress_total=update.expected_total,
                training_progress_ratio=training_progress_ratio,
                training_samples_per_sec=training_samples_per_sec,
                training_eta_seconds=training_eta_seconds,
            )

            should_refresh_trace = (
                update.stage == "train"
                and args.live_trace_every > 0
                and update.completed >= last_live_trace_emit + args.live_trace_every
            ) or (
                update.stage == "train"
                and update.expected_total is not None
                and update.completed == update.expected_total
            )
            if should_refresh_trace:
                write_live_trace(epoch=epoch_index, trace_source="train_progress")
                print(
                    f"live_trace_refreshed epoch={epoch_index} samples={update.completed}",
                    flush=True,
                )
                last_live_trace_emit = update.completed

        train_metrics = pipeline.train_epoch(
            train_samples,
            ticks_per_sample=args.ticks,
            expected_total=args.train_limit,
            progress_every=min(
                value
                for value in (args.progress_every, args.live_trace_every)
                if value > 0
            )
            if any(value > 0 for value in (args.progress_every, args.live_trace_every))
            else 0,
            progress_callback=handle_progress,
        )
        last_train_accuracy = train_metrics.get("epoch_accuracy", 0.0)
        eval_stage_started_at = time.perf_counter()
        last_train_progress_print = 0
        train_stage_started_at = eval_stage_started_at
        eval_metrics = pipeline.evaluate(
            eval_samples,
            ticks_per_sample=args.ticks,
            expected_total=args.test_limit,
            progress_every=args.progress_every,
            progress_callback=handle_progress,
        )
        last_eval_accuracy = eval_metrics.get("eval_accuracy", 0.0)
        prediction_history.append(int(eval_metrics.get("prediction", -1.0)))

        diagnostics = diagnostics_for_step15(eval_metrics, prediction_history)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:09d}.h5"
        pipeline.save_state(str(checkpoint_path))
        last_checkpoint_path = checkpoint_path

        print(
            f"epoch={epoch + 1} "
            f"train_accuracy={train_metrics.get('epoch_accuracy', 0.0):.4f} "
            f"eval_accuracy={eval_metrics.get('eval_accuracy', 0.0):.4f}"
        )
        for line in diagnostics:
            print(f"diag={line}")

        append_metrics_line(
            metrics_path,
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": "epoch_end",
                "epoch": epoch + 1,
                "dataset_mode": dataset_mode,
                "train_limit": args.train_limit,
                "test_limit": args.test_limit,
                "ticks": args.ticks,
                "train": train_metrics,
                "eval": eval_metrics,
                "diagnostics": diagnostics,
                "checkpoint_path": str(checkpoint_path),
            },
        )
        (
            epoch_progress_ratio,
            epoch_samples_per_sec,
            epoch_eta_seconds,
        ) = compute_training_progress(epoch_index=epoch_index)
        write_metrics_snapshot(
            metrics_snapshot_path,
            exporter_snapshot_payload(
                snapshot_source="training_epoch_end",
                dataset_mode=dataset_mode,
                epoch=epoch_index,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
                ticks=args.ticks,
                checkpoint_path=checkpoint_path,
                runtime_metrics=eval_metrics,
                train_accuracy=train_metrics.get("epoch_accuracy", 0.0),
                test_accuracy=eval_metrics.get("eval_accuracy", 0.0),
                progress_stage="epoch_end",
                progress_completed=work_items_per_epoch,
                progress_total=work_items_per_epoch,
                training_progress_ratio=epoch_progress_ratio,
                training_samples_per_sec=epoch_samples_per_sec,
                training_eta_seconds=epoch_eta_seconds,
            ),
        )
        visualizer_trace = capture_trace_from_state(
            state_path=str(checkpoint_path),
            config_path=args.config,
            sample=visualizer_sample,
            ticks_per_sample=args.ticks,
        )
        write_metrics_snapshot(
            visualizer_trace_path,
            visualizer_trace_payload(
                trace_source="training_epoch_end",
                dataset_mode=dataset_mode,
                epoch=epoch_index,
                checkpoint_path=checkpoint_path,
                sample=visualizer_sample,
                trace=visualizer_trace,
            ),
        )
        print(f"visualizer_trace_saved={visualizer_trace_path}")

        if not args.no_early_stop and last_eval_accuracy >= args.target_accuracy:
            print(
                f"target_reached=true eval_accuracy={last_eval_accuracy:.4f} "
                f"target_accuracy={args.target_accuracy:.4f}"
            )
            break

    if last_checkpoint_path is None:
        raise RuntimeError("Training produced no checkpoints")

    state_path = Path(args.state_out)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_state(str(state_path))
    print(f"state_saved={state_path}")

    if not args.skip_robustness:
        robustness = robustness_report(
            state_path=str(last_checkpoint_path),
            config_path=args.config,
            sample_factory=lambda: make_samples(
                dataset_mode=dataset_mode,
                data_root=args.data_root,
                train=False,
                limit=args.test_limit,
                epoch_seed=999,
                download=False,
            ),
            ticks_per_sample=args.ticks,
            remove_fraction=args.robust_remove_fraction,
            noise_sigma=args.robust_noise_sigma,
        )
        print(
            "robustness "
            f"baseline={robustness['baseline_accuracy']:.4f} "
            f"pruned={robustness['pruned_accuracy']:.4f} "
            f"noise={robustness['noise_accuracy']:.4f} "
            f"prune_drop={robustness['prune_drop']:.4f} "
            f"noise_drop={robustness['noise_drop']:.4f}"
        )
        append_metrics_line(
            metrics_path,
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": "robustness",
                "checkpoint_path": str(last_checkpoint_path),
                "target_accuracy": args.target_accuracy,
                "last_eval_accuracy": last_eval_accuracy,
                "metrics": robustness,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
