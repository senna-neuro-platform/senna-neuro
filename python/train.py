from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from senna.training import (
    TrainingPipeline,
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
    if args.dataset != "mnist":
        return "synthetic"

    try:
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
    except Exception as exc:
        print(f"mnist_unavailable={exc}; fallback=synthetic")
        return "synthetic"


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
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    dataset_mode = resolve_dataset_mode(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)
    prediction_history: list[int] = []
    last_eval_accuracy = 0.0
    last_checkpoint_path: Path | None = None

    for epoch in range(args.epochs):
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

        train_metrics = pipeline.train_epoch(train_samples, ticks_per_sample=args.ticks)
        eval_metrics = pipeline.evaluate(eval_samples, ticks_per_sample=args.ticks)
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
