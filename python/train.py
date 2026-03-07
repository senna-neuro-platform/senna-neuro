from __future__ import annotations

import argparse
from pathlib import Path

from senna.training import (
    TrainingPipeline,
    iter_mnist_samples,
    make_synthetic_digit_samples,
)


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
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2000,
        help="Number of train samples per epoch",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=1000,
        help="Number of test samples per epoch",
    )
    parser.add_argument("--ticks", type=int, default=100, help="Ticks per sample")
    parser.add_argument(
        "--state-out",
        default="data/artifacts/outbox/train_state.h5",
        help="Path to state file",
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)

    for epoch in range(args.epochs):
        if args.dataset == "mnist":
            try:
                train_samples = iter_mnist_samples(
                    root=args.data_root,
                    train=True,
                    limit=args.train_limit,
                    download=args.download,
                )
                eval_samples = iter_mnist_samples(
                    root=args.data_root,
                    train=False,
                    limit=args.test_limit,
                    download=False,
                )
            except RuntimeError as exc:
                print(f"mnist_unavailable={exc}; fallback=synthetic")
                train_samples = make_synthetic_digit_samples(
                    args.train_limit, seed=42 + epoch
                )
                eval_samples = make_synthetic_digit_samples(
                    args.test_limit, seed=142 + epoch
                )
        else:
            train_samples = make_synthetic_digit_samples(
                args.train_limit, seed=42 + epoch
            )
            eval_samples = make_synthetic_digit_samples(
                args.test_limit, seed=142 + epoch
            )

        train_metrics = pipeline.train_epoch(train_samples, ticks_per_sample=args.ticks)
        eval_metrics = pipeline.evaluate(eval_samples, ticks_per_sample=args.ticks)
        print(
            f"epoch={epoch + 1} "
            f"train_accuracy={train_metrics.get('epoch_accuracy', 0.0):.4f} "
            f"eval_accuracy={eval_metrics.get('eval_accuracy', 0.0):.4f}"
        )

    state_path = Path(args.state_out)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_state(str(state_path))
    print(f"state_saved={state_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
