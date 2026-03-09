#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pytest", "pytest"),
    ("ruff", "ruff"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
)


def missing_imports(target_dir: Path) -> list[tuple[str, str]]:
    sys.path.insert(0, str(target_dir))
    missing: list[tuple[str, str]] = []
    for import_name, package_name in PROJECT_PACKAGES:
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append((import_name, package_name))
    return missing


def install_missing(target_dir: Path, missing: list[tuple[str, str]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    package_specs = [package_name for _, package_name in missing]
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target_dir),
        *package_specs,
    ]
    print(f"[python-setup] installing into {target_dir}: {' '.join(package_specs)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure project-local Python packages exist in .python-packages."
    )
    parser.add_argument(
        "--target",
        default=".python-packages",
        help="Target directory for project-local Python packages",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify imports, do not install missing packages",
    )
    args = parser.parse_args()

    target_dir = Path(args.target).resolve()
    missing = missing_imports(target_dir)
    if not missing:
        print(f"[python-setup] all required packages are present in {target_dir}")
        return 0

    missing_text = ", ".join(import_name for import_name, _ in missing)
    if args.check_only:
        print(
            f"[python-setup] missing required packages in {target_dir}: {missing_text}",
            file=sys.stderr,
        )
        return 1

    install_missing(target_dir, missing)
    still_missing = missing_imports(target_dir)
    if still_missing:
        still_missing_text = ", ".join(import_name for import_name, _ in still_missing)
        print(
            f"[python-setup] package setup incomplete, still missing: {still_missing_text}",
            file=sys.stderr,
        )
        return 1

    print(f"[python-setup] package setup completed in {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
