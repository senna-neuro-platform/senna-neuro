#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clang-tidy for every project translation unit in compile_commands.json."
    )
    parser.add_argument(
        "--build-dir",
        default="build/debug",
        help="Build directory that contains compile_commands.json (default: build/debug).",
    )
    parser.add_argument(
        "--clang-tidy",
        default="clang-tidy",
        help="clang-tidy executable to use (default: clang-tidy).",
    )
    parser.add_argument(
        "--warnings-as-errors",
        default="*",
        help="Value passed to --warnings-as-errors (default: *).",
    )
    return parser.parse_args()


def load_translation_units(source_root: Path, build_dir: Path) -> list[Path]:
    compile_commands_path = build_dir / "compile_commands.json"
    if not compile_commands_path.is_file():
        raise FileNotFoundError(
            f"compile_commands.json not found: {compile_commands_path}"
        )

    with compile_commands_path.open(encoding="utf-8") as handle:
        compile_commands = json.load(handle)

    translation_units: list[Path] = []
    for entry in compile_commands:
        file_path = Path(entry["file"]).resolve()
        try:
            relative_path = file_path.relative_to(source_root)
        except ValueError:
            continue

        if relative_path.suffix != ".cpp":
            continue

        if not (
            str(relative_path).startswith("src/")
            or str(relative_path).startswith("tests/")
        ):
            continue

        translation_units.append(relative_path)

    return sorted(dict.fromkeys(translation_units))


def run_clang_tidy(
    source_root: Path, build_dir: Path, clang_tidy_bin: str, warnings_as_errors: str
) -> int:
    translation_units = load_translation_units(source_root, build_dir)
    if not translation_units:
        print("No translation units found in compile_commands.json", file=sys.stderr)
        return 1

    failures = 0
    for translation_unit in translation_units:
        print(f"[clang-tidy] {translation_unit}", flush=True)
        command = [
            clang_tidy_bin,
            str(translation_unit),
            "-p",
            str(build_dir),
            "--quiet",
            f"--warnings-as-errors={warnings_as_errors}",
        ]
        result = subprocess.run(
            command,
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        stderr_output = result.stderr
        if result.returncode == 0 and stderr_output:
            stderr_output = "\n".join(
                line
                for line in stderr_output.splitlines()
                if not re.fullmatch(r"\d+ warnings generated\.", line)
            )
            if stderr_output:
                stderr_output += "\n"
        if stderr_output:
            print(stderr_output, end="", file=sys.stderr)
        if result.returncode != 0:
            failures += 1

    if failures != 0:
        print(f"clang-tidy failed for {failures} translation unit(s)", file=sys.stderr)
        return 1

    print(f"clang-tidy passed for {len(translation_units)} translation unit(s)")
    return 0


def main() -> int:
    args = parse_args()
    source_root = Path(__file__).resolve().parent.parent
    build_dir = (source_root / args.build_dir).resolve()
    try:
        return run_clang_tidy(
            source_root=source_root,
            build_dir=build_dir,
            clang_tidy_bin=args.clang_tidy,
            warnings_as_errors=args.warnings_as_errors,
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
