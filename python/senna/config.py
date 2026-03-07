from __future__ import annotations

from pathlib import Path


def resolve_config_path(path: str | None = None) -> Path:
    if path:
        return Path(path)
    return Path("configs/default.yaml")
