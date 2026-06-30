"""Tiny configuration helpers: load YAML configs and a .env file.

Kept dependency-light on purpose (only PyYAML). We avoid python-dotenv so the
data/metrics modules can run with a minimal install.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: sets os.environ for KEY=VALUE lines.

    Existing environment variables are not overwritten. Lines starting with '#'
    and blank lines are ignored. No quoting/expansion magic — keep it simple.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
