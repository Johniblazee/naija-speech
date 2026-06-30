"""Weights & Biases tracking helpers (optional; no-op without WANDB_API_KEY)."""
from __future__ import annotations

import os
from typing import Any


def _project(project: str | None) -> str:
    return project or os.environ.get("WANDB_PROJECT", "naija-speech")


def maybe_log_wandb(results: list[dict], params: dict[str, Any],
                    project: str | None = None, name: str | None = None):
    """Log stratified WER/CER to W&B if WANDB_API_KEY is set; else no-op."""
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set — skipping logging.")
        return None
    import wandb

    run = wandb.init(project=_project(project), name=name, config=params)
    for r in results:
        if not r.get("n"):
            continue
        tag = f"{r['group']}/{r['key']}"
        run.log({f"WER/{tag}": r["wer"], f"CER/{tag}": r["cer"]})
    run.finish()
    print("[wandb] logged run.")
    return run


def maybe_log_wandb_tables(tables: dict, project: str | None = None, name: str | None = None):
    """Log EDA summary tables (dict of DataFrames) to W&B; else no-op."""
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set — skipping EDA logging.")
        return None
    import wandb

    run = wandb.init(project=_project(project), name=name or "eda", job_type="eda")
    for key, df in tables.items():
        run.log({f"eda/{key}": wandb.Table(dataframe=df.reset_index())})
    run.finish()
    print("[wandb] logged EDA tables.")
    return run
