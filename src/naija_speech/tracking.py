"""Comet experiment-tracking helpers (optional; no-op without an API key)."""
from __future__ import annotations

import os
from typing import Any


def maybe_log_comet(
    results: list[dict],
    params: dict[str, Any],
    project: str | None = None,
    name: str | None = None,
):
    """Log stratified WER/CER results to Comet if COMET_API_KEY is set.

    Safe to call unconditionally: prints and returns None when no key is present
    or comet_ml is unavailable.
    """
    if not os.environ.get("COMET_API_KEY"):
        print("[comet] COMET_API_KEY not set — skipping logging.")
        return None
    try:
        import comet_ml

        exp = comet_ml.Experiment(
            project_name=project or os.environ.get("COMET_PROJECT_NAME", "naija-speech-stt"),
        )
        if name:
            exp.set_name(name)
        exp.log_parameters(params)
        for r in results:
            if not r.get("n"):
                continue
            tag = f"{r['group']}/{r['key']}"
            exp.log_metric(f"WER/{tag}", r["wer"])
            exp.log_metric(f"CER/{tag}", r["cer"])
        exp.end()
        print("[comet] logged experiment.")
        return exp
    except Exception as e:  # noqa: BLE001
        print(f"[comet] logging failed: {e}")
        return None
