"""Accuracy metrics: Word Error Rate (WER) and Character Error Rate (CER).

We use `jiwer` (the de-facto reference implementation cited in the thesis) and
add the thesis's distinguishing feature: **accent-aware**, stratified reporting.
Aggregate WER is never reported alone — we break it down by macro-accent and by
domain so a model that looks fine on average but fails on one accent is exposed.

Pure-Python (jiwer only) so it can be unit-tested without torch/datasets.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import jiwer

from text_normalization import normalize_text


def compute_wer_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize: bool = True,
    lowercase: bool = True,
) -> dict:
    """Compute WER and CER over aligned reference/hypothesis lists.

    Pairs whose reference is empty after normalization are dropped (jiwer cannot
    score an empty reference). Returns counts so groups can be re-aggregated.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) "
            "must be the same length"
        )

    refs, hyps = [], []
    for ref, hyp in zip(references, hypotheses):
        r = normalize_text(ref, lowercase) if normalize else (ref or "")
        h = normalize_text(hyp, lowercase) if normalize else (hyp or "")
        if not r:  # unscorable reference
            continue
        refs.append(r)
        hyps.append(h)

    n = len(refs)
    if n == 0:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}

    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "n": n,
    }


def compute_stratified(
    rows: Iterable[dict],
    ref_key: str = "reference",
    hyp_key: str = "hypothesis",
    group_keys: Sequence[str] = ("macro_accent", "domain"),
    normalize: bool = True,
    lowercase: bool = True,
) -> list[dict]:
    """Return overall + per-group WER/CER.

    `rows` is an iterable of dicts each containing the reference, the hypothesis,
    and the grouping fields. Output is a list of result dicts (easy to turn into
    a pandas DataFrame / CSV for Chapter 5 tables):

        [{"group": "overall", "key": "all", "wer": ..., "cer": ..., "n": ...},
         {"group": "macro_accent", "key": "Yoruba", "wer": ..., ...},
         {"group": "domain", "key": "clinical", "wer": ..., ...}, ...]
    """
    rows = list(rows)
    results: list[dict] = []

    overall = compute_wer_cer(
        [r[ref_key] for r in rows], [r[hyp_key] for r in rows], normalize, lowercase
    )
    results.append({"group": "overall", "key": "all", **overall})

    for gk in group_keys:
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(gk, "unknown")), []).append(r)
        for key in sorted(buckets):
            sub = buckets[key]
            m = compute_wer_cer(
                [r[ref_key] for r in sub], [r[hyp_key] for r in sub], normalize, lowercase
            )
            results.append({"group": gk, "key": key, **m})

    return results
