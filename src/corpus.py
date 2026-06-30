"""Build the Nigerian-accented English corpus from AfriSpeech-200.

Implements thesis §3.3 (corpus construction) for the STT vertical slice:
- load AfriSpeech-200 by per-accent config (efficient; only fetches the accents
  we ask for),
- keep only Nigerian (country == NG) clips within the duration bounds,
- standardize to a fixed schema, normalize transcripts (keeping Nigerian lexis),
- map raw accent -> macro-accent (Yoruba / Igbo / Hausa / Other),
- save a HuggingFace DatasetDict to disk + a human-readable manifest.csv,
- verify the train/val/test splits are speaker-disjoint.

Requires the `datasets` library (and audio backends). Heavy parts run on
Colab/RunPod, not on a laptop.
"""
from __future__ import annotations

import itertools
import os
from collections import Counter
from typing import Any

from text_normalization import normalize_text

# Splits as named by the AfriSpeech-200 loading script (dev -> validation).
_SPLITS = ("train", "validation", "test")


def macro_accent(raw_accent: str, macro_map: dict[str, str]) -> str:
    """Map a raw accent string to the thesis macro-accent, else 'Other'."""
    return macro_map.get((raw_accent or "").lower(), "Other")


def _standardize(ds, cfg: dict[str, Any]):
    """Project an AfriSpeech split onto our fixed schema (audio kept, not decoded)."""
    macro_map = {k.lower(): v for k, v in cfg["macro_accent_map"].items()}
    lowercase = cfg.get("lowercase", True)
    source = cfg["source"]
    license_ = cfg["license"]
    original_cols = ds.column_names

    def fn(ex):
        raw_accent = (ex.get("accent") or "").lower()
        transcript = ex.get("transcript") or ""
        duration = ex.get("duration")
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = -1.0
        return {
            "clip_id": str(ex.get("audio_id") or ex.get("audio_ids") or ex.get("path") or ""),
            "speaker_id": str(ex.get("user_id") or ex.get("user_ids") or ""),
            "transcript_raw": transcript,
            "transcript_norm": normalize_text(transcript, lowercase),
            "accent_raw": raw_accent,
            "macro_accent": macro_accent(raw_accent, macro_map),
            "domain": ex.get("domain") or "unknown",
            "country": ex.get("country") or "",
            "duration": duration,
            "age_group": ex.get("age_group") or "",
            "gender": ex.get("gender") or "",
            "dataset": source,
            "license": license_,
        }

    keep = "audio"
    remove = [c for c in original_cols if c != keep]
    return ds.map(fn, remove_columns=remove, desc="standardize")


def _keep_row(country: str, duration: float, cfg: dict[str, Any]) -> bool:
    if cfg.get("country_filter") and country != cfg["country_filter"]:
        return False
    if duration is not None and duration >= 0:
        if duration < cfg["min_duration_sec"] or duration > cfg["max_duration_sec"]:
            return False
    return True


def build_corpus(cfg: dict[str, Any], max_per_split: int | None = None,
                 push_repo: str | None = None):
    """Build and persist the corpus. Returns the saved DatasetDict.

    `max_per_split` caps examples per accent/split for fast iteration (note: the
    full accent shards are still downloaded by `load_dataset`).
    `push_repo` (e.g. "user/name") also pushes the curated DatasetDict to a
    private HF dataset for streaming into training.
    """
    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

    parts: dict[str, list] = {s: [] for s in _SPLITS}

    for accent in cfg["accents"]:
        print(f"[corpus] loading accent config: {accent}")
        # AfriSpeech-200 ships a loading script -> trust_remote_code required.
        dd = load_dataset(cfg["hf_dataset_id"], accent, trust_remote_code=True)
        for split in dd:
            if split not in parts:
                continue
            ds = dd[split]
            if max_per_split:
                ds = ds.select(range(min(max_per_split, ds.num_rows)))
            # Don't decode audio while we filter/standardize on metadata.
            ds = ds.cast_column("audio", Audio(decode=False))
            ds = _standardize(ds, cfg)
            ds = ds.filter(
                lambda c, d: _keep_row(c, d, cfg),
                input_columns=["country", "duration"],
                desc="filter NG + duration",
            )
            if ds.num_rows == 0:
                continue
            ds = ds.add_column("split", [split] * ds.num_rows)
            # Decode at the target sample rate for downstream training.
            ds = ds.cast_column("audio", Audio(sampling_rate=cfg["target_sample_rate"]))
            parts[split].append(ds)

    out = DatasetDict()
    for split, plist in parts.items():
        if plist:
            out[split] = concatenate_datasets(plist)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    out.save_to_disk(output_dir)
    _write_manifest(out, output_dir)
    if push_repo:
        print(f"[corpus] pushing to private HF dataset: {push_repo}")
        out.push_to_hub(push_repo, private=True)
    return out


def _write_manifest(dsd, output_dir: str):
    import pandas as pd

    frames = []
    for split, ds in dsd.items():
        df = ds.remove_columns("audio").to_pandas()
        frames.append(df)
    if frames:
        manifest = pd.concat(frames, ignore_index=True)
        path = os.path.join(output_dir, "manifest.csv")
        manifest.to_csv(path, index=False)
        print(f"[corpus] wrote manifest: {path} ({len(manifest)} rows)")


def check_speaker_disjoint(dsd) -> dict[str, set]:
    """Return overlapping speaker_ids between split pairs (empty == good)."""
    speakers = {s: set(dsd[s]["speaker_id"]) for s in dsd}
    overlaps = {}
    names = list(speakers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = speakers[a] & speakers[b]
            overlaps[f"{a}∩{b}"] = inter
    return overlaps


def summarize(dsd) -> dict[str, Any]:
    """Counts per split, macro-accent, and domain — for sanity + thesis tables."""
    summary: dict[str, Any] = {}
    for split, ds in dsd.items():
        summary[split] = {
            "n": ds.num_rows,
            "macro_accent": dict(Counter(ds["macro_accent"])),
            "domain": dict(Counter(ds["domain"])),
            "speakers": len(set(ds["speaker_id"])),
        }
    return summary


def peek_streaming(cfg: dict[str, Any], n: int = 25) -> list[dict]:
    """Stream a few rows per accent (no full download) to sanity-check filtering."""
    from datasets import Audio, load_dataset

    macro_map = {k.lower(): v for k, v in cfg["macro_accent_map"].items()}
    rows: list[dict] = []
    for accent in cfg["accents"]:
        try:
            ds = load_dataset(
                cfg["hf_dataset_id"], accent, split="train",
                streaming=True, trust_remote_code=True,
            )
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not stream {accent}: {e}")
            continue
        for ex in itertools.islice(ds, n):
            country = ex.get("country", "")
            if cfg.get("country_filter") and country != cfg["country_filter"]:
                continue
            raw = (ex.get("accent") or "").lower()
            rows.append(
                {
                    "accent_raw": raw,
                    "macro_accent": macro_accent(raw, macro_map),
                    "domain": ex.get("domain"),
                    "country": country,
                    "speaker_id": ex.get("user_id") or ex.get("user_ids"),
                    "transcript": ex.get("transcript"),
                }
            )
    return rows
