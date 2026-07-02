"""Stream multiple speech sources into ONE unified HF dataset, disk-safe.

Why: `load_dataset(...).save_to_disk()` downloads whole shards + a local copy and
fills Colab's disk. Instead we STREAM each source, map every row to a shared
schema, buffer `shard_size` rows, write a single Parquet shard, upload it to the
private HF dataset, and delete it. Peak disk stays ~one shard.

Adding a source = write one generator that yields the unified dict below and
register it in SOURCES. AfriSpeech-200 is the first; others slot in the same way.
"""
from __future__ import annotations

import os
import tempfile
from typing import Iterator

from corpus import macro_accent
from text_normalization import normalize_text

SPLITS = ("train", "validation", "test")
# AfriSpeech-200's loading script names the dev split "dev".
_AFRISPEECH_SPLIT = {"train": "train", "validation": "dev", "test": "test"}

# The one schema every source must produce. Training reads these names only.
_UNIFIED_COLUMNS = (
    "audio", "text", "text_raw", "source", "accent", "macro_accent",
    "domain", "speaker_id", "gender", "age_group", "duration", "license",
)


def unified_features():
    from datasets import Audio, Features, Value

    return Features({
        "audio": Audio(),                 # native SR bytes; resampled to 16k at train time
        "text": Value("string"),          # normalized transcript
        "text_raw": Value("string"),      # original transcript
        "source": Value("string"),
        "accent": Value("string"),
        "macro_accent": Value("string"),  # Yoruba / Igbo / Hausa / Other
        "domain": Value("string"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "age_group": Value("string"),
        "duration": Value("float32"),
        "license": Value("string"),
    })


def afrispeech_stream(cfg: dict, split: str) -> Iterator[dict]:
    """Stream AfriSpeech-200 Nigerian accents for one split -> unified rows."""
    from datasets import Audio, load_dataset

    macro_map = {k.lower(): v for k, v in cfg["macro_accent_map"].items()}
    src_split = _AFRISPEECH_SPLIT[split]
    for accent in cfg["accents"]:
        try:
            ds = load_dataset(cfg["hf_dataset_id"], accent, split=src_split,
                              streaming=True, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(decode=False))  # keep bytes, don't decode
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {accent}/{src_split}: {e}")
            continue
        for ex in ds:
            if cfg.get("country_filter") and ex.get("country") != cfg["country_filter"]:
                continue
            raw = (ex.get("accent") or "").lower()
            text_raw = ex.get("transcript") or ""
            try:
                dur = float(ex.get("duration"))
            except (TypeError, ValueError):
                dur = -1.0
            yield {
                "audio": ex["audio"],  # {"bytes": ..., "path": ...}
                "text": normalize_text(text_raw, cfg.get("lowercase", True)),
                "text_raw": text_raw,
                "source": cfg["source"],
                "accent": raw,
                "macro_accent": macro_accent(raw, macro_map),
                "domain": ex.get("domain") or "unknown",
                "speaker_id": str(ex.get("user_id") or ex.get("user_ids") or ""),
                "gender": ex.get("gender") or "",
                "age_group": ex.get("age_group") or "",
                "duration": dur,
                "license": cfg["license"],
            }


# Register sources here. Add e.g. "common_voice": common_voice_stream later.
SOURCES = {"afrispeech-200": afrispeech_stream}


def load_curated(repo: str, split: str | None = None):
    """Load the curated HF dataset (all splits, or one) with audio at 16 kHz."""
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, split=split)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _flush_shard(records, split, idx, repo, api, features):
    from datasets import Dataset

    ds = Dataset.from_list(records, features=features)
    path = os.path.join(tempfile.gettempdir(), f"{split}-{idx:05d}.parquet")
    ds.to_parquet(path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"data/{split}-{idx:05d}.parquet",
        repo_id=repo, repo_type="dataset",
    )
    os.remove(path)
    print(f"  [curate] uploaded data/{split}-{idx:05d}.parquet ({len(records)} rows)")


def curate_to_hub(cfg: dict, repo: str, shard_size: int = 500,
                  limit: int | None = None, verify: bool = True) -> dict:
    """Stream the configured source into a private HF dataset, one shard at a time.

    `limit` caps rows PER SPLIT (for a cheap smoke run).
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo

    features = unified_features()
    api = HfApi()
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    try:  # start clean so re-runs don't leave stale shards
        api.delete_folder("data", repo_id=repo, repo_type="dataset")
    except Exception:  # noqa: BLE001
        pass

    stream_fn = SOURCES[cfg["source"]]
    totals = {}
    for split in SPLITS:
        buf, idx, n = [], 0, 0
        for rec in stream_fn(cfg, split):
            buf.append(rec)
            n += 1
            if len(buf) >= shard_size:
                _flush_shard(buf, split, idx, repo, api, features)
                idx, buf = idx + 1, []
            if limit and n >= limit:
                break
        if buf:
            _flush_shard(buf, split, idx, repo, api, features)
            idx += 1
        totals[split] = n
        print(f"[curate] {split}: {n} clips in {idx} shard(s)")

    print(f"[curate] done -> https://huggingface.co/datasets/{repo}")
    if verify and any(totals.values()):
        first = next(s for s in SPLITS if totals.get(s))
        ex = next(iter(load_dataset(repo, split=first, streaming=True)))
        ok = isinstance(ex.get("audio"), dict) and ex["audio"].get("array") is not None
        print(f"[curate] verify audio decodes on '{first}': {'OK' if ok else 'FAILED'}")
    return totals
