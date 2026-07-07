"""Stream multiple speech sources into ONE unified HF dataset, disk-safe.

Why: `load_dataset(...).save_to_disk()` downloads whole shards + a local copy and
fills Colab's disk. Instead we STREAM each source, map every row to a shared
schema, buffer `shard_size` rows, write a single Parquet shard, upload it to the
private HF dataset, and delete it. Peak disk stays ~one shard.

Adding a source = write one `<source>_stream(cfg, split)` generator that yields the
unified dict below and register it in SOURCES. curate_to_hub() runs every source
in `cfg["sources"]` into one corpus (shards tagged by source in the filename).
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Iterator

from corpus import macro_accent
from text_normalization import normalize_text

SPLITS = ("train", "validation", "test")
# AfriSpeech-200's loading script names the dev split "dev".
_AFRISPEECH_SPLIT = {"train": "train", "validation": "dev", "test": "test"}
_MAX_SEG_SEC = 30.0  # Whisper processes <= 30 s per clip

# The one schema every source must produce. Training reads these names only.
_UNIFIED_COLUMNS = (
    "audio", "text", "text_raw", "source", "language", "task",
    "accent", "macro_accent", "domain", "speaker_id", "gender",
    "age_group", "duration", "quality", "license",
)


def unified_features():
    from datasets import Audio, Features, Value

    return Features({
        "audio": Audio(),                 # native SR; resampled to 16k at train time
        "text": Value("string"),          # normalized transcript
        "text_raw": Value("string"),      # original transcript
        "source": Value("string"),
        "language": Value("string"),      # en / en-codeswitch / pcm / yor / hau / ibo
        "task": Value("string"),          # stt / tts
        "accent": Value("string"),
        "macro_accent": Value("string"),  # Yoruba / Igbo / Hausa / Other
        "domain": Value("string"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "age_group": Value("string"),
        "duration": Value("float32"),
        "quality": Value("string"),       # unrated / clean / noisy (for TTS filtering)
        "license": Value("string"),
    })


def _macro_map(cfg):
    return {k.lower(): v for k, v in cfg["macro_accent_map"].items()}


# --------------------------------------------------------------------------- #
# Source adapters
# --------------------------------------------------------------------------- #
def afrispeech_stream(cfg: dict, split: str) -> Iterator[dict]:
    """AfriSpeech-200 (read, utterance-level) Nigerian accents -> unified rows."""
    from datasets import Audio, load_dataset

    macro_map = _macro_map(cfg)
    src_split = _AFRISPEECH_SPLIT[split]
    # "all" (default) streams the whole corpus; country_filter keeps NG. The per-row
    # accent below still comes from ex["accent"], so macro-mapping is unaffected.
    configs = cfg.get("afrispeech_configs") or cfg["accents"]
    for cfg_name in configs:
        try:
            ds = load_dataset(cfg["hf_dataset_id"], cfg_name, split=src_split,
                              streaming=True, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(decode=False))  # keep bytes, don't decode
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] afrispeech-200 {cfg_name}/{src_split}: {e}")
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
                "source": "afrispeech-200",
                "language": "en",
                "task": "stt",
                "accent": raw,
                "macro_accent": macro_accent(raw, macro_map),
                "domain": ex.get("domain") or "unknown",
                "speaker_id": str(ex.get("user_id") or ex.get("user_ids") or ""),
                "gender": ex.get("gender") or "",
                "age_group": ex.get("age_group") or "",
                "duration": dur,
                "quality": "unrated",
                "license": "CC-BY-NC-SA-4.0",
            }


_TS_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$")
_SPK_RE = re.compile(r"^\s*\[Speaker\s*(\d+)\]:\s*(.*)$")


def parse_dialog_turns(transcript: str) -> list[tuple]:
    """Parse AfriSpeech-Dialog's timestamped transcript into utterance turns.

    Format is lines of `MM:SS:CC` timestamps around `[Speaker N]: text` lines.
    Returns [(start_sec, end_sec, speaker_num, text), ...]. Pure/testable.
    """
    def to_sec(m):
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0

    toks: list = []  # ["ts", secs] | ["spk", [num, text]]
    for ln in (transcript or "").splitlines():
        mts = _TS_RE.match(ln)
        if mts:
            toks.append(["ts", to_sec(mts)])
            continue
        msp = _SPK_RE.match(ln)
        if msp:
            toks.append(["spk", [int(msp.group(1)), msp.group(2).strip()]])
            continue
        if ln.strip() and toks and toks[-1][0] == "spk":  # continuation line
            toks[-1][1][1] = (toks[-1][1][1] + " " + ln.strip()).strip()

    turns = []
    for i, (kind, payload) in enumerate(toks):
        if kind != "spk":
            continue
        spk, text = payload
        start = next((toks[j][1] for j in range(i - 1, -1, -1) if toks[j][0] == "ts"), None)
        end = next((toks[j][1] for j in range(i + 1, len(toks)) if toks[j][0] == "ts"), None)
        if start is None:
            continue
        if end is None or end <= start:
            end = start + _MAX_SEG_SEC
        turns.append((start, end, spk, text))
    return turns


def afrispeech_dialog_stream(cfg: dict, split: str) -> Iterator[dict]:
    """AfriSpeech-Dialog (spontaneous conversations) -> segmented unified rows.

    Conversations are long; we slice each into utterance turns via the timestamped
    transcript. Only a `train` split exists upstream.
    """
    if split != "train":
        return
    from datasets import load_dataset

    macro_map = _macro_map(cfg)
    try:
        ds = load_dataset("intronhealth/afrispeech-dialog", split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] afrispeech-dialog: {e}")
        return

    for i, ex in enumerate(ds):
        if cfg.get("country_filter") and ex.get("country") != cfg["country_filter"]:
            continue
        audio = ex.get("audio") or {}
        arr, sr = audio.get("array"), audio.get("sampling_rate")
        if arr is None or not sr:
            continue
        raw = (ex.get("accent") or "").lower()
        for start, end, spk, text_raw in parse_dialog_turns(ex.get("transcript") or ""):
            if not text_raw.strip():
                continue
            end = min(end, start + _MAX_SEG_SEC)
            seg = arr[int(start * sr):int(end * sr)]
            if len(seg) < int(0.3 * sr):  # skip sub-0.3s fragments
                continue
            yield {
                "audio": {"array": seg, "sampling_rate": sr},
                "text": normalize_text(text_raw, cfg.get("lowercase", True)),
                "text_raw": text_raw,
                "source": "afrispeech-dialog",
                "language": "en",
                "task": "stt",
                "accent": raw,
                "macro_accent": macro_accent(raw, macro_map),
                "domain": "conversational",
                "speaker_id": f"afdialog-{i}-spk{spk}",
                "gender": "",
                "age_group": ex.get("age_group") or "",
                "duration": float(end - start),
                "quality": "unrated",
                "license": "CC-BY-NC-SA-4.0",
            }


# Register sources here. Add e.g. "common-voice": common_voice_stream later.
SOURCES = {
    "afrispeech-200": afrispeech_stream,
    "afrispeech-dialog": afrispeech_dialog_stream,
}


# --------------------------------------------------------------------------- #
# Curation driver + loader
# --------------------------------------------------------------------------- #
def load_curated(repo: str, split: str | None = None):
    """Load the curated HF dataset (all splits, or one) with audio at 16 kHz."""
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, split=split)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _flush_shard(records, tag, idx, repo, api, features):
    from datasets import Dataset

    ds = Dataset.from_list(records, features=features)
    path = os.path.join(tempfile.gettempdir(), f"{tag}-{idx:05d}.parquet")
    ds.to_parquet(path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=f"data/{tag}-{idx:05d}.parquet",
        repo_id=repo, repo_type="dataset",
    )
    os.remove(path)
    print(f"  [curate] uploaded data/{tag}-{idx:05d}.parquet ({len(records)} rows)")


def curate_to_hub(cfg: dict, repo: str, shard_size: int = 500,
                  limit: int | None = None, verify: bool = True) -> dict:
    """Stream every source in cfg['sources'] into ONE private HF dataset, disk-safe.

    Shards are named `data/<split>-<source>-<idx>.parquet` so HF infers the split
    and sources coexist in one corpus. `limit` caps rows per (split, source) for
    a cheap smoke run.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo

    sources = cfg.get("sources") or [cfg.get("source")]
    features = unified_features()
    api = HfApi()
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    try:  # start clean so re-runs don't leave stale shards
        api.delete_folder("data", repo_id=repo, repo_type="dataset")
    except Exception:  # noqa: BLE001
        pass

    totals = {}
    for split in SPLITS:
        n_split = 0
        for src in sources:
            stream_fn = SOURCES.get(src)
            if stream_fn is None:
                print(f"  [warn] unknown source '{src}'")
                continue
            buf, idx, n = [], 0, 0
            for rec in stream_fn(cfg, split):
                buf.append(rec)
                n += 1
                if len(buf) >= shard_size:
                    _flush_shard(buf, f"{split}-{src}", idx, repo, api, features)
                    idx, buf = idx + 1, []
                if limit and n >= limit:
                    break
            if buf:
                _flush_shard(buf, f"{split}-{src}", idx, repo, api, features)
                idx += 1
            if n:
                print(f"[curate] {split}/{src}: {n} clips in {idx} shard(s)")
            n_split += n
        totals[split] = n_split

    print(f"[curate] done -> https://huggingface.co/datasets/{repo}")
    if verify and any(totals.values()):
        first = next(s for s in SPLITS if totals.get(s))
        ex = next(iter(load_dataset(repo, split=first, streaming=True)))
        ok = isinstance(ex.get("audio"), dict) and ex["audio"].get("array") is not None
        print(f"[curate] verify audio decodes on '{first}': {'OK' if ok else 'FAILED'}")
    return totals
