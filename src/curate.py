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

import io
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
def afrispeech_units(cfg: dict, split: str) -> list[str]:
    """Resumable units for AfriSpeech-200 = the HF configs we stream, one at a time.

    - `afrispeech_configs` set in cfg  -> stream exactly those (e.g. [igbo, yoruba, hausa]
      for a fast run, or [all] for one complete-but-not-resumable pass).
    - unset (default)                  -> every per-accent config (excluding the aggregate
      "all"), so each accent is its own checkpoint. Coverage == "all", but resumable.
    """
    configs = cfg.get("afrispeech_configs")
    if configs:
        return list(configs)
    try:
        from datasets import get_dataset_config_names

        names = get_dataset_config_names(cfg["hf_dataset_id"], trust_remote_code=True)
        return [n for n in names if n != "all"]
    except Exception as e:  # noqa: BLE001 — fall back to the thesis macro-accents
        print(f"  [warn] could not list afrispeech configs ({e}); using cfg['accents']")
        return list(cfg["accents"])


def afrispeech_stream(cfg: dict, split: str, unit: str) -> Iterator[dict]:
    """Stream ONE AfriSpeech-200 config (`unit`) -> unified Nigerian rows."""
    from datasets import Audio, load_dataset

    macro_map = _macro_map(cfg)
    src_split = _AFRISPEECH_SPLIT[split]
    try:
        ds = load_dataset(cfg["hf_dataset_id"], unit, split=src_split,
                          streaming=True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(decode=False))  # keep bytes, don't decode
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] afrispeech-200 {unit}/{src_split}: {e}")
        return
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


def afrispeech_dialog_units(cfg: dict, split: str) -> list[str]:
    """One resumable unit; only a `train` split exists upstream."""
    return ["dialog"] if split == "train" else []


def afrispeech_dialog_stream(cfg: dict, split: str, unit: str = "dialog") -> Iterator[dict]:
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


# Register sources here as (units_fn, stream_fn). units_fn lists resumable checkpoint
# units (streamed one at a time); stream_fn(cfg, split, unit) yields that unit's rows.
# Add e.g. "common-voice": (common_voice_units, common_voice_stream) later.
SOURCES = {
    "afrispeech-200": (afrispeech_units, afrispeech_stream),
    "afrispeech-dialog": (afrispeech_dialog_units, afrispeech_dialog_stream),
}


# --------------------------------------------------------------------------- #
# Curation driver + loader
# --------------------------------------------------------------------------- #
def load_curated(repo: str, split: str | None = None):
    """Load the curated HF dataset (all splits, or one) with audio at 16 kHz."""
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, split=split)
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _sanitize(name: str) -> str:
    """Make a config/unit name safe for a shard filename."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-").lower() or "x"


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


def _mark_done(api, repo, key):
    """Write a tiny checkpoint marker so --resume can skip this unit next time."""
    api.upload_file(
        path_or_fileobj=io.BytesIO(b"done"),
        path_in_repo=key, repo_id=repo, repo_type="dataset",
    )


def curate_to_hub(cfg: dict, repo: str, shard_size: int = 500,
                  limit: int | None = None, verify: bool = True,
                  resume: bool = False) -> dict:
    """Stream every source in cfg['sources'] into ONE private HF dataset, disk-safe.

    Each source exposes resumable *units* (AfriSpeech-200 = one per accent config;
    Dialog = one). A unit is streamed, sharded to
    `data/<split>-<src>-<unit>-<idx>.parquet`, then a marker
    `_checkpoints/<split>-<src>-<unit>.done` is written.

    resume=False (default): wipe data/ + _checkpoints/ and start clean.
    resume=True: keep what's there and skip units whose .done marker exists — so a run
    killed by a network drop continues where it left off. Completed units are never
    re-downloaded; a unit interrupted mid-stream is simply re-done from scratch.
    `limit` caps rows per (split, source) for a cheap smoke run (no markers written).

    Workflow: first run WITHOUT resume (clean start), then re-run WITH resume after
    any interruption until it completes.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo

    sources = cfg.get("sources") or [cfg.get("source")]
    features = unified_features()
    api = HfApi()
    create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    if not resume:  # clean start: drop stale shards AND stale checkpoints
        for folder in ("data", "_checkpoints"):
            try:
                api.delete_folder(folder, repo_id=repo, repo_type="dataset")
            except Exception:  # noqa: BLE001
                pass

    done = set()
    if resume:
        try:
            done = {f for f in api.list_repo_files(repo, repo_type="dataset")
                    if f.startswith("_checkpoints/") and f.endswith(".done")}
        except Exception:  # noqa: BLE001
            pass

    totals = {}
    for split in SPLITS:
        n_split = 0
        for src in sources:
            entry = SOURCES.get(src)
            if entry is None:
                print(f"  [warn] unknown source '{src}'")
                continue
            units_fn, stream_fn = entry
            n_src = 0
            for unit in units_fn(cfg, split):
                key = f"_checkpoints/{split}-{src}-{_sanitize(unit)}.done"
                if key in done:
                    print(f"  [resume] skip done: {split}/{src}/{unit}")
                    continue
                tag = f"{split}-{src}-{_sanitize(unit)}"
                buf, idx, n = [], 0, 0
                for rec in stream_fn(cfg, split, unit):
                    buf.append(rec)
                    n += 1
                    if len(buf) >= shard_size:
                        _flush_shard(buf, tag, idx, repo, api, features)
                        idx, buf = idx + 1, []
                    if limit and (n_src + n) >= limit:
                        break
                if buf:
                    _flush_shard(buf, tag, idx, repo, api, features)
                    idx += 1
                if limit is None:  # only checkpoint real (uncapped) runs
                    _mark_done(api, repo, key)
                if n:
                    print(f"[curate] {split}/{src}/{unit}: {n} clips in {idx} shard(s)")
                n_src += n
                if limit and n_src >= limit:
                    break
            n_split += n_src
        totals[split] = n_split

    print(f"[curate] done -> https://huggingface.co/datasets/{repo}")
    if verify and any(totals.values()):
        first = next(s for s in SPLITS if totals.get(s))
        ex = next(iter(load_dataset(repo, split=first, streaming=True)))
        ok = isinstance(ex.get("audio"), dict) and ex["audio"].get("array") is not None
        print(f"[curate] verify audio decodes on '{first}': {'OK' if ok else 'FAILED'}")
    return totals
