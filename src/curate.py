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

import json
import os
import re
import tempfile
import time
from typing import Iterator

from corpus import macro_accent
from text_normalization import normalize_text

SPLITS = ("train", "validation", "test")
# AfriSpeech-200 configs name their splits train / validation / test (NOT "dev").
_AFRISPEECH_SPLIT = {"train": "train", "validation": "validation", "test": "test"}
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


def _write_shard(records, tag, idx, features):
    """Write one shard to a local temp parquet; return (path_in_repo, local_path, bytes)."""
    from datasets import Dataset

    ds = Dataset.from_list(records, features=features)
    path = os.path.join(tempfile.gettempdir(), f"{tag}-{idx:05d}.parquet")
    ds.to_parquet(path)
    return f"data/{tag}-{idx:05d}.parquet", path, os.path.getsize(path)


def _retry_after_seconds(err, default=60):
    """Seconds to wait after a 429, from the Retry-After header or the message text."""
    resp = getattr(err, "response", None)
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra and str(ra).isdigit():
            return int(ra) + 1
    m = re.search(r"[Rr]etry after (\d+)", str(err))
    return int(m.group(1)) + 1 if m else default


def _commit(api, repo, ops, message, max_retries=6):
    """Commit many files in ONE request; retry on 429 honouring Retry-After.

    Batching files into few commits is what keeps us under HF's 128-commits/hour cap;
    the backoff is the safety net if a burst still trips it.
    """
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(max_retries):
        try:
            api.create_commit(repo_id=repo, repo_type="dataset",
                              operations=ops, commit_message=message)
            return
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429 and attempt < max_retries - 1:
                wait = _retry_after_seconds(e)
                print(f"  [rate-limit] 429 — sleeping {wait}s then retrying (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            raise


def curate_to_hub(cfg: dict, repo: str, shard_size: int = 500,
                  limit: int | None = None, verify: bool = True,
                  resume: bool = False, batch_bytes: int = 700_000_000,
                  batch_units: int = 15) -> dict:
    """Stream every source in cfg['sources'] into ONE private HF dataset, disk-safe.

    Each source exposes resumable *units* (AfriSpeech-200 = one per accent config;
    Dialog = one). Units are streamed and sharded to
    `data/<split>-<src>-<unit>-<idx>.parquet`; progress is recorded in
    `_checkpoints/progress.json`.

    Uploads are BATCHED: shards accumulate locally until ~`batch_bytes` of data (or
    `batch_units` finished units), then go up in ONE commit — this keeps us well under
    HF's 128-commits/hour cap (the old commit-per-file approach 429'd). `_commit` also
    backs off on 429.

    resume=False (default): wipe data/ + _checkpoints/ and start clean.
    resume=True: read progress.json and skip units already recorded there.
    `limit` caps rows per (split, source) for a cheap smoke run (progress not recorded).

    Workflow: first run WITHOUT resume (clean start), then re-run WITH resume after any
    interruption until it completes.
    """
    from datasets import load_dataset
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo, hf_hub_download

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

    done = set()  # unit keys already committed to progress.json
    if resume:
        try:
            p = hf_hub_download(repo, "_checkpoints/progress.json",
                                repo_type="dataset", force_download=True)
            with open(p, encoding="utf-8") as f:
                done = set(json.load(f))
            print(f"[curate] resume: {len(done)} units already done")
        except Exception:  # noqa: BLE001
            pass

    ops, tmp_paths, pending_units = [], [], []
    staged_bytes = 0

    def flush(message):
        nonlocal staged_bytes
        if not ops and not pending_units:
            return
        ops.append(CommitOperationAdd(
            path_in_repo="_checkpoints/progress.json",
            path_or_fileobj=(json.dumps(sorted(done)) + "\n").encode("utf-8"),
        ))
        _commit(api, repo, ops, message)
        for pth in tmp_paths:
            try:
                os.remove(pth)
            except OSError:
                pass
        print(f"  [curate] committed {len(ops)} file(s); {len(done)} units done")
        ops.clear(); tmp_paths.clear(); pending_units.clear(); staged_bytes = 0

    def stage(records, tag, idx):
        nonlocal staged_bytes
        path_in_repo, path, size = _write_shard(records, tag, idx, features)
        ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=path))
        tmp_paths.append(path)
        staged_bytes += size
        if staged_bytes >= batch_bytes:
            flush(f"curate: {len(ops)} shard(s)")

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
                key = f"{split}-{src}-{_sanitize(unit)}"
                if key in done:
                    print(f"  [resume] skip done: {split}/{src}/{unit}")
                    continue
                buf, idx, n = [], 0, 0
                for rec in stream_fn(cfg, split, unit):
                    buf.append(rec)
                    n += 1
                    if len(buf) >= shard_size:
                        stage(buf, key, idx)
                        idx, buf = idx + 1, []
                    if limit and (n_src + n) >= limit:
                        break
                if buf:
                    stage(buf, key, idx)
                    idx += 1
                if limit is None:  # record unit done (committed on the next flush)
                    done.add(key)
                    pending_units.append(key)
                if n:
                    print(f"[curate] {split}/{src}/{unit}: {n} clips in {idx} shard(s)")
                n_src += n
                if len(pending_units) >= batch_units:
                    flush(f"curate: {len(pending_units)} unit(s)")
                if limit and n_src >= limit:
                    break
            n_split += n_src
        totals[split] = n_split

    flush("curate: final batch")
    print(f"[curate] done -> https://huggingface.co/datasets/{repo}")
    if verify and any(totals.values()):
        first = next(s for s in SPLITS if totals.get(s))
        ex = next(iter(load_dataset(repo, split=first, streaming=True)))
        ok = isinstance(ex.get("audio"), dict) and ex["audio"].get("array") is not None
        print(f"[curate] verify audio decodes on '{first}': {'OK' if ok else 'FAILED'}")
    return totals
