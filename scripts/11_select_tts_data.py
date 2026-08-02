"""Step 11 — Select + export the StyleTTS 2 fine-tuning dataset (TTS Phase 2).

Builds the TTS-grade training set the fine-tune consumes, in the OFFICIAL
StyleTTS2 format (verified against yl4579/StyleTTS2 @ main, 2026-07-30):
  - WAVs: 24,000 Hz MONO (resampled offline — the repo resamples on-the-fly
    otherwise, slowly), silence-trimmed with ~100 ms trailing pad (Discussion
    #81: prevents end-of-utterance artifacts).
  - train_list/val_list lines: "file.wav|IPA_PHONEMES|speaker_int" — transcripts
    must be PRE-PHONEMIZED (espeak-ng via phonemizer, en-us, punctuation +
    stress preserved); the training code only maps IPA symbols to indices.
  - Speaker ints are labels for same-speaker reference sampling, so every kept
    speaker has >=2 clips. Unseen speakers are fine (LibriTTS ckpt is zero-shot).
  - Keep the repo's default Data/OOD_texts.txt (mandatory load, already IPA).

Pipeline (each stage checkpoints to outputs/tts_ft_data/, resume-safe):
  meta   candidates from the train split: 1.5-10 s (max_len<=800 frames = 10 s),
         read speech (afrispeech-200 only, ADR-24), speakers with >=2 clips.
  audio  stream candidates until --pool-hours: decode, heuristic gate
         (clipping/silence), trim + pad, resample 24k mono -> staging WAVs.
  score  UTMOSv2 predicted MOS over the staging dir (GPU; pip install
         git+https://github.com/sarulab-speech/UTMOSv2.git).
  lists  rank by MOS, accent-cap, enforce >=2 clips/speaker, take --hours,
         phonemize, write train_list.txt / val_list.txt + selection.csv.

Runs on the pod (venv-main + utmosv2 + phonemizer; apt-get install espeak-ng).

Usage:
    python scripts/11_select_tts_data.py                   # all stages
    python scripts/11_select_tts_data.py --stage meta
    python scripts/11_select_tts_data.py --pool-hours 25 --hours 10
    python scripts/11_select_tts_data.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402

OUT_DIR = "outputs/tts_ft_data"
WAV_DIR = os.path.join(OUT_DIR, "wavs")
TARGET_SR = 24000
MIN_S, MAX_S = 1.5, 10.0  # max_len 800 frames = 10 s is the quality sweet spot


def _mod(name):
    p = os.path.join(os.path.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- #
def stage_meta(cfg):
    """Candidate metadata: duration window + speakers (path join) + >=2 clips."""
    import pandas as pd

    from corpus import macro_accent
    from eda import load_ng_metadata

    ev = _mod("08_build_tts_evalset.py")
    audit = _mod("07_tts_data_audit.py")
    repo = ev._check_repo(cfg["hf_curated_repo"])
    train = ev._read_split(repo, "train")
    manifest = load_ng_metadata(cfg["hf_dataset_id"], cache_dir=".cache_eda",
                                country=cfg.get("country_filter", "NG"))
    df, _ = audit.match_speakers(train, manifest)
    mm = {k.lower(): v for k, v in cfg.get("macro_accent_map", {}).items()}
    df["macro_accent"] = df["accent"].map(lambda a: macro_accent(a, mm))
    df = df[(df["duration"] >= MIN_S) & (df["duration"] <= MAX_S)
            & df["backfilled_speaker"].notna()].copy()
    counts = df.groupby("backfilled_speaker")["apath"].transform("size")
    df = df[counts >= 2]
    df = df.sort_values("apath").reset_index(drop=True)  # deterministic order
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "candidates.csv")
    df.drop(columns=["key"], errors="ignore").to_csv(path, index=False)
    hours = df["duration"].sum() / 3600
    print(f"[meta] {len(df):,} candidates / {hours:.1f} h / "
          f"{df['backfilled_speaker'].nunique():,} speakers -> {path}")
    return df


def trim_pad(x, sr, top_db=35, pad_ms=100):
    """Trim leading/trailing silence, append ~100 ms of silence (issue #81)."""
    import librosa
    import numpy as np

    if not np.abs(x).any():  # librosa trims relative to peak; pure silence has none
        return x[:0]
    y, _ = librosa.effects.trim(x, top_db=top_db)
    if y.size == 0:
        return y
    return np.concatenate([y, np.zeros(int(sr * pad_ms / 1000), dtype=y.dtype)])


def stage_audio(cfg, pool_hours, shards_dir=None):
    """Stream candidates, gate on heuristics, export 24 kHz mono staging WAVs.

    shards_dir: local dir of downloaded train-*.parquet shards — reads locally
    (fast, no CDN timeouts) instead of streaming from the Hub.
    """
    import glob
    import librosa
    import pandas as pd
    import soundfile as sf
    from datasets import Audio, load_dataset

    audit = _mod("07_tts_data_audit.py")
    cand = pd.read_csv(os.path.join(OUT_DIR, "candidates.csv"))
    want = dict(zip(cand["apath"], cand.index))
    os.makedirs(WAV_DIR, exist_ok=True)
    done_s = 0.0
    rows = []
    if shards_dir:
        files = sorted(glob.glob(os.path.join(shards_dir, "train-*.parquet")))
        if not files:
            raise SystemExit(f"no train-*.parquet under {shards_dir}")
        print(f"[audio] reading {len(files)} local shards from {shards_dir}")
        stream = load_dataset("parquet", data_files=files, split="train",
                              streaming=True)  # audio arrives as raw struct dicts
    else:
        stream = (load_dataset(cfg["hf_curated_repo"], split="train", streaming=True)
                  .cast_column("audio", Audio(decode=False)))
    print(f"[audio] streaming until {pool_hours} h of staged audio ...")
    for c in stream:
        if done_s >= pool_hours * 3600:
            break
        apath = (c["audio"] or {}).get("path")
        if apath not in want:
            continue
        i = want[apath]
        out = os.path.join(WAV_DIR, f"{i:06d}.wav")
        if os.path.exists(out):  # resume
            done_s += sf.info(out).duration
            rows.append({"idx": i, "apath": apath, "wav": out})
            continue
        b = (c["audio"] or {}).get("bytes")
        if not b:
            continue
        try:
            x, sr = sf.read(io.BytesIO(b), dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        q = audit.quality_heuristics(x, sr)
        if q["clipping_pct"] > 1.0 or q["silence_pct"] > 60.0:
            continue  # not TTS-grade
        y = trim_pad(x, sr)
        if y.size == 0 or not (MIN_S <= y.size / sr <= MAX_S + 0.2):
            continue
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sf.write(out, y, TARGET_SR, subtype="PCM_16")
        done_s += y.size / TARGET_SR
        rows.append({"idx": i, "apath": apath, "wav": out})
        if len(rows) % 500 == 0:
            print(f"  ... {len(rows):,} staged, {done_s / 3600:.1f} h")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "staged.csv"), index=False)
    print(f"[audio] staged {len(rows):,} clips / {done_s / 3600:.1f} h -> {WAV_DIR}")


def stage_score():
    """UTMOSv2 predicted MOS for every staged WAV."""
    import pandas as pd
    import utmosv2

    model = utmosv2.create_model(pretrained=True)
    results = model.predict(input_dir=WAV_DIR)
    mos = {os.path.basename(r["file_path"]): float(r["predicted_mos"]) for r in results}
    staged = pd.read_csv(os.path.join(OUT_DIR, "staged.csv"))
    staged["mos"] = staged["wav"].map(lambda w: mos.get(os.path.basename(w)))
    staged.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)
    print(f"[score] mean MOS {staged['mos'].mean():.2f} over {len(staged):,} clips")


def select_clips(df, hours, accent_cap=0.5):
    """Greedy by MOS with a per-accent share cap; then enforce >=2 clips/speaker."""
    budget = hours * 3600
    acc_used: dict[str, float] = {}
    chosen, total = [], 0.0
    for r in df.sort_values("mos", ascending=False).itertuples():
        if total >= budget:
            break
        if acc_used.get(r.macro_accent, 0.0) + r.duration > budget * accent_cap:
            continue
        chosen.append(r.Index)
        total += r.duration
        acc_used[r.macro_accent] = acc_used.get(r.macro_accent, 0.0) + r.duration
    sub = df.loc[chosen]
    counts = sub.groupby("speaker").size()
    return sub[sub["speaker"].isin(counts[counts >= 2].index)]


def _phonemize(texts):
    from phonemizer import phonemize

    return phonemize(texts, language="en-us", backend="espeak",
                     preserve_punctuation=True, with_stress=True)


def stage_lists(hours, val_frac, seed=42):
    import pandas as pd

    cand = pd.read_csv(os.path.join(OUT_DIR, "candidates.csv"))
    scores = pd.read_csv(os.path.join(OUT_DIR, "scores.csv")).dropna(subset=["mos"])
    df = scores.merge(cand.reset_index().rename(columns={"index": "idx"}), on="idx",
                      suffixes=("", "_c"))
    df = df.rename(columns={"backfilled_speaker": "speaker"})
    sel = select_clips(df, hours)
    print(f"[lists] selected {len(sel):,} clips / {sel['duration'].sum() / 3600:.1f} h / "
          f"{sel['speaker'].nunique():,} speakers | accents: "
          f"{sel.groupby('macro_accent')['duration'].sum().div(3600).round(1).to_dict()}")

    spk_ids = {s: i for i, s in enumerate(sorted(sel["speaker"].unique()))}
    # pipe is the list delimiter — it must never appear in the text
    texts = sel["text_raw"].astype(str).str.replace("|", "/", regex=False)
    print(f"[lists] phonemizing {len(texts):,} transcripts (espeak-ng) ...")
    sel = sel.assign(ipa=_phonemize(texts.tolist()),
                     spk=sel["speaker"].map(spk_ids))

    val = sel.sample(frac=val_frac, random_state=seed)
    train = sel.drop(val.index)
    for name, part in (("train_list.txt", train), ("val_list.txt", val)):
        lines = [f"{os.path.basename(r.wav)}|{r.ipa}|{r.spk}" for r in part.itertuples()]
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"[lists] {name}: {len(lines):,} lines")
    sel.drop(columns=["ipa"]).to_csv(os.path.join(OUT_DIR, "selection.csv"), index=False)
    print(f"[lists] root_path for config_ft.yml: {WAV_DIR}")


def self_test():
    import numpy as np
    import pandas as pd

    sr = 24000
    x = np.concatenate([np.zeros(sr), 0.3 * np.sin(np.linspace(0, 900 * np.pi, 3 * sr)),
                        np.zeros(sr)]).astype("float32")
    y = trim_pad(x, sr)
    assert 2.9 * sr < y.size < 3.4 * sr, y.size  # silence gone, ~100ms pad kept
    assert trim_pad(np.zeros(sr, dtype="float32"), sr).size == 0

    df = pd.DataFrame({
        "mos": [4.5, 4.4, 4.3, 4.2, 2.0, 4.1],
        "duration": [3600.0] * 6,
        "macro_accent": ["Yoruba", "Yoruba", "Yoruba", "Igbo", "Igbo", "Igbo"],
        "speaker": ["a", "a", "b", "c", "c", "c"],
    })
    sel = select_clips(df, hours=4, accent_cap=0.5)
    # accent cap: max 2h/accent -> Yoruba a,a (b's clip exceeds cap), Igbo c,c
    assert sorted(sel["speaker"]) == ["a", "a", "c", "c"], sel
    assert (sel.groupby("speaker").size() >= 2).all()
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Select + export StyleTTS2 FT data (Phase 2).")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--stage", default="all", choices=("all", "meta", "audio", "score", "lists"))
    ap.add_argument("--pool-hours", type=float, default=25.0,
                    help="Hours to stage for scoring (superset of the final cut).")
    ap.add_argument("--shards-dir", default=None,
                    help="Local dir of downloaded train-*.parquet shards "
                         "(hf download ... --include 'data/train-*.parquet'); "
                         "reads locally instead of streaming from the Hub.")
    ap.add_argument("--hours", type=float, default=10.0,
                    help="Final training-set size (top-MOS, accent-capped).")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    load_dotenv()
    cfg = load_yaml(args.config)
    if args.stage in ("all", "meta"):
        stage_meta(cfg)
    if args.stage in ("all", "audio"):
        stage_audio(cfg, args.pool_hours, args.shards_dir)
    if args.stage in ("all", "score"):
        stage_score()
    if args.stage in ("all", "lists"):
        stage_lists(args.hours, args.val_frac)


if __name__ == "__main__":
    main()
