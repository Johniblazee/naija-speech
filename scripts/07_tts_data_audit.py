"""Step 7 — TTS data audit (Phase 0 of the TTS track).

Answers the three questions the TTS fine-tunes depend on, WITHOUT re-downloading
the 70 GB corpus:

  A. Speaker backfill — can AfriSpeech-200 rows recover their speaker IDs by
     joining the source transcript manifest on transcript text?  (Orpheus needs
     speaker conditioning; our curated schema has empty speaker_id for that source.)
     -> metadata-only: DuckDB reads JUST the text/accent/duration columns out of
        the remote parquet shards (columnar projection; ~MBs, not GBs).

  B. Sample rates & channels — what did "store native" actually store?
     (StyleTTS 2 trains at 24 kHz; clips recorded below that can't be upsampled
     into quality.)  -> header-only reads on a streamed sample.

  C. Quality heuristics — clipping / silence / dynamic-range proxies per
     source+domain, to shape the TTS-grade selection. Labeled HEURISTIC; the
     GPU-based UTMOSv2 scoring pass happens at Phase 2 selection time.

Usage:
    python scripts/07_tts_data_audit.py                    # full audit (A + B + C)
    python scripts/07_tts_data_audit.py --sample 500       # smaller audio sample
    python scripts/07_tts_data_audit.py --skip-audio       # metadata part only
    python scripts/07_tts_data_audit.py --self-test        # no network; checks logic

Writes outputs/tts_audit/tts_audit_report.md (+ CSVs).
"""
from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402

OUT_DIR = "outputs/tts_audit"


# --------------------------------------------------------------------------- #
# A. Speaker backfill (metadata-only, remote columnar reads)
# --------------------------------------------------------------------------- #
def _norm_text(s: str) -> str:
    return " ".join(str(s).split()).strip().lower()


def match_speakers(corpus_df, manifest_df):
    """Join corpus rows to manifest user_ids by transcript text.

    Tiers: (1) text unique in manifest -> direct match; (2) text ambiguous ->
    disambiguate by (accent, duration rounded to 0.1 s); (3) still ambiguous;
    (4) text not in manifest at all. Returns (annotated corpus_df, tier counts).
    """
    import pandas as pd

    m = manifest_df.copy()
    m["key"] = m["transcript"].map(_norm_text)
    corpus_df = corpus_df.copy()
    corpus_df["key"] = corpus_df["text_raw"].map(_norm_text)

    per_key = m.groupby("key")["user_ids"].nunique()
    unique_keys = set(per_key[per_key == 1].index)
    key_to_user = m.drop_duplicates("key").set_index("key")["user_ids"]

    m["key2"] = m["key"] + "|" + m["accent"].astype(str) + "|" + m["duration"].round(1).astype(str)
    per_key2 = m.groupby("key2")["user_ids"].nunique()
    unique_key2 = set(per_key2[per_key2 == 1].index)
    key2_to_user = m.drop_duplicates("key2").set_index("key2")["user_ids"]

    tiers, users = [], []
    for _, r in corpus_df.iterrows():
        k = r["key"]
        if k in unique_keys:
            tiers.append("unique_text"); users.append(key_to_user[k]); continue
        k2 = k + "|" + str(r["accent"]) + "|" + str(round(float(r["duration"]), 1))
        if k2 in unique_key2:
            tiers.append("text_accent_dur"); users.append(key2_to_user[k2]); continue
        if k in set(m["key"]):
            tiers.append("ambiguous"); users.append(None)
        else:
            tiers.append("unmatched"); users.append(None)
    corpus_df["backfill_tier"] = tiers
    corpus_df["backfilled_speaker"] = users
    counts = corpus_df["backfill_tier"].value_counts().to_dict()
    return corpus_df, counts


def audit_speakers(cfg):
    import duckdb
    import pandas as pd
    from eda import load_ng_metadata

    import re

    repo = cfg["hf_curated_repo"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):  # HF repo id only — keeps the SQL literal clean
        raise SystemExit(f"invalid hf_curated_repo: {repo!r}")
    print(f"[A] reading corpus metadata columns remotely via DuckDB (no audio) …")
    con = duckdb.connect()
    corpus = con.execute(
        f"""SELECT text_raw, accent, domain, duration, gender
            FROM read_parquet('hf://datasets/{repo}/data/*.parquet')
            WHERE source = 'afrispeech-200'"""
    ).df()
    print(f"[A] corpus rows (afrispeech-200): {len(corpus):,}")

    print("[A] loading source transcript manifest …")
    manifest = load_ng_metadata(cfg["hf_dataset_id"], cache_dir=".cache_eda",
                                country=cfg.get("country_filter", "NG"))
    print(f"[A] manifest rows: {len(manifest):,} | speakers: {manifest['user_ids'].nunique():,}")

    annotated, counts = match_speakers(corpus, manifest)
    total = len(annotated)
    matched = counts.get("unique_text", 0) + counts.get("text_accent_dur", 0)
    print(f"\n[A] SPEAKER BACKFILL: {matched:,}/{total:,} rows recoverable "
          f"({matched / total:.1%})")
    for tier, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"     {tier:<16} {n:>8,}  ({n / total:.1%})")
    os.makedirs(OUT_DIR, exist_ok=True)
    annotated.drop(columns=["key"]).to_csv(
        os.path.join(OUT_DIR, "speaker_backfill.csv"), index=False)
    return counts, total, annotated


# --------------------------------------------------------------------------- #
# B + C. Audio sample: headers (rates/channels) + quality heuristics
# --------------------------------------------------------------------------- #
def _raw_bytes(audio_field):
    """Raw encoded bytes from a streamed audio cell, across datasets versions."""
    if isinstance(audio_field, dict) and audio_field.get("bytes"):
        return audio_field["bytes"]
    return None


def quality_heuristics(x, sr):
    """HEURISTIC quality proxies on a decoded waveform (float32 mono).

    clipping_pct: share of samples at >0.99 full scale (distortion proxy).
    silence_pct: share of 50 ms frames under -45 dBFS (dead-air proxy).
    dyn_range_db: 95th-vs-10th percentile frame energy (SNR-ish proxy —
                  honest name, it is NOT a true SNR).
    """
    import numpy as np

    x = np.asarray(x, dtype="float32")
    if x.size == 0:
        return {"clipping_pct": 0.0, "silence_pct": 100.0, "dyn_range_db": 0.0}
    clip = float((np.abs(x) > 0.99).mean() * 100)
    frame = max(1, int(sr * 0.05))
    n = (x.size // frame) * frame
    frames = x[:n].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    silence = float((db < -45).mean() * 100)
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    return {"clipping_pct": round(clip, 2), "silence_pct": round(silence, 1),
            "dyn_range_db": round(dyn, 1)}


def audit_audio(cfg, sample_n, decode_n, seed=7):
    import pandas as pd
    import soundfile as sf
    from datasets import Audio, load_dataset

    repo = cfg["hf_curated_repo"]
    print(f"\n[B] streaming a {sample_n}-clip sample for header stats …")
    stream = (load_dataset(repo, split="train", streaming=True)
              .cast_column("audio", Audio(decode=False))
              .shuffle(seed=seed, buffer_size=500))
    rows = []
    for i, c in enumerate(stream):
        if i >= sample_n:
            break
        b = _raw_bytes(c["audio"])
        if b is None:
            continue
        try:
            info = sf.info(io.BytesIO(b))
        except Exception:
            continue
        row = {"source": c["source"], "domain": c["domain"], "accent": c["accent"],
               "duration": c["duration"], "samplerate": info.samplerate,
               "channels": info.channels, "format": info.format}
        if i < decode_n:  # quality heuristics on the first decode_n clips
            try:
                x, sr = sf.read(io.BytesIO(b), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                row.update(quality_heuristics(x, sr))
            except Exception:
                pass
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  … {i + 1}/{sample_n}")
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "audio_sample_stats.csv"), index=False)

    print(f"\n[B] SAMPLE RATES (n={len(df)}):")
    print(df.groupby("samplerate").size().to_string())
    print(f"\n[B] channels: {df.groupby('channels').size().to_dict()}")
    q = df.dropna(subset=["dyn_range_db"]) if "dyn_range_db" in df else pd.DataFrame()
    if len(q):
        print(f"\n[C] QUALITY HEURISTICS (n={len(q)}, by source+domain; "
              f"dyn_range_db higher=better, others lower=better):")
        print(q.groupby(["source", "domain"])[
            ["clipping_pct", "silence_pct", "dyn_range_db"]].mean().round(1).to_string())
    return df


# --------------------------------------------------------------------------- #
def write_report(cfg, speaker_counts, speaker_total, audio_df):
    lines = [f"# TTS Data Audit — {cfg['hf_curated_repo']}", "",
             "Phase 0 of the TTS track. Heuristic quality proxies; UTMOSv2 scoring",
             "happens at selection time on GPU.", ""]
    if speaker_counts:
        matched = speaker_counts.get("unique_text", 0) + speaker_counts.get("text_accent_dur", 0)
        lines += ["## A. Speaker backfill (afrispeech-200 rows)", "",
                  f"- Recoverable: **{matched:,}/{speaker_total:,} ({matched/speaker_total:.1%})**"]
        lines += [f"- {t}: {n:,} ({n/speaker_total:.1%})" for t, n in sorted(
            speaker_counts.items(), key=lambda kv: -kv[1])]
        lines += ["", "Full annotation: `speaker_backfill.csv`", ""]
    if audio_df is not None and len(audio_df):
        lines += ["## B. Sample rates (streamed sample)", "",
                  audio_df.groupby("samplerate").size().to_markdown(), "",
                  "## C. Quality heuristics (by source + domain)", ""]
        if "dyn_range_db" in audio_df:
            q = audio_df.dropna(subset=["dyn_range_db"])
            lines += [q.groupby(["source", "domain"])[
                ["clipping_pct", "silence_pct", "dyn_range_db"]]
                .mean().round(1).to_markdown(), ""]
    path = os.path.join(OUT_DIR, "tts_audit_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[report] -> {path}")


def self_test():
    """No-network check of the join and heuristics logic."""
    import numpy as np
    import pandas as pd

    manifest = pd.DataFrame({
        "transcript": ["Hello world", "Hello world", "Unique line", "Shared text", "Shared text"],
        "user_ids": ["u1", "u2", "u3", "u4", "u4"],
        "accent": ["yoruba", "igbo", "hausa", "igbo", "igbo"],
        "duration": [2.0, 3.0, 4.0, 5.0, 5.0],
    })
    corpus = pd.DataFrame({
        "text_raw": ["Unique line", "Hello  world", "Shared text", "Never seen"],
        "accent": ["hausa", "yoruba", "igbo", "tiv"],
        "duration": [4.0, 2.01, 5.0, 1.0],
        "domain": ["general"] * 4, "gender": ["Male"] * 4,
    })
    out, counts = match_speakers(corpus, manifest)
    assert out.backfill_tier.tolist() == ["unique_text", "text_accent_dur",
                                          "unique_text", "unmatched"], out.backfill_tier.tolist()
    assert out.backfilled_speaker.tolist()[:3] == ["u3", "u1", "u4"]

    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.3 * np.sin(2 * np.pi * 220 * t).astype("float32")
    q = quality_heuristics(clean, sr)
    assert q["clipping_pct"] == 0.0 and q["silence_pct"] < 5, q
    clipped = np.clip(clean * 10, -1, 1)
    assert quality_heuristics(clipped, sr)["clipping_pct"] > 50
    silent = np.zeros(sr, dtype="float32")
    assert quality_heuristics(silent, sr)["silence_pct"] == 100.0
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="TTS data audit (Phase 0).")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--sample", type=int, default=1000,
                    help="Clips to stream for header stats.")
    ap.add_argument("--decode", type=int, default=300,
                    help="Of those, clips to fully decode for quality heuristics.")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-speakers", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    load_dotenv()
    cfg = load_yaml(args.config)

    speaker_counts, speaker_total = None, 0
    if not args.skip_speakers:
        speaker_counts, speaker_total, _ = audit_speakers(cfg)
    audio_df = None
    if not args.skip_audio:
        audio_df = audit_audio(cfg, args.sample, args.decode)
    write_report(cfg, speaker_counts, speaker_total, audio_df)


if __name__ == "__main__":
    main()
