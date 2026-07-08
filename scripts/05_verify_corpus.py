"""Step 5 — verify the curated corpus (metadata only, no audio download).

Reads just the metadata columns from the curated dataset's Parquet shards (skipping
the ~70 GB of audio bytes via columnar projection) and reports clip/hour counts per
source, macro-accent, accent and domain, plus a speaker-disjoint (train vs test /
validation leakage) check. Produces the tables that feed thesis Chapters 4 and 5.

Run on Colab/RunPod where `pyarrow` + fast network are available:
    python scripts/05_verify_corpus.py
Writes the report to outputs/verify/corpus_verification.md.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402

_META_COLS = ["source", "macro_accent", "accent", "duration", "speaker_id",
              "gender", "domain", "language"]


def _split_of(path: str) -> str:
    """Shard path is data/<split>-<source>-<accent>-<idx>.parquet."""
    return path.split("/")[-1].split("-")[0]


def read_metadata(repo: str):
    """Return one DataFrame of metadata columns across every shard (no audio read)."""
    import pandas as pd
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet")]
    print(f"[verify] reading metadata from {len(files)} shards …")
    frames = []
    for i, f in enumerate(files):
        try:
            d = pd.read_parquet(f"hf://datasets/{repo}/{f}", columns=_META_COLS)
        except Exception as e:  # noqa: BLE001 — tolerate a bad/edge shard
            print(f"  [warn] {f}: {e}")
            continue
        d["split"] = _split_of(f)
        frames.append(d)
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{len(files)} shards")
    return pd.concat(frames, ignore_index=True)


def _hours(s):
    """Sum only positive durations (some source rows carry -1.0 = unknown)."""
    v = s[s > 0]
    return round(float(v.sum()) / 3600, 1)


def summarize(df) -> dict:
    tables = {"by_split": df.groupby("split").agg(
        clips=("source", "size"), hours=("duration", _hours),
        speakers=("speaker_id", "nunique")).sort_values("clips", ascending=False)}
    for col in ("source", "macro_accent", "accent", "domain", "gender"):
        tables[f"by_{col}"] = df.groupby(col).agg(
            clips=("source", "size"), hours=("duration", _hours),
            speakers=("speaker_id", "nunique")).sort_values("clips", ascending=False)
    return tables


def speaker_disjoint(df) -> dict:
    """Shared speaker_ids between train and test/validation (want 0 = no leakage)."""
    sets = {sp: set(g["speaker_id"].dropna()) - {""} for sp, g in df.groupby("split")}
    train = sets.get("train", set())
    return {f"train n {other}": len(train & sets.get(other, set()))
            for other in ("test", "validation")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the curated corpus (metadata only).")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    repo = cfg["hf_curated_repo"]

    df = read_metadata(repo)
    dur_cov = float((df["duration"] > 0).mean()) * 100
    tables = summarize(df)
    overlap = speaker_disjoint(df)

    print(f"\n[verify] {len(df):,} clips | duration present for {dur_cov:.0f}% of rows")
    for name, t in tables.items():
        print(f"\n=== {name} ===\n{t.to_string()}")
    print("\n=== speaker-disjoint (want 0) ===")
    for k, v in overlap.items():
        print(f"  {k}: {v} shared speakers  [{'OK' if v == 0 else 'LEAKAGE'}]")

    out_dir = "outputs/verify"  # fixed location — no user-controlled path reaches open()
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"# Corpus verification — {repo}", "",
             f"- Total clips: **{len(df):,}**",
             f"- Duration present for **{dur_cov:.0f}%** of rows", ""]
    for name, t in tables.items():
        try:
            md = t.to_markdown()
        except Exception:  # noqa: BLE001 — tabulate missing
            md = "```\n" + t.to_string() + "\n```"
        lines += [f"## {name}", "", md, ""]
    lines += ["## Speaker-disjoint check (want 0)", ""]
    lines += [f"- {k}: {v} shared speakers ({'OK' if v == 0 else 'LEAKAGE'})"
              for k, v in overlap.items()]
    report = os.path.join(out_dir, "corpus_verification.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[verify] report -> {report}")


if __name__ == "__main__":
    main()
