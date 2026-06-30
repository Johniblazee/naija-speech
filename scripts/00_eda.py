"""Step 0 — EDA on the Nigerian subset of AfriSpeech-200 (metadata only, no audio).

Usage:
    python scripts/00_eda.py                 # writes outputs/eda/eda_report.md + figures
    python scripts/00_eda.py --wandb         # also log tables/figures to W&B
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eda  # noqa: E402
from config import load_dotenv, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA on AfriSpeech-200 Nigerian subset.")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)

    df = eda.load_ng_metadata(
        cfg["hf_dataset_id"],
        cache_dir=os.path.join(args.out, "_cache"),
        country=cfg.get("country_filter", "NG"),
        macro_map=cfg.get("macro_accent_map", {}),
    )
    report = eda.write_report(df, args.out)
    print(f"Nigerian clips: {len(df):,} | hours: {eda._hours(df['duration'])} | "
          f"speakers: {df['user_ids'].nunique():,} | accents: {df['accent'].nunique()}")
    print("Macro-accent split:", df["macro_accent"].value_counts().to_dict())
    print("NG accents:", eda.ng_accents(df))
    print(f"Report: {report}")

    if args.wandb:
        from tracking import maybe_log_wandb_tables

        maybe_log_wandb_tables(eda.summary_tables(df), project=os.environ.get("WANDB_PROJECT", "naija-speech"))


if __name__ == "__main__":
    main()
