"""Step 1 — build the Nigerian-accented English corpus from AfriSpeech-200.

Usage:
    python scripts/01_build_corpus.py --peek                 # quick streaming sanity check
    python scripts/01_build_corpus.py --max-per-split 50     # small build to validate the pipeline
    python scripts/01_build_corpus.py                        # full build (downloads accent shards)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import corpus  # noqa: E402
from config import load_dotenv, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Nigerian AfriSpeech-200 corpus.")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--max-per-split", type=int, default=None,
                    help="Cap examples per accent/split (fast iteration).")
    ap.add_argument("--peek", action="store_true",
                    help="Stream a few rows per accent without a full download.")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)

    if args.peek:
        rows = corpus.peek_streaming(cfg)
        print(f"\nPeeked {len(rows)} Nigerian rows across accents {cfg['accents']}")
        print("macro_accent:", dict(Counter(r["macro_accent"] for r in rows)))
        print("domain:", dict(Counter(r["domain"] for r in rows)))
        print("\nExamples:")
        for r in rows[:5]:
            print(f"  [{r['macro_accent']}/{r['domain']}] {(r['transcript'] or '')[:70]}")
        return

    dsd = corpus.build_corpus(cfg, max_per_split=args.max_per_split)
    print("\n=== Corpus summary ===")
    print(json.dumps(corpus.summarize(dsd), indent=2, ensure_ascii=False))

    print("\n=== Speaker-disjoint check (should all be 0) ===")
    for pair, overlap in corpus.check_speaker_disjoint(dsd).items():
        flag = "OK" if not overlap else "WARNING: leakage!"
        print(f"  {pair}: {len(overlap)} shared speakers  [{flag}]")

    print(f"\nSaved corpus + manifest.csv to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
