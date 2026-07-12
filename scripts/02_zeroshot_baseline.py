"""Step 2 — zero-shot baseline: measure the accent gap before any fine-tuning.

Runs the pretrained Whisper model as-is on the Nigerian test set and reports
WER/CER overall and stratified by macro-accent and domain. This is the number
fine-tuning must beat (thesis decision threshold).

Usage:
    python scripts/02_zeroshot_baseline.py --limit 50        # quick check
    python scripts/02_zeroshot_baseline.py                   # full test split
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402
from metrics import compute_stratified, write_hypotheses_csv  # noqa: E402
from tracking import maybe_log_wandb  # noqa: E402
from whisper_lora import build_processor, transcribe_dataset  # noqa: E402


def write_results_csv(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "key", "wer", "cer", "n"])
        writer.writeheader()
        writer.writerows(results)


def print_table(results: list[dict]) -> None:
    print(f"\n{'group':<14}{'key':<14}{'WER':>8}{'CER':>8}{'N':>8}")
    print("-" * 52)
    for r in results:
        wer = f"{r['wer']:.3f}" if r["n"] else "--"
        cer = f"{r['cer']:.3f}" if r["n"] else "--"
        print(f"{r['group']:<14}{str(r['key']):<14}{wer:>8}{cer:>8}{r['n']:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot Whisper baseline on Nigerian test set.")
    ap.add_argument("--data-config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--model-config", default="configs/stt_whisper_small_lora.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="outputs/zeroshot/baseline_results.csv")
    args = ap.parse_args()

    load_dotenv()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    from curate import load_curated

    ds = load_curated(data_cfg["hf_curated_repo"], split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, ds.num_rows)))
    print(f"Evaluating {ds.num_rows} clips from split '{args.split}' "
          f"with zero-shot {model_cfg['model_id']}")

    from transformers import WhisperForConditionalGeneration

    processor = build_processor(model_cfg)
    model = WhisperForConditionalGeneration.from_pretrained(model_cfg["model_id"])

    hyps = transcribe_dataset(
        model, processor, ds,
        batch_size=args.batch_size,
        language=model_cfg["language"], task=model_cfg["task"],
    )

    rows = [
        {
            "reference": ds[i]["text_raw"],
            "hypothesis": hyps[i],
            "macro_accent": ds[i]["macro_accent"],
            "domain": ds[i]["domain"],
        }
        for i in range(ds.num_rows)
    ]
    results = compute_stratified(rows)

    print_table(results)
    write_results_csv(results, args.out)
    hyp_path = os.path.join(os.path.dirname(args.out) or ".", "hypotheses.csv")
    write_hypotheses_csv(rows, hyp_path)
    print(f"\nWrote results to {args.out}\nWrote per-clip pairs to {hyp_path}")

    maybe_log_wandb(
        results,
        params={
            "stage": "zero-shot-baseline",
            "model_id": model_cfg["model_id"],
            "split": args.split,
            "n_clips": ds.num_rows,
        },
        name=f"zeroshot-{model_cfg['model_id'].split('/')[-1]}",
    )


if __name__ == "__main__":
    main()
