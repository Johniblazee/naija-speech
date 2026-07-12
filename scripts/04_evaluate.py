"""Step 4 — evaluate the fine-tuned (LoRA) model and produce Chapter-5 tables.

Loads the base Whisper model + the trained LoRA adapter, runs the same decoding
path as the baseline, and reports WER/CER overall + stratified by macro-accent
and domain so the improvement over zero-shot is directly comparable.

Usage:
    python scripts/04_evaluate.py --limit 50
    python scripts/04_evaluate.py
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
from whisper_lora import transcribe_dataset  # noqa: E402


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
    ap = argparse.ArgumentParser(description="Evaluate the fine-tuned Whisper+LoRA model.")
    ap.add_argument("--data-config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--model-config", default="configs/stt_whisper_small_lora.yaml")
    ap.add_argument("--adapter-dir", default=None,
                    help="Defaults to <model output_dir>/adapter.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_dotenv()
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.model_config)
    adapter_dir = args.adapter_dir or os.path.join(cfg["output_dir"], "adapter")
    out_path = args.out or os.path.join(cfg["results_dir"], "finetuned_results.csv")

    from curate import load_curated
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ds = load_curated(data_cfg["hf_curated_repo"], split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, ds.num_rows)))
    print(f"Evaluating {ds.num_rows} clips from split '{args.split}' with adapter {adapter_dir}")

    processor = WhisperProcessor.from_pretrained(adapter_dir)
    base = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
    model = PeftModel.from_pretrained(base, adapter_dir)

    hyps = transcribe_dataset(
        model, processor, ds,
        batch_size=args.batch_size,
        language=cfg["language"], task=cfg["task"],
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
    write_results_csv(results, out_path)
    hyp_path = os.path.join(os.path.dirname(out_path) or ".", "hypotheses.csv")
    write_hypotheses_csv(rows, hyp_path)
    print(f"\nWrote results to {out_path}\nWrote per-clip pairs to {hyp_path}")

    maybe_log_wandb(
        results,
        params={
            "stage": "finetuned-lora",
            "model_id": cfg["model_id"],
            "adapter_dir": adapter_dir,
            "split": args.split,
            "n_clips": ds.num_rows,
        },
        name=f"finetuned-{cfg['model_id'].split('/')[-1]}-lora",
    )


if __name__ == "__main__":
    main()
