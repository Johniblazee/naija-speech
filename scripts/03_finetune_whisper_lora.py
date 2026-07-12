"""Step 3 — LoRA fine-tune Whisper on the Nigerian corpus.

Trains LoRA adapters (parameter-efficient: ~1-5% of weights) and logs to W&B.
Saves the adapter + processor to the config's output_dir.

Two model-build backends (the rest of the pipeline is identical either way):
  --backend hf       vanilla HuggingFace + PEFT (default; CPU-importable, robust)
  --backend unsloth  Unsloth FastModel — ~2x faster, and in 4-bit fits
                     whisper-large-v3(-turbo) on a *free* Colab T4.

Usage:
    python scripts/03_finetune_whisper_lora.py --max-steps 50            # smoke run
    python scripts/03_finetune_whisper_lora.py \
        --model-config configs/stt_whisper_large_v3_turbo_unsloth.yaml \
        --backend unsloth                                                # headline run
    python scripts/03_finetune_whisper_lora.py ... --backend unsloth --resume  # after a Colab disconnect
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402
from whisper_lora import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    apply_lora,
    build_processor,
    load_model,
    make_compute_metrics,
    make_prepare_fn,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune Whisper for Nigerian English.")
    ap.add_argument("--data-config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--model-config", default="configs/stt_whisper_small_lora.yaml")
    ap.add_argument("--backend", choices=["hf", "unsloth"], default="hf",
                    help="Model-build backend. 'unsloth' = 2x faster + 4-bit large models on a T4.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override max_steps for a quick smoke run.")
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap the number of training clips (fast subset run).")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="Override checkpoint frequency (e.g. 50 for a checkpointed smoke run).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint in output_dir (survive Colab disconnects).")
    args = ap.parse_args()

    load_dotenv()
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.model_config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.max_train is not None:
        cfg["max_train"] = args.max_train
    if args.save_steps is not None:
        cfg["save_steps"] = args.save_steps

    # W&B auto-logging via the Trainer reads WANDB_* env vars.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))

    from curate import load_curated

    # IMPORTANT: import unsloth BEFORE transformers so its kernels/patches apply.
    if args.backend == "unsloth":
        from whisper_unsloth import build_unsloth
        model, processor = build_unsloth(cfg)     # already LoRA-wrapped
    else:
        processor = build_processor(cfg)
        model = apply_lora(load_model(cfg), cfg)

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    prepare = make_prepare_fn(processor)

    # Load ONLY the split(s) we need. load_curated(repo) with no split pulls the
    # entire ~70 GB corpus (all splits) before training starts — that is what
    # silently stalled/filled the disk on free Colab. Train split alone is the
    # bulk of it, but the download is cached and reused by later runs.
    print("[data] loading 'train' split (first run downloads it; cached afterwards)")
    train_ds = load_curated(data_cfg["hf_curated_repo"], split="train")
    if cfg.get("max_train"):
        train_ds = train_ds.select(range(min(cfg["max_train"], train_ds.num_rows)))
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                            desc="prepare train")

    # In-training generation eval is heavy for a large model on a T4; the turbo
    # config disables it (eval_strategy: "no") and 04_evaluate.py produces the
    # authoritative stratified WER on the full test split instead.
    do_eval = cfg["eval_strategy"] != "no"
    eval_ds = None
    if do_eval:
        eval_ds = load_curated(data_cfg["hf_curated_repo"], split="validation")
        eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names,
                              desc="prepare eval")

    # Free generation of language/task during eval.
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg["max_steps"],
        fp16=cfg["fp16"],
        optim=cfg.get("optim", "adamw_torch"),
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg.get("save_total_limit", 2),  # cap Colab disk
        logging_steps=cfg["logging_steps"],
        predict_with_generate=True,
        generation_max_length=cfg["generation_max_length"],
        report_to=["wandb"] if use_wandb else ["none"],
        remove_unused_columns=False,   # required: our collator reads custom columns
        label_names=["labels"],        # required for PEFT + Seq2SeqTrainer
        seed=cfg["seed"],
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor) if do_eval else None,
        processing_class=processor.feature_extractor,  # transformers v5 renamed `tokenizer`
    )
    model.config.use_cache = False  # silence warning; required during training

    trainer.train(resume_from_checkpoint=args.resume or None)

    adapter_dir = os.path.join(cfg["output_dir"], "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\nSaved LoRA adapter + processor to: {adapter_dir}")


if __name__ == "__main__":
    main()
