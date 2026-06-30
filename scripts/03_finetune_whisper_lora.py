"""Step 3 — LoRA fine-tune Whisper on the Nigerian corpus.

Trains LoRA adapters (parameter-efficient: ~1-5% of weights) and logs to Comet.
Saves the adapter + processor to the config's output_dir.

Usage:
    python scripts/03_finetune_whisper_lora.py --max-steps 50    # smoke run
    python scripts/03_finetune_whisper_lora.py                   # full run (uses epochs)
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
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override max_steps for a quick smoke run.")
    args = ap.parse_args()

    load_dotenv()
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.model_config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    # W&B auto-logging via the Trainer reads WANDB_* env vars.
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))

    from datasets import load_from_disk
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    processor = build_processor(cfg)
    dsd = load_from_disk(data_cfg["output_dir"])

    prepare = make_prepare_fn(processor)
    train_ds = dsd["train"].map(prepare, remove_columns=dsd["train"].column_names,
                                desc="prepare train")
    eval_split = "validation" if "validation" in dsd else "test"
    eval_ds = dsd[eval_split].map(prepare, remove_columns=dsd[eval_split].column_names,
                                  desc="prepare eval")

    model = load_model(cfg)
    model = apply_lora(model, cfg)
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
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
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
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False  # silence warning; required during training

    trainer.train()

    adapter_dir = os.path.join(cfg["output_dir"], "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\nSaved LoRA adapter + processor to: {adapter_dir}")


if __name__ == "__main__":
    main()
