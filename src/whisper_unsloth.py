"""Unsloth build path for Whisper LoRA fine-tuning.

Unsloth trains Whisper ~2x faster and, in 4-bit, fits `whisper-large-v3(-turbo)`
on a *free* Colab T4 — so we can fine-tune a large model without renting an A100.

This module only replaces the model-build step. It returns a `(model, processor)`
pair that is already LoRA-wrapped and drop-in compatible with the rest of the
pipeline (`whisper_lora.DataCollatorSpeechSeq2SeqWithPadding`, the `prepare` fn,
`make_compute_metrics`, and `Seq2SeqTrainer`) — Unsloth's returned processor
exposes the same `.feature_extractor` / `.tokenizer` / `.batch_decode` API as a
`WhisperProcessor`. Recipe verified against unslothai/notebooks `nb/Whisper.ipynb`.

`unsloth` is a heavy GPU-only dependency, so it is imported lazily inside the
function — importing this module (and unit-testing the rest of the package) does
not require it to be installed.
"""
from __future__ import annotations

from typing import Any

# Unsloth's `whisper_language` wants the capitalized language *name*.
_WHISPER_LANG = {"english": "English"}


def build_unsloth(cfg: dict[str, Any]):
    """FastModel.from_pretrained + get_peft_model -> (model, processor), LoRA-ready."""
    from unsloth import FastModel  # MUST precede transformers so Unsloth's patches apply
    from transformers import WhisperForConditionalGeneration

    lang = _WHISPER_LANG.get(str(cfg["language"]).lower(), cfg["language"])

    model, processor = FastModel.from_pretrained(
        model_name=cfg["model_id"],
        auto_model=WhisperForConditionalGeneration,
        whisper_language=lang,
        whisper_task=cfg["task"],
        load_in_4bit=cfg.get("load_in_4bit", False),
        dtype=None,  # auto-detect (bf16 on Ampere+, else fp16)
    )

    lc = cfg["lora"]
    model = FastModel.get_peft_model(
        model,
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # ~30% less VRAM, fits 2x batch
        random_state=cfg.get("seed", 3407),
        task_type=None,  # verified: Whisper LoRA passes task_type=None
    )
    return model, processor
