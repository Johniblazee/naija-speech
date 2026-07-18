"""Whisper + LoRA helpers (model, processor, data collation, metrics).

Follows the standard Hugging Face PEFT/LoRA recipe for Whisper. Imports of
torch/transformers/peft are done lazily inside functions so this module can be
imported (and the rest of the package unit-tested) without the heavy ML stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from metrics import compute_wer_cer


def build_processor(cfg: dict[str, Any]):
    """WhisperProcessor pinned to English transcription."""
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(
        cfg["model_id"], language=cfg["language"], task=cfg["task"]
    )


def load_model(cfg: dict[str, Any]):
    """Load the base Whisper model, optionally in 8-bit for large-v3."""
    from transformers import WhisperForConditionalGeneration

    kwargs: dict[str, Any] = {}
    if cfg.get("load_in_8bit"):
        kwargs.update(load_in_8bit=True, device_map="auto")

    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"], **kwargs)
    # Let the model predict the language/task freely during fine-tuning.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if cfg.get("load_in_8bit"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, cfg: dict[str, Any]):
    """Wrap the model with LoRA adapters per the config."""
    from peft import LoraConfig, get_peft_model

    lc = cfg["lora"]
    lora_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        target_modules=lc["target_modules"],
        lora_dropout=lc["dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def make_prepare_fn(processor):
    """Return a `.map` function turning (audio, text) -> model inputs."""

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads input features and label sequences independently (HF recipe)."""

    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # Replace padding with -100 so it's ignored by the loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # If a BOS token was prepended by the tokenizer, drop it (added later).
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    """Return a compute_metrics fn (normalized WER + CER) for the Trainer."""

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        res = compute_wer_cer(label_str, pred_str, normalize=True)
        return {"wer": res["wer"], "cer": res["cer"]}

    return compute_metrics


def transcribe_dataset(
    model,
    processor,
    ds,
    device: str | None = None,
    batch_size: int = 8,
    language: str = "english",
    task: str = "transcribe",
    max_new_tokens: int = 225,
) -> list[str]:
    """Run inference over a dataset (with an 'audio' column) -> hypotheses.

    Shared by the zero-shot baseline and the fine-tuned evaluation so both use
    the exact same decoding path. Works for a plain Whisper model or a PEFT model.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # transformers v5 loads checkpoints in native precision (fp16 for turbo);
    # features come out float32 — cast them to the model's dtype or conv1 crashes.
    model_dtype = next(model.parameters()).dtype
    # Our max_new_tokens is the only length cap we want; drop the factory
    # max_length=448 so transformers stops warning once per batch.
    model.generation_config.max_length = None

    from tqdm import tqdm

    hyps: list[str] = []
    for start in tqdm(range(0, ds.num_rows, batch_size),
                      desc="transcribe", unit="batch"):
        batch = ds[start : start + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        sr = batch["audio"][0]["sampling_rate"]
        features = processor.feature_extractor(
            arrays, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(device=device, dtype=model_dtype)
        with torch.no_grad():
            # language/task kwargs are the current Whisper API; the old
            # forced_decoder_ids path was removed in transformers v5.
            generated = model.generate(
                input_features=features,
                language=language,
                task=task,
                max_new_tokens=max_new_tokens,
            )
        hyps.extend(processor.batch_decode(generated, skip_special_tokens=True))
    return hyps
