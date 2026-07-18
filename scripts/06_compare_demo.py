"""Step 6 — side-by-side demo: zero-shot vs fine-tuned on the same audio.

Loads the base Whisper model and the base+LoRA-adapter model, transcribes the
same clips through both, and prints reference / before / after per clip. This is
the qualitative companion to 04_evaluate's numbers — for eyeballing, thesis
examples, and the defense demo.

Usage:
    python scripts/06_compare_demo.py                       # 5 random test clips
    python scripts/06_compare_demo.py --n 10 --seed 7
    python scripts/06_compare_demo.py --audio my_voice.wav  # any 16kHz-able file
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402
from whisper_lora import transcribe_dataset  # noqa: E402


def _load_models(cfg, adapter_dir):
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(adapter_dir)
    base = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
    tuned_base = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"])
    tuned = PeftModel.from_pretrained(tuned_base, adapter_dir)
    return processor, base, tuned


def _transcribe_file(path, processor, model):
    """Single-file path: load audio -> features -> generate (mirrors transcribe_dataset)."""
    import librosa
    import torch

    audio, _ = librosa.load(path, sr=16000, mono=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    dtype = next(model.parameters()).dtype
    model.generation_config.max_length = None
    feats = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model.generate(input_features=feats, language="english",
                             task="transcribe", max_new_tokens=225)
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side zero-shot vs fine-tuned demo.")
    ap.add_argument("--data-config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--model-config", default="configs/stt_whisper_large_v3_turbo_hf.yaml")
    ap.add_argument("--adapter-dir", default=None,
                    help="Defaults to <model output_dir>/adapter.")
    ap.add_argument("--n", type=int, default=5, help="Number of random test clips.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio", default=None,
                    help="Transcribe ONE local audio file instead of test clips.")
    args = ap.parse_args()

    load_dotenv()
    data_cfg = load_yaml(args.data_config)
    cfg = load_yaml(args.model_config)
    adapter_dir = args.adapter_dir or os.path.join(cfg["output_dir"], "adapter")

    processor, base, tuned = _load_models(cfg, adapter_dir)

    if args.audio:
        print(f"\n=== {args.audio} ===")
        print(f"  zero-shot : {_transcribe_file(args.audio, processor, base)}")
        print(f"  fine-tuned: {_transcribe_file(args.audio, processor, tuned)}")
        return

    from curate import filter_duration, load_curated

    ds = load_curated(data_cfg["hf_curated_repo"], split="test")
    ds = filter_duration(ds, data_cfg.get("min_duration_sec", 0.5),
                         data_cfg.get("max_duration_sec", 30.0), label="test")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, ds.num_rows)))

    before = transcribe_dataset(base, processor, ds, batch_size=args.n,
                                language=cfg["language"], task=cfg["task"])
    after = transcribe_dataset(tuned, processor, ds, batch_size=args.n,
                               language=cfg["language"], task=cfg["task"])

    for i in range(ds.num_rows):
        print(f"\n=== clip {i} [{ds[i]['macro_accent']} | {ds[i]['domain']}] ===")
        print(f"  reference : {ds[i]['text_raw']}")
        print(f"  zero-shot : {before[i]}")
        print(f"  fine-tuned: {after[i]}")


if __name__ == "__main__":
    main()
