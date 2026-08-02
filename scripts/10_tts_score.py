"""Step 10 — Score one TTS model's synthesized eval set.

Two automatic metrics per the TTS plan:
  1. Resynthesis-WER: transcribe each synthesized clip and compare against the
     input sentence. Judged by OUR fine-tuned Whisper (intelligibility *as
     Nigerian-accented speech* — the STT half of the thesis becomes the meter
     for the TTS half) AND by base Whisper as the neutral reference.
  2. UTMOSv2 naturalness (predicted MOS; documented English-training caveat).
     Soft dependency: pip install git+https://github.com/sarulab-speech/UTMOSv2.git

Runs in the MAIN project venv (Whisper stack) — not the synthesis venvs; it only
reads the WAVs the synthesis step produced.

Usage:
    python scripts/10_tts_score.py --model xtts
    python scripts/10_tts_score.py --model yarngpt --skip-utmos
    python scripts/10_tts_score.py --model qwen3tts --judges ft

Outputs (outputs/tts_eval/<model>/):
    scores.csv    per-clip: hypotheses + WER per judge + predicted MOS
    results.csv   aggregate: overall / per-voice / per-accent, per judge
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402
from metrics import compute_stratified, compute_wer_cer  # noqa: E402

EVAL_DIR = "outputs/tts_eval"
# Local unpacked RunPod bundle — fallback when the pod-side output_dir is absent.
BUNDLE_ADAPTER = "outputs/stt_whisper_turbo_results/outputs/whisper_turbo_hf/adapter"


def _demo():
    """Model loading + single-file transcription live in script 06 — reuse them."""
    p = os.path.join(os.path.dirname(__file__), "06_compare_demo.py")
    spec = importlib.util.spec_from_file_location("compare_demo", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _utmos_scores(wav_dir: str) -> dict[str, float]:
    """basename -> predicted MOS; {} if utmosv2 is not installed."""
    try:
        import utmosv2
    except ImportError:
        print("[utmos] utmosv2 not installed — skipping naturalness scores")
        return {}
    model = utmosv2.create_model(pretrained=True)
    results = model.predict(input_dir=wav_dir)
    return {os.path.basename(r["file_path"]): round(float(r["predicted_mos"]), 3)
            for r in results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score synthesized TTS clips.")
    ap.add_argument("--model", required=True, help="Subdir of outputs/tts_eval/.")
    ap.add_argument("--model-config", default="configs/stt_whisper_large_v3_turbo_hf.yaml")
    ap.add_argument("--adapter-dir", default=None,
                    help="Defaults to <model output_dir>/adapter, else the local bundle.")
    ap.add_argument("--judges", default="both", choices=("both", "ft", "base"))
    ap.add_argument("--skip-utmos", action="store_true")
    ap.add_argument("--utmos-only", action="store_true",
                    help="Only add predicted MOS to an existing scores.csv.")
    args = ap.parse_args()

    import pandas as pd

    mdir = os.path.join(EVAL_DIR, args.model)
    if args.utmos_only:  # add MOS to an existing scores.csv, no re-transcription
        try:
            import utmosv2  # noqa: F401 — this mode's ONLY job is MOS: fail loudly
        except ImportError:
            raise SystemExit('utmosv2 not installed — pip install '
                             '"git+https://github.com/sarulab-speech/UTMOSv2.git"')
        spath = os.path.join(mdir, "scores.csv")
        scores = pd.read_csv(spath)
        mos = _utmos_scores(os.path.join(mdir, "wav"))
        scores["predicted_mos"] = scores["wav"].map(
            lambda w: mos.get(os.path.basename(w)))
        scores.to_csv(spath, index=False)
        print(f"[score] mean predicted MOS: {scores['predicted_mos'].mean():.2f} -> {spath}")
        return

    manifest = pd.read_csv(os.path.join(mdir, "manifest.csv"))
    manifest = manifest[manifest["wav"].map(os.path.exists)].reset_index(drop=True)
    print(f"[score] {len(manifest)} clips from {mdir}")

    load_dotenv()
    cfg = load_yaml(args.model_config)
    adapter = args.adapter_dir or os.path.join(cfg["output_dir"], "adapter")
    if not os.path.isdir(adapter):
        adapter = BUNDLE_ADAPTER
    print(f"[score] adapter: {adapter}")

    demo = _demo()
    processor, base, tuned = demo._load_models(cfg, adapter)
    judges = {"ft": tuned, "base": base}
    if args.judges != "both":
        judges = {args.judges: judges[args.judges]}

    for name, model in judges.items():
        print(f"[score] transcribing with judge={name} ...")
        hyps = []
        for i, wav in enumerate(manifest["wav"]):
            hyps.append(demo._transcribe_file(wav, processor, model))
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(manifest)}")
        manifest[f"hyp_{name}"] = hyps
        manifest[f"wer_{name}"] = [
            compute_wer_cer([r], [h])["wer"]
            for r, h in zip(manifest["text"], hyps)]

    # persist transcriptions BEFORE the optional MOS step — they cost GPU-hours,
    # MOS costs minutes and can be re-added anytime with --utmos-only
    manifest.to_csv(os.path.join(mdir, "scores.csv"), index=False)

    if not args.skip_utmos:
        try:
            mos = _utmos_scores(os.path.join(mdir, "wav"))
        except Exception as e:
            print(f"[utmos] failed ({e}) — MOS skipped; transcriptions are saved. "
                  f"Add later with --utmos-only in a torch>=2.6 env.")
            mos = {}
        if mos:
            manifest["predicted_mos"] = manifest["wav"].map(
                lambda w: mos.get(os.path.basename(w)))
            manifest.to_csv(os.path.join(mdir, "scores.csv"), index=False)

    agg = []
    for name in judges:
        rows = manifest.rename(columns={"text": "reference",
                                        f"hyp_{name}": "hypothesis"}).to_dict("records")
        for r in compute_stratified(rows, group_keys=("voice", "macro_accent")):
            agg.append({"judge": name, **r})
    results = pd.DataFrame(agg)
    if "predicted_mos" in manifest:
        results.loc[len(results)] = {"judge": "utmosv2", "group": "overall",
                                     "key": "all", "wer": None, "cer": None,
                                     "n": manifest["predicted_mos"].notna().sum()}
        print(f"[score] mean predicted MOS: {manifest['predicted_mos'].mean():.2f}")
    results.to_csv(os.path.join(mdir, "results.csv"), index=False)
    print(f"[score] -> {mdir}/scores.csv + results.csv")
    print(results[results["group"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
