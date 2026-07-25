"""Step 9 — Synthesize the fixed eval set with ONE zero-shot TTS baseline.

Reads outputs/tts_eval/evalset.csv (+ ref_voices.csv for the cloning models) and
writes one WAV per (sentence, voice) plus a manifest with per-clip synthesis time
(RTF feeds the Phase-4 efficiency scorecard). Idempotent: existing WAVs are kept,
so a killed pod run resumes by re-running the same command.

EACH MODEL FAMILY RUNS IN ITS OWN VENV (deps conflict — STT lesson). Installs,
verified 2026-07 against the official READMEs:

  xtts      pip install coqui-tts               (XTTS-v2; ~2 GB weights on first run)
  yarngpt   git clone https://github.com/saheedniyi02/yarngpt.git
            pip install outetts==0.2.3 uroman torch torchaudio transformers gdown
            + WavTokenizer config (HF: novateur/WavTokenizer-medium-speech-75token)
            + checkpoint wavtokenizer_large_speech_320_24k.ckpt (gdown, see README)
  qwen3tts  pip install -U qwen-tts             (open weights since 2026-01, Apache-2.0)

Usage:
    python scripts/09_tts_synthesize.py --model xtts
    python scripts/09_tts_synthesize.py --model yarngpt --yarngpt-dir third_party/yarngpt
    python scripts/09_tts_synthesize.py --model qwen3tts
    python scripts/09_tts_synthesize.py --model xtts --limit 5   # smoke test

Outputs: outputs/tts_eval/<model>/wav/<eval_id>__<voice>.wav + manifest.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

EVAL_DIR = "outputs/tts_eval"

# YarnGPT preset voices (its README list); 3 fixed = same count as cloned voices.
YARNGPT_VOICES = ("idera", "tayo", "zainab")


def _ref_voices():
    import pandas as pd

    path = os.path.join(EVAL_DIR, "ref_voices.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run scripts/08_build_tts_evalset.py first")
    return pd.read_csv(path).to_dict("records")


# --------------------------------------------------------------------------- #
# Backends: each returns (voices, speak) where speak(text, voice, out_path)
# writes the WAV and returns nothing. Imports stay inside so each venv only
# needs its own model's dependencies.
# --------------------------------------------------------------------------- #
def _load_xtts(args):
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    refs = {r["macro_accent"].lower(): r["wav"] for r in _ref_voices()}

    def speak(text, voice, out_path):
        tts.tts_to_file(text=text, speaker_wav=refs[voice], language="en",
                        file_path=out_path)

    return list(refs), speak


def _load_yarngpt(args):
    import torch  # noqa: F401 — device selection happens inside AudioTokenizer
    import torchaudio
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, args.yarngpt_dir)  # provides yarngpt.audiotokenizer
    from yarngpt.audiotokenizer import AudioTokenizer

    hf_path = "saheedniyi/YarnGPT"
    tok = AudioTokenizer(hf_path, args.wavtok_ckpt, args.wavtok_config)
    model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype="auto").to(tok.device)

    def speak(text, voice, out_path):
        prompt = tok.create_prompt(text, speaker_name=voice)
        ids = tok.tokenize_prompt(prompt)
        out = model.generate(input_ids=ids, temperature=0.1,
                             repetition_penalty=1.1, max_length=4000)
        audio = tok.get_audio(tok.get_codes(out))
        torchaudio.save(out_path, audio, sample_rate=24000)

    return list(YARNGPT_VOICES), speak


def _load_qwen3tts(args):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.bfloat16)
    refs = {r["macro_accent"].lower(): r for r in _ref_voices()}

    def speak(text, voice, out_path):
        r = refs[voice]
        wavs, sr = model.generate_voice_clone(
            text=text, language="English",
            ref_audio=r["wav"], ref_text=r["ref_text"])
        sf.write(out_path, wavs[0], sr)

    return list(refs), speak


BACKENDS = {"xtts": _load_xtts, "yarngpt": _load_yarngpt, "qwen3tts": _load_qwen3tts}


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot TTS synthesis over the fixed eval set.")
    ap.add_argument("--model", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--limit", type=int, default=None, help="First N sentences only.")
    ap.add_argument("--voices", type=int, default=3, help="Cap number of voices.")
    ap.add_argument("--yarngpt-dir", default="third_party/yarngpt")
    ap.add_argument("--wavtok-ckpt", default="third_party/wavtokenizer_large_speech_320_24k.ckpt")
    ap.add_argument("--wavtok-config",
                    default="third_party/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    args = ap.parse_args()

    import pandas as pd
    import soundfile as sf

    evalset = pd.read_csv(os.path.join(EVAL_DIR, "evalset.csv"))
    if args.limit:
        evalset = evalset.head(args.limit)

    voices, speak = BACKENDS[args.model](args)
    voices = voices[: args.voices]
    wav_dir = os.path.join(EVAL_DIR, args.model, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    print(f"[{args.model}] {len(evalset)} sentences x {len(voices)} voices -> {wav_dir}")

    rows, done = [], 0
    for _, s in evalset.iterrows():
        for voice in voices:
            out = os.path.join(wav_dir, f"{s['eval_id']}__{voice}.wav")
            synth_sec = None
            if not os.path.exists(out):
                t0 = time.perf_counter()
                try:
                    speak(str(s["text"]), voice, out)
                except Exception as e:  # keep going; scoring drops missing clips
                    print(f"  FAIL {s['eval_id']}/{voice}: {e}")
                    continue
                synth_sec = round(time.perf_counter() - t0, 2)
            info = sf.info(out)
            rows.append({"eval_id": s["eval_id"], "voice": voice, "wav": out,
                         "text": s["text"], "macro_accent": s["macro_accent"],
                         "audio_sec": round(info.duration, 2),
                         "sample_rate": info.samplerate, "synth_sec": synth_sec})
        done += 1
        if done % 20 == 0:
            print(f"  ... {done}/{len(evalset)} sentences")

    mpath = os.path.join(EVAL_DIR, args.model, "manifest.csv")
    pd.DataFrame(rows).to_csv(mpath, index=False)
    timed = [r for r in rows if r["synth_sec"]]
    if timed:
        rtf = sum(r["synth_sec"] for r in timed) / sum(r["audio_sec"] for r in timed)
        print(f"[{args.model}] RTF (synth_sec/audio_sec, this run): {rtf:.2f}")
    print(f"[{args.model}] {len(rows)} clips -> {mpath}")


if __name__ == "__main__":
    main()
