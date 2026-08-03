"""Step 12 — Synthesize the fixed eval set with a fine-tuned StyleTTS 2 checkpoint.

The fine-tuned model takes the SAME locked exam as the zero-shot baselines
(evalset.csv, 3 reference voices) and writes 09-compatible outputs, so scoring
is unchanged: `10_tts_score.py --model styletts2-ft`.

Inference replicates the official Demo/Inference_LibriTTS.ipynb verbatim
(compute_style, diffusion sampler, duration/F0 prediction, [..:-50] tail trim;
verified 2026-08-03). Cloning defaults = the demo's reference-cloning values
(alpha 0.3, beta 0.7, 5 diffusion steps, embedding_scale 1). Per-clip seeding
(crc32 of eval_id|voice) makes runs reproducible AND resume-safe.

Runs in the StyleTTS2 venv on the pod (venv-stts + `pip install phonemizer`;
system espeak-ng + nltk punkt already required by training).

Usage (from /workspace/naija-speech):
    python scripts/12_styletts2_synthesize.py \
        --checkpoint /workspace/StyleTTS2/Models/naija_ft/epoch_2nd_00014.pth
    python scripts/12_styletts2_synthesize.py --checkpoint ... --limit 5   # smoke
    python scripts/12_styletts2_synthesize.py --checkpoint ... \
        --evalset outputs/tts_eval/evalset_clinical.csv --tag clinical

Outputs: outputs/tts_eval/styletts2-ft[-<tag>]/wav/*.wav + manifest.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

# repo checkpoints are 2024-era full pickles; torch>=2.6 refuses them without this
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(ROOT, "outputs", "tts_eval")
SR = 24000


def load_styletts2(args):
    """Build models + sampler exactly as the official inference notebook does."""
    os.chdir(args.styletts2_dir)  # repo code resolves Utils/... paths relatively
    sys.path.insert(0, args.styletts2_dir)

    import torch
    import yaml
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch
    from text_utils import TextCleaner
    from Utils.PLBERT.util import load_plbert
    from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = yaml.safe_load(open(args.config))
    text_aligner = load_ASR_models(config.get("ASR_path"), config.get("ASR_config"))
    pitch_extractor = load_F0_models(config.get("F0_path"))
    plbert = load_plbert(config.get("PLBERT_dir"))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    params = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["net"]
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception:  # DataParallel-saved: strip the `module.` prefix
                from collections import OrderedDict
                model[key].load_state_dict(OrderedDict(
                    (k[7:], v) for k, v in params[key].items()), strict=False)
    for key in model:
        model[key].eval()
        model[key].to(device)

    sampler = DiffusionSampler(
        model.diffusion.diffusion, sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False)
    return model, model_params, sampler, TextCleaner(), device


def make_speak(model, model_params, sampler, textcleaner, device, args):
    import librosa
    import numpy as np
    import phonemizer
    import torch
    import torchaudio
    from nltk.tokenize import word_tokenize

    phon = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True)
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0) \
            .expand(lengths.shape[0], -1).type_as(lengths)
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def compute_style(path):
        wave, _ = librosa.load(path, sr=SR)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        mel = to_mel(torch.from_numpy(audio).float())
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std
        mel = mel.to(device)
        with torch.no_grad():
            ref_s = model.style_encoder(mel.unsqueeze(1))
            ref_p = model.predictor_encoder(mel.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def infer(text, ref_s):
        ps = phon.phonemize([text.strip()])
        ps = " ".join(word_tokenize(ps[0]))
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            s_pred = sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                embedding=bert_dur, embedding_scale=args.embedding_scale,
                features=ref_s, num_steps=args.diffusion_steps).squeeze(1)
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = args.alpha * ref + (1 - args.alpha) * ref_s[:, :128]
            s = args.beta * s + (1 - args.beta) * ref_s[:, 128:]
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c = 0
            for i in range(pred_aln.size(0)):
                pred_aln[i, c:c + int(pred_dur[i].data)] = 1
                c += int(pred_dur[i].data)
            en = d.transpose(-1, -2) @ pred_aln.unsqueeze(0).to(device)
            if model_params.decoder.type == "hifigan":  # one-frame shift quirk
                shifted = torch.zeros_like(en)
                shifted[:, :, 0] = en[:, :, 0]
                shifted[:, :, 1:] = en[:, :, :-1]
                en = shifted
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            asr = t_en @ pred_aln.unsqueeze(0).to(device)
            if model_params.decoder.type == "hifigan":
                shifted = torch.zeros_like(asr)
                shifted[:, :, 0] = asr[:, :, 0]
                shifted[:, :, 1:] = asr[:, :, :-1]
                asr = shifted
            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]  # upstream: end-pulse trim

    return compute_style, infer


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuned StyleTTS2 over the fixed eval set.")
    ap.add_argument("--checkpoint", required=True, help="Fine-tuned .pth (best epoch).")
    ap.add_argument("--styletts2-dir", default="/workspace/StyleTTS2")
    ap.add_argument("--config", default=None,
                    help="Defaults to <styletts2-dir>/Models/naija_ft/config_ft.yml")
    ap.add_argument("--run-name", default="styletts2-ft")
    ap.add_argument("--tag", default="", help="e.g. clinical -> styletts2-ft-clinical/")
    ap.add_argument("--evalset", default=os.path.join(EVAL_DIR, "evalset.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--embedding-scale", type=float, default=1.0)
    args = ap.parse_args()
    args.config = args.config or os.path.join(
        args.styletts2_dir, "Models", "naija_ft", "config_ft.yml")

    import pandas as pd
    import soundfile as sf
    import torch

    evalset = pd.read_csv(args.evalset)
    if args.limit:
        evalset = evalset.head(args.limit)
    refs = pd.read_csv(os.path.join(EVAL_DIR, "ref_voices.csv"))
    refs["wav"] = refs["wav"].str.replace("\\", "/", regex=False)
    refs["wav"] = refs["wav"].map(lambda w: os.path.join(ROOT, w))
    voices = {r["macro_accent"].lower(): r["wav"] for _, r in refs.iterrows()}

    run_name = args.run_name + (f"-{args.tag}" if args.tag else "")
    wav_dir = os.path.join(EVAL_DIR, run_name, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"[{run_name}] loading StyleTTS2 from {args.checkpoint} ...")
    model, model_params, sampler, tc, device = load_styletts2(args)
    compute_style, infer = make_speak(model, model_params, sampler, tc, device, args)
    styles = {v: compute_style(p) for v, p in voices.items()}
    print(f"[{run_name}] {len(evalset)} sentences x {len(styles)} voices -> {wav_dir}")

    rows, done = [], 0
    for _, s in evalset.iterrows():
        for voice, ref_s in styles.items():
            out = os.path.join(wav_dir, f"{s['eval_id']}__{voice}.wav")
            synth_sec = None
            if not os.path.exists(out):
                torch.manual_seed(zlib.crc32(f"{s['eval_id']}|{voice}".encode()))
                t0 = time.perf_counter()
                try:
                    wav = infer(str(s["text"]), ref_s)
                except Exception as e:
                    print(f"  FAIL {s['eval_id']}/{voice}: {e}")
                    continue
                synth_sec = round(time.perf_counter() - t0, 2)
                sf.write(out, wav, SR)
            info = sf.info(out)
            rows.append({"eval_id": s["eval_id"], "voice": voice, "wav":
                         os.path.relpath(out, ROOT).replace("\\", "/"),
                         "text": s["text"], "macro_accent": s["macro_accent"],
                         "audio_sec": round(info.duration, 2),
                         "sample_rate": info.samplerate, "synth_sec": synth_sec})
        done += 1
        if done % 20 == 0:
            print(f"  ... {done}/{len(evalset)} sentences")

    mpath = os.path.join(EVAL_DIR, run_name, "manifest.csv")
    pd.DataFrame(rows).to_csv(mpath, index=False)
    timed = [r for r in rows if r["synth_sec"]]
    if timed:
        rtf = sum(r["synth_sec"] for r in timed) / sum(r["audio_sec"] for r in timed)
        print(f"[{run_name}] RTF (this run): {rtf:.2f}")
    print(f"[{run_name}] {len(rows)} clips -> {mpath}")


if __name__ == "__main__":
    main()
