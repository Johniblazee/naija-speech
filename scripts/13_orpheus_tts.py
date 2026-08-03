"""Step 13 — Orpheus-3B QLoRA fine-tune (TTS Phase 3): prepare -> train -> synthesize.

Codec-LM paradigm representative. Recipe verified 2026-08-03 against Unsloth's
official Orpheus notebook (unslothai/notebooks Orpheus_(3B)-TTS) + canopylabs
model cards: token ids, SNAC flatten/unflatten offsets, prompt layout, and
generation params are copied verbatim from those sources.

Stages (each resumable; run in venv-orpheus on the pod):
  prepare     Phase-2's staged 24 kHz wavs + selection -> SNAC-tokenized HF
              dataset. Multi-speaker conditioning is lexical ("{voice}: {text}");
              the 3 benchmark reference speakers become named voices
              (yoruba/igbo/hausa — same voices every other system was tested
              with), topped up with their best staged clips; all other speakers
              get short hash names (accent diversity for the accent-adaptation).
  train       Unsloth 4-bit QLoRA on unsloth/orpheus-3b-0.1-ft: r=64, bs 1 x
              grad-accum 4, lr 2e-4, 1 epoch (notebook values).
  synthesize  The locked exam (evalset.csv / clinical) with the 3 named voices;
              09/12-compatible outputs -> score with 10_tts_score.py.

Installs (venv-orpheus):
  pip install unsloth && pip install transformers==4.56.2 && \
  pip install --no-deps trl==0.22.2 && \
  pip install snac "datasets>=3.4.1,<4.0.0" soundfile pandas

Usage (from /workspace/naija-speech):
  python scripts/13_orpheus_tts.py --stage prepare
  python scripts/13_orpheus_tts.py --stage train [--epochs 1] [--resume ckpt]
  python scripts/13_orpheus_tts.py --stage synthesize [--limit 5]
  python scripts/13_orpheus_tts.py --stage synthesize \
      --evalset outputs/tts_eval/evalset_clinical.csv --tag clinical
  python scripts/13_orpheus_tts.py --self-test        # offset arithmetic, no GPU
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FT_DATA = os.path.join(ROOT, "outputs", "tts_ft_data")
EVAL_DIR = os.path.join(ROOT, "outputs", "tts_eval")
DATA_DIR = os.path.join(ROOT, "outputs", "orpheus_data")
OUT_DIR = os.path.join(ROOT, "outputs", "orpheus_ft")
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"
SR = 24000

# special token ids — verbatim from the Unsloth notebook
SOT, EOT = 128000, 128009
SOS, EOS = 128257, 128258          # start/end of speech
SOH, EOH = 128259, 128260          # start/end of human turn
SOA, EOA = 128261, 128262          # start/end of ai turn
PAD = 128263
AUDIO_BASE = 128266                # + slot*4096 per position in the 7-token frame
MAX_SEQ = 2048


def flatten_codes(c1, c2, c3):
    """3 SNAC layers -> flat token list (7 per frame, verbatim offset scheme)."""
    out = []
    for i in range(len(c1)):
        out += [c1[i] + AUDIO_BASE,
                c2[2 * i] + AUDIO_BASE + 4096,
                c3[4 * i] + AUDIO_BASE + 2 * 4096,
                c3[4 * i + 1] + AUDIO_BASE + 3 * 4096,
                c2[2 * i + 1] + AUDIO_BASE + 4 * 4096,
                c3[4 * i + 2] + AUDIO_BASE + 5 * 4096,
                c3[4 * i + 3] + AUDIO_BASE + 6 * 4096]
    return out


def redistribute_codes(code_list):
    """Flat (AUDIO_BASE already subtracted) -> 3 SNAC layers (verbatim)."""
    l1, l2, l3 = [], [], []
    for i in range(len(code_list) // 7):
        l1.append(code_list[7 * i])
        l2.append(code_list[7 * i + 1] - 4096)
        l3.append(code_list[7 * i + 2] - 2 * 4096)
        l3.append(code_list[7 * i + 3] - 3 * 4096)
        l2.append(code_list[7 * i + 4] - 4 * 4096)
        l3.append(code_list[7 * i + 5] - 5 * 4096)
        l3.append(code_list[7 * i + 6] - 6 * 4096)
    return l1, l2, l3


def remove_duplicate_frames(codes):
    """Drop a 7-token frame when its layer-1 code repeats the previous frame's."""
    assert len(codes) % 7 == 0
    out = codes[:7]
    for i in range(7, len(codes), 7):
        if codes[i] != out[-7]:
            out += codes[i:i + 7]
    return out


def _voice_names():
    """speaker hash -> voice name; the 3 benchmark refs get their accent names."""
    import pandas as pd

    refs = pd.read_csv(os.path.join(EVAL_DIR, "ref_voices.csv"))
    return {r["speaker"]: r["macro_accent"].lower() for _, r in refs.iterrows()}


def stage_prepare(args):
    import pandas as pd
    import soundfile as sf
    import torch
    from datasets import Dataset
    from snac import SNAC
    from transformers import AutoTokenizer

    sel = pd.read_csv(os.path.join(FT_DATA, "selection.csv"))
    ref_names = _voice_names()

    # top up the 3 reference speakers with their best staged clips (canopy:
    # >=50/speaker works, ~300 is ideal) so they become strong named voices
    cand = pd.read_csv(os.path.join(FT_DATA, "candidates.csv"))
    scores = pd.read_csv(os.path.join(FT_DATA, "scores.csv")).dropna(subset=["mos"])
    pool = scores.merge(cand.reset_index().rename(columns={"index": "idx"}), on="idx")
    pool = pool.rename(columns={"backfilled_speaker": "speaker"})
    extra = pool[pool["speaker"].isin(ref_names) & ~pool["idx"].isin(sel["idx"])]
    extra = (extra.sort_values("mos", ascending=False)
             .groupby("speaker").head(args.ref_topup))
    df = pd.concat([sel, extra], ignore_index=True)
    df["voice"] = df["speaker"].map(
        lambda s: ref_names.get(s, f"s{zlib.crc32(str(s).encode()) & 0xffffff:06x}"))
    print(f"[prepare] {len(df):,} clips | ref-voice clips: "
          f"{df['voice'].isin(ref_names.values()).sum():,} "
          f"({df[df.voice.isin(ref_names.values())].groupby('voice').size().to_dict()})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    from tqdm import tqdm

    rows, dropped = [], 0
    for r in tqdm(df.itertuples(), total=len(df), desc="snac-encode"):
        wav_path = r.wav if os.path.isabs(r.wav) else os.path.join(ROOT, r.wav)
        x, sr = sf.read(wav_path, dtype="float32")
        w = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # 24k mono already
        with torch.inference_mode():
            codes = snac.encode(w)
        flat = flatten_codes(codes[0][0].tolist(), codes[1][0].tolist(),
                             codes[2][0].tolist())
        flat = remove_duplicate_frames(flat)
        text_ids = tok.encode(f"{r.voice}: {str(r.text_raw)}", add_special_tokens=True)
        ids = ([SOH] + text_ids + [EOT, EOH] + [SOA, SOS] + flat + [EOS, EOA])
        if len(ids) > MAX_SEQ:
            dropped += 1
            continue
        rows.append({"input_ids": ids, "labels": ids,
                     "attention_mask": [1] * len(ids)})
    print(f"[prepare] {len(rows):,} examples (dropped {dropped} > {MAX_SEQ} tokens)")
    Dataset.from_list(rows).save_to_disk(DATA_DIR)
    print(f"[prepare] -> {DATA_DIR}")


def stage_train(args):
    from unsloth import FastLanguageModel  # must import before transformers bits
    import torch  # noqa: F401
    from datasets import load_from_disk
    from transformers import Trainer, TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=64, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False)

    ds = load_from_disk(DATA_DIR)
    report = ["wandb"] if os.environ.get("WANDB_PROJECT") else "none"
    trainer = Trainer(
        model=model, train_dataset=ds,
        args=TrainingArguments(
            per_device_train_batch_size=1,   # notebook: bs>1 needs a collator; 1 is the safe verbatim path
            gradient_accumulation_steps=4, warmup_steps=5,
            num_train_epochs=args.epochs, learning_rate=2e-4,
            logging_steps=25, optim="adamw_8bit", weight_decay=0.001,
            lr_scheduler_type="linear", seed=3407, fp16=False, bf16=True,
            output_dir=OUT_DIR, save_strategy="epoch", report_to=report))
    trainer.train(resume_from_checkpoint=args.resume or None)
    model.save_pretrained(os.path.join(OUT_DIR, "adapter"))
    tokenizer.save_pretrained(os.path.join(OUT_DIR, "adapter"))
    print(f"[train] adapter -> {OUT_DIR}/adapter")


def stage_synthesize(args):
    from unsloth import FastLanguageModel
    import pandas as pd
    import soundfile as sf
    import torch
    from snac import SNAC

    adapter = args.adapter or os.path.join(OUT_DIR, "adapter")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    voices = sorted(_voice_names().values())  # yoruba / igbo / hausa

    evalset = pd.read_csv(args.evalset)
    if args.limit:
        evalset = evalset.head(args.limit)
    run_name = "orpheus-ft" + (f"-{args.tag}" if args.tag else "")
    wav_dir = os.path.join(EVAL_DIR, run_name, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    print(f"[{run_name}] {len(evalset)} sentences x {len(voices)} voices -> {wav_dir}")

    rows, done = [], 0
    for _, s in evalset.iterrows():
        for voice in voices:
            out = os.path.join(wav_dir, f"{s['eval_id']}__{voice}.wav")
            synth_sec = None
            if not os.path.exists(out):
                torch.manual_seed(zlib.crc32(f"{s['eval_id']}|{voice}".encode()))
                prompt = tokenizer.encode(f"{voice}: {str(s['text']).strip()}",
                                          add_special_tokens=True)
                ids = torch.tensor([[SOH] + prompt + [EOT, EOH]], device=device)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                        eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
                seq = gen[0].tolist()
                if SOS not in seq:
                    print(f"  FAIL {s['eval_id']}/{voice}: no speech tokens")
                    continue
                codes = seq[len(seq) - seq[::-1].index(SOS):]  # after LAST SOS
                codes = [c - AUDIO_BASE for c in codes
                         if c not in (EOS, EOA, PAD) and c >= AUDIO_BASE]
                codes = codes[: (len(codes) // 7) * 7]
                if not codes:
                    print(f"  FAIL {s['eval_id']}/{voice}: empty codes")
                    continue
                l1, l2, l3 = redistribute_codes(codes)
                with torch.inference_mode():
                    audio = snac.decode([torch.tensor(l).unsqueeze(0).to(device)
                                         for l in (l1, l2, l3)])
                synth_sec = round(time.perf_counter() - t0, 2)
                sf.write(out, audio.squeeze().cpu().numpy(), SR)
            info = sf.info(out)
            rows.append({"eval_id": s["eval_id"], "voice": voice,
                         "wav": os.path.relpath(out, ROOT).replace("\\", "/"),
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


def self_test():
    import random

    rnd = random.Random(7)
    n = 12
    c1 = [rnd.randrange(4096) for _ in range(n)]
    c2 = [rnd.randrange(4096) for _ in range(2 * n)]
    c3 = [rnd.randrange(4096) for _ in range(4 * n)]
    flat = flatten_codes(c1, c2, c3)
    assert len(flat) == 7 * n and all(t >= AUDIO_BASE for t in flat)
    r1, r2, r3 = redistribute_codes([t - AUDIO_BASE for t in flat])
    assert (r1, r2, r3) == (c1, c2, c3), "flatten/unflatten must be inverses"

    dup = flatten_codes([5, 5, 9], [1, 2, 3, 4, 5, 6], [1] * 12)
    ded = remove_duplicate_frames(dup)
    assert len(ded) == 14, len(ded)  # middle frame (repeat layer-1 code 5) dropped
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orpheus-3B QLoRA (TTS Phase 3).")
    ap.add_argument("--stage", default=None,
                    choices=("prepare", "train", "synthesize"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--ref-topup", type=int, default=300,
                    help="Max extra best-MOS clips per reference speaker.")
    ap.add_argument("--evalset", default=os.path.join(EVAL_DIR, "evalset.csv"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2400,
                    help="~84 audio tokens/s -> 2400 = ~28 s generation cap.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "synthesize":
        stage_synthesize(args)
    else:
        raise SystemExit("--stage prepare|train|synthesize (or --self-test)")


if __name__ == "__main__":
    main()
