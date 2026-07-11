# naija-speech

Fine-tuning and benchmarking open **Speech-to-Text (STT)** and **Text-to-Speech (TTS)**
models for **Nigerian-accented English** — the implementation artifact for the MIT thesis
*"Speech-to-Text & Text-to-Speech for Nigerian English Accents"* (John Ikpe Michael,
MIVA Open University).

This repo follows the thesis's **Design Science Research** plan: build a curated corpus,
fine-tune models with parameter-efficient methods (LoRA), and evaluate them on three axes —
**accuracy**, **fairness** (per macro-accent), and **production efficiency**.

> **Status:** corpus-first vertical slice. The first milestone is one STT model
> (Whisper) fine-tuned with LoRA on the Nigerian subset of AfriSpeech-200, evaluated against
> its zero-shot baseline. TTS comes after the STT slice works end-to-end.

## TTS roster (locked 2026-07-11; code lands at TTS kickoff)

- **Fine-tuned (3):** StyleTTS 2 (primary, style-diffusion, MIT) → Orpheus-3B (codec-LM,
  Apache-2.0, Unsloth 4-bit QLoRA — same toolchain as the STT slice) → F5-TTS (flow-matching
  DiT, stretch; MIT code / CC-BY-NC weights).
- **Zero-shot baselines (3):** YarnGPT (Nigerian domain baseline), Qwen3-TTS, XTTS-v2
  (demoted from fine-tune slot: CPML non-commercial).

Compute tiers: free Colab T4 for dev + 4-bit QLoRA runs → RunPod Community A5000 24GB
(~$0.27/hr) / A40 48GB (~$0.44/hr) for most fine-tunes → A100-80 (~$1.40–1.65/hr RunPod, or
HF Jobs per-minute) for final citable runs only.

---

## Why "vertical slice" first

Instead of building all 3 STT + 3 TTS models at once, we prove the **entire pipeline** on one
model first: `corpus → zero-shot baseline → LoRA fine-tune → accent-stratified evaluation`.
Once that works, adding more architectures is repetition, not risk. This is the de-risking
strategy described in the thesis methodology.

## Pipeline (run in order)

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `scripts/00_eda.py` | EDA on the Nigerian subset (metadata only — no audio): accent/domain/gender distributions, disparity, report + figures. |
| 1 | `scripts/01_build_corpus.py` | Download Nigerian AfriSpeech-200 accents, normalize text, build a manifest + speaker-disjoint splits, save to disk. |
| 2 | `scripts/02_zeroshot_baseline.py` | Measure the **gap**: zero-shot Whisper WER/CER on the Nigerian test set, stratified by accent and domain. |
| 3 | `scripts/03_finetune_whisper_lora.py` | LoRA fine-tune Whisper; log to W&B. |
| 4 | `scripts/04_evaluate.py` | Evaluate the fine-tuned model; produce the accuracy + fairness tables for Chapter 5. |
| 5 | `scripts/05_verify_corpus.py` | Audit the curated corpus (metadata only): clips/hours per source/accent, speaker-disjoint check. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in WANDB_API_KEY + HF_TOKEN
```

On Google Colab / RunPod, run the same `pip install` and set the env vars in the notebook/shell.

## Compute strategy (matches the thesis)

- **Unsloth 4-bit → a large model on free hardware.** The headline STT run fine-tunes
  `whisper-large-v3-turbo` (809M) with LoRA via [Unsloth](https://github.com/unslothai/unsloth)
  in 4-bit, which fits a **free Colab T4** — no A100 rental. Config:
  `configs/stt_whisper_large_v3_turbo_unsloth.yaml`; run step 3 with `--backend unsloth`.
  The Colab driver is `notebooks/02_stt_slice.ipynb`.
- **Two backends, one pipeline.** `--backend hf` (vanilla HF+PEFT, CPU-importable) or
  `--backend unsloth` (~2x faster). Everything downstream — data prep, collator, metrics,
  eval — is identical, so results are comparable.
- **PEFT by default.** LoRA trains ~1–5% of parameters, so even `large-v3` fits a single GPU.
- **Fallback to rent.** For full `whisper-large-v3` (1.55B) at scale, the same config swap +
  a rented A100 still works — but the turbo run above usually makes it unnecessary.

## Tracking

Experiment metrics (loss, WER/CER, hyperparameters) go to **Weights & Biases**.
Set `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` in `.env`.

## Data & licensing

- **AfriSpeech-200** — CC-BY-NC-SA-4.0 (**non-commercial**). Used here for **research only**,
  consistent with the thesis scope. Not for any commercial/deployable release.
- Every clip's source and license is recorded in the generated `manifest.csv`.

## Layout

```
naija-speech/
├── configs/                  # YAML configs for data + models
├── src/                      # library modules (added to sys.path by scripts)
│   ├── config.py             # tiny YAML/.env loader
│   ├── text_normalization.py # transcript normalization (preserves Nigerian lexis)
│   ├── metrics.py            # WER/CER, overall + stratified
│   ├── eda.py                # metadata-only EDA (distributions, report)
│   ├── corpus.py             # build the Nigerian corpus + manifest + splits
│   ├── whisper_lora.py       # Whisper + LoRA model/processor/collator helpers
│   └── tracking.py           # optional W&B logging
├── scripts/                  # thin CLI entry points (00_eda → 04_evaluate)
└── requirements.txt
```
