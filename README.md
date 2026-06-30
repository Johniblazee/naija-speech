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

---

## Why "vertical slice" first

Instead of building all 3 STT + 3 TTS models at once, we prove the **entire pipeline** on one
model first: `corpus → zero-shot baseline → LoRA fine-tune → accent-stratified evaluation`.
Once that works, adding more architectures is repetition, not risk. This is the de-risking
strategy described in the thesis methodology.

## Pipeline (run in order)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/01_build_corpus.py` | Download Nigerian AfriSpeech-200 accents, normalize text, build a manifest + speaker-disjoint splits, save to disk. |
| 2 | `scripts/02_zeroshot_baseline.py` | Measure the **gap**: zero-shot Whisper WER/CER on the Nigerian test set, stratified by accent and domain. |
| 3 | `scripts/03_finetune_whisper_lora.py` | LoRA fine-tune Whisper; log to Comet. |
| 4 | `scripts/04_evaluate.py` | Evaluate the fine-tuned model; produce the accuracy + fairness tables for Chapter 5. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in COMET_API_KEY + HF_TOKEN
```

On Google Colab / RunPod, run the same `pip install` and set the env vars in the notebook/shell.

## Compute strategy (matches the thesis)

- **Develop small, then scale.** Start with `whisper-small` on a free Colab T4 to validate the
  whole pipeline cheaply, then rerun with `whisper-large-v3` on a rented A100. Switch by
  editing one line in `configs/stt_whisper_small_lora.yaml` (`model_id`).
- **PEFT by default.** LoRA trains ~1–5% of parameters, so even `large-v3` fits a single GPU.

## Tracking

Experiment metrics (loss, WER/CER, hyperparameters) go to **Comet** (Experiment Management).
Set `COMET_API_KEY`, `COMET_WORKSPACE`, and `COMET_PROJECT_NAME` in `.env`.

## Data & licensing

- **AfriSpeech-200** — CC-BY-NC-SA-4.0 (**non-commercial**). Used here for **research only**,
  consistent with the thesis scope. Not for any commercial/deployable release.
- Every clip's source and license is recorded in the generated `manifest.csv`.

## Layout

```
naija-speech/
├── configs/                  # YAML configs for data + models
├── src/naija_speech/         # importable library code
│   ├── config.py             # tiny YAML config loader
│   ├── text_normalization.py # transcript normalization (preserves Nigerian lexis)
│   ├── metrics.py            # WER/CER, overall + stratified
│   ├── corpus.py             # build the Nigerian corpus + manifest + splits
│   └── whisper_lora.py       # Whisper + LoRA model/processor/collator helpers
├── scripts/                  # thin CLI entry points (run in order)
└── requirements.txt
```
