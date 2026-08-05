"""Step 14 — Build the blinded MOS / accent-match listening pack (TTS Phase 4).

Assembles the human-panel materials per thesis §3.7:
  - 6 fixed general-domain sentences (seeded), each synthesized by ALL five
    systems in the same voice -> raters compare like with like, blind.
  - The 3 genuine human reference recordings hidden among the clips as anchors
    (same speakers the systems clone/imitate).
  - Clips peak-normalized (-3 dBFS) so loudness doesn't bias ratings, trimmed
    to <=15 s with a fade (a runaway is heard as repetition, not endured),
    resampled to 24 kHz mono WAV.
  - Anonymized filenames (crc-coded); the un-blinding KEY is written separately
    and must NOT be sent to raters.
  - rating_sheet.csv (one row per clip, randomized order) + instructions.md
    (P.800-style ACR naturalness + accent-match scales).

Runs on the laptop; pulls the ~33 needed clips from the HF results repo.

Usage:
    python scripts/14_build_mos_pack.py                # outputs/mos_pack/
    python scripts/14_build_mos_pack.py --sentences 6 --seed 42
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv  # noqa: E402

REPO = "Johniblazee/naija-speech-afrispeech-ng"
BASE = "experiments/tts/phase1-zeroshot"
SYSTEMS = ("styletts2-ft", "orpheus-ft", "xtts", "qwen3tts", "yarngpt")
VOICES = ("yoruba", "igbo", "hausa")
# YarnGPT cannot clone; its preset voices stand in per accent (documented in §3.7)
YARN_VOICE = {"yoruba": "idera", "igbo": "tayo", "hausa": "zainab"}
OUT = "outputs/mos_pack"
SR = 24000
MAX_S, FADE_S, PEAK = 15.0, 0.4, 0.707  # -3 dBFS


def fetch(path):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, path, repo_type="dataset")


def process(src, dst):
    import librosa
    import numpy as np
    import soundfile as sf

    x, _ = librosa.load(src, sr=SR, mono=True)
    x = x[: int(MAX_S * SR)]
    n = int(FADE_S * SR)
    if x.size > n:
        x[-n:] *= np.linspace(1, 0, n)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (PEAK / peak)
    sf.write(dst, x, SR, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the blinded MOS listening pack.")
    ap.add_argument("--sentences", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd

    load_dotenv()
    rnd = random.Random(args.seed)
    clips_dir = os.path.join(OUT, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    ev = pd.read_csv(fetch(f"{BASE.rsplit('/',1)[0]}/evalset/evalset.csv"))
    pool = ev[(ev["duration"] >= 6) & (ev["duration"] <= 12)]
    sents = pool.sample(n=args.sentences, random_state=args.seed).reset_index(drop=True)
    sents["voice"] = [VOICES[i % 3] for i in range(len(sents))]  # round-robin voices

    items = []
    for _, s in sents.iterrows():
        for system in SYSTEMS:
            v = YARN_VOICE[s["voice"]] if system == "yarngpt" else s["voice"]
            items.append({"system": system, "eval_id": s["eval_id"], "voice": s["voice"],
                          "text": s["text"],
                          "hf": f"{BASE}/{system}/wav/{s['eval_id']}__{v}.wav"})
    refs = pd.read_csv(fetch(f"{BASE.rsplit('/',1)[0]}/evalset/ref_voices.csv"))
    for _, r in refs.iterrows():
        wav = str(r["wav"]).replace("\\", "/").split("/")[-1]
        items.append({"system": "human", "eval_id": f"ref-{r['macro_accent'].lower()}",
                      "voice": r["macro_accent"].lower(), "text": r["ref_text"],
                      "hf": f"{BASE.rsplit('/',1)[0]}/evalset/ref_voices/{wav}"})

    rows = []
    for it in items:
        code = f"{zlib.crc32((it['system'] + it['eval_id'] + str(args.seed)).encode()) & 0xffffff:06x}"
        fname = f"clip_{code}.wav"
        try:
            process(fetch(it["hf"]), os.path.join(clips_dir, fname))
        except Exception as e:
            print(f"SKIP {it['system']}/{it['eval_id']}: {e}")
            continue
        rows.append({**it, "file": fname})
        print(f"  {fname}  <- {it['system']:12} {it['eval_id']} ({it['voice']})")

    rnd.shuffle(rows)
    key = pd.DataFrame(rows)[["file", "system", "eval_id", "voice", "text"]]
    key.to_csv(os.path.join(OUT, "KEY_do_not_share.csv"), index=False)

    sheet = pd.DataFrame({
        "order": range(1, len(rows) + 1),
        "clip": [r["file"] for r in rows],
        "naturalness_1to5": "", "accent_match_1to5": "", "comments": ""})
    sheet.to_csv(os.path.join(OUT, "rating_sheet.csv"), index=False)

    with open(os.path.join(OUT, "instructions.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# Listening Study — Instructions (≈15 minutes)

Thank you for taking part. You will hear {len(rows)} short clips of English
speech. Some are real recordings; most are computer-generated. Please:

1. Use **headphones** in a quiet room.
2. Play the clips **in the order given** in `rating_sheet.csv`.
3. Listen to each clip **once, fully**, then rate it immediately — first
   impressions are exactly what we need. Some clips repeat themselves or sound
   strange; that is expected — rate them as you hear them.

For each clip give two ratings (1–5):

**Naturalness** — how natural does the speech sound overall?
5 Excellent — completely natural, like a real person ·
4 Good — natural with minor artificial moments ·
3 Fair — noticeably synthetic but easy to listen to ·
2 Poor — clearly artificial, effortful to listen to ·
1 Bad — very unnatural or distorted.

**Accent match** — how authentically *Nigerian* does the accent sound?
5 Unmistakably Nigerian — could be a person from your community ·
4 Mostly Nigerian with occasional foreign-sounding words ·
3 Mixed / hard to place ·
2 Mostly foreign (e.g. American/British) with traces of Nigerian ·
1 Clearly non-Nigerian.

The two ratings are independent: speech can sound natural but not Nigerian,
or Nigerian but robotic. Optional comments are welcome. There are no right or
wrong answers — your honest impression is the data.
""")
    print(f"\n[pack] {len(rows)} clips -> {OUT}/  (send clips/ + rating_sheet.csv "
          f"+ instructions.md; KEEP KEY_do_not_share.csv private)")


if __name__ == "__main__":
    main()
