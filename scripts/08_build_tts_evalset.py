"""Step 8 — Build the FIXED TTS evaluation set (Phase 1 of the TTS track).

200 general-domain sentences from the SCREENED test split (same duration window
as the STT benchmark), sampled once with a fixed seed and persisted forever —
every TTS model (zero-shot and fine-tuned) synthesizes exactly these sentences,
so scores stay comparable across the whole benchmark table.

Also picks 3 reference voices (one per thesis macro-accent) for the cloning
baselines (XTTS-v2 / Qwen3-TTS): the train-split speaker with the most clips per
accent, one 8-20 s reference clip each, extracted by downloading only the parquet
shard(s) that contain them (not the 70 GB corpus).

Usage:
    python scripts/08_build_tts_evalset.py                  # evalset + ref voices
    python scripts/08_build_tts_evalset.py --n 200 --seed 42
    python scripts/08_build_tts_evalset.py --skip-refs      # metadata only, no download
    python scripts/08_build_tts_evalset.py --self-test

Outputs (outputs/tts_eval/):
    evalset.csv        eval_id,text,accent,macro_accent,domain,duration,apath
    ref_voices.csv     speaker,macro_accent,gender,duration,ref_text,apath,shard,wav
    ref_voices/*.wav   the reference clips themselves
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import load_dotenv, load_yaml  # noqa: E402

OUT_DIR = "outputs/tts_eval"


def _audit():
    """The path-join speaker machinery lives in script 07 — import, don't copy."""
    p = os.path.join(os.path.dirname(__file__), "07_tts_data_audit.py")
    spec = importlib.util.spec_from_file_location("tts_audit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_repo(repo: str) -> str:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):  # HF repo id only — keeps the SQL literal clean
        raise SystemExit(f"invalid hf_curated_repo: {repo!r}")
    return repo


def _read_split(repo: str, split: str, with_filename: bool = False):
    import duckdb

    fn = ", filename=true" if with_filename else ""
    cols = "audio.path AS apath, text_raw, accent, domain, duration, gender"
    if with_filename:
        cols += ", filename"
    df = duckdb.connect().execute(
        f"""SELECT {cols}
            FROM read_parquet('hf://datasets/{repo}/data/{split}-*.parquet'{fn})
            WHERE source = 'afrispeech-200'"""
    ).df()
    if with_filename:
        df["shard"] = df.pop("filename").map(lambda f: os.path.basename(str(f)))
    return df


def build_evalset(df, cfg, n: int, seed: int, domain: str = "general"):
    """Screen (duration window + one domain), then a seeded sample of n."""
    from corpus import macro_accent

    lo = cfg.get("min_duration_sec", 0.5)
    hi = cfg.get("max_duration_sec", 30.0)
    mm = {k.lower(): v for k, v in cfg.get("macro_accent_map", {}).items()}
    df = df[(df["duration"] >= lo) & (df["duration"] <= hi)]
    df = df[df["domain"] == domain].copy()
    df["macro_accent"] = df["accent"].map(lambda a: macro_accent(a, mm))
    # sort before sampling so the seed is reproducible regardless of read order
    df = df.sort_values("apath").reset_index(drop=True)
    sample = df.sample(n=min(n, len(df)), random_state=seed).sort_values("apath")
    sample = sample.reset_index(drop=True)
    # "tts-NNN" is the locked general set's id space; other domains get their own
    prefix = "tts" if domain == "general" else domain[:3]
    sample.insert(0, "eval_id", [f"{prefix}-{i:03d}" for i in range(len(sample))])
    sample = sample.rename(columns={"text_raw": "text"})
    return sample[["eval_id", "text", "accent", "macro_accent", "domain",
                   "duration", "apath"]]


def cover_shards(shards_by_accent: dict[str, list[str]]) -> list[str]:
    """Greedy set cover: fewest shards such that every accent has a clip in one."""
    remaining = {a: set(s) for a, s in shards_by_accent.items()}
    for a, s in remaining.items():
        if not s:
            raise SystemExit(f"no candidate reference clips for accent {a!r}")
    picked: list[str] = []
    while remaining:
        cnt = Counter(s for shards in remaining.values() for s in shards)
        best = cnt.most_common(1)[0][0]
        picked.append(best)
        remaining = {a: s for a, s in remaining.items() if best not in s}
    return picked


def pick_ref_voices(cfg, dur_range=(8.0, 20.0), min_clips=50):
    """One reference clip per macro-accent, from that accent's biggest speaker."""
    from corpus import macro_accent
    from eda import load_ng_metadata

    audit = _audit()
    repo = _check_repo(cfg["hf_curated_repo"])
    print("[refs] reading train metadata (path column, no audio) ...")
    train = _read_split(repo, "train", with_filename=True)
    manifest = load_ng_metadata(cfg["hf_dataset_id"], cache_dir=".cache_eda",
                                country=cfg.get("country_filter", "NG"))
    annotated, _ = audit.match_speakers(train, manifest)
    mm = {k.lower(): v for k, v in cfg.get("macro_accent_map", {}).items()}
    annotated["macro_accent"] = annotated["accent"].map(lambda a: macro_accent(a, mm))

    chosen = []  # one row per accent: the biggest speaker's best-length clip candidates
    for acc in ("Yoruba", "Igbo", "Hausa"):
        sub = annotated[(annotated["macro_accent"] == acc)
                        & annotated["backfilled_speaker"].notna()]
        counts = sub.groupby("backfilled_speaker").size()
        for spk in counts[counts >= min_clips].sort_values(ascending=False).index:
            cand = sub[(sub["backfilled_speaker"] == spk)
                       & (sub["duration"] >= dur_range[0])
                       & (sub["duration"] <= dur_range[1])]
            if len(cand):
                chosen.append((acc, spk, cand))
                break
        else:
            raise SystemExit(f"no {acc} speaker with >={min_clips} clips has an "
                             f"{dur_range[0]}-{dur_range[1]} s clip")

    shard_sets = {acc: cand["shard"].unique().tolist() for acc, _, cand in chosen}
    shards = cover_shards(shard_sets)
    print(f"[refs] clips found in {len(shards)} shard(s): {shards}")

    rows = []
    for acc, spk, cand in chosen:
        pick = cand[cand["shard"].isin(shards)].sort_values("apath").iloc[0]
        rows.append({"speaker": spk, "macro_accent": acc, "gender": pick["gender"],
                     "duration": pick["duration"], "ref_text": pick["text_raw"],
                     "apath": pick["apath"], "shard": pick["shard"]})
    return rows, shards


def extract_ref_wavs(cfg, rows, shards):
    """Download only the needed shard(s) and write the reference WAVs."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pads
    from huggingface_hub import hf_hub_download

    repo = _check_repo(cfg["hf_curated_repo"])
    want = {r["apath"]: r for r in rows}
    wav_dir = os.path.join(OUT_DIR, "ref_voices")
    os.makedirs(wav_dir, exist_ok=True)
    for shard in shards:
        print(f"[refs] downloading shard {shard} ...")
        local = hf_hub_download(repo, f"data/{shard}", repo_type="dataset")
        tbl = pads.dataset(local).to_table(
            columns=["audio"],
            filter=pc.field("audio", "path").isin(list(want)))
        for cell in tbl.column("audio").to_pylist():
            r = want[cell["path"]]
            spk = re.sub(r"[^\w-]", "_", str(r["speaker"]))[:8]  # manifest is external data
            out = os.path.join(wav_dir, f"{r['macro_accent'].lower()}_{spk}.wav")
            with open(out, "wb") as fh:
                fh.write(cell["bytes"])  # stored native WAV bytes, written as-is
            r["wav"] = out
            print(f"[refs]   {r['macro_accent']}: {out} ({r['duration']:.1f}s)")
    missing = [r for r in rows if "wav" not in r]
    if missing:
        raise SystemExit(f"reference clips not found in shards: {missing}")
    return rows


def self_test():
    assert cover_shards({"a": ["s1", "s2"], "b": ["s2"], "c": ["s2", "s3"]}) == ["s2"]
    multi = cover_shards({"a": ["s1"], "b": ["s2"]})
    assert sorted(multi) == ["s1", "s2"]
    try:
        cover_shards({"a": []})
        raise AssertionError("empty candidate set must fail")
    except SystemExit:
        pass
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the fixed TTS eval set (Phase 1).")
    ap.add_argument("--config", default="configs/data_afrispeech_ng.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain", default="general",
                    help="Corpus domain to sample from (e.g. clinical for the supplement).")
    ap.add_argument("--out", default="evalset.csv",
                    help="Output CSV name inside outputs/tts_eval/.")
    ap.add_argument("--skip-refs", action="store_true",
                    help="Only build the evalset CSV (no shard download).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.domain != "general":
        args.skip_refs = True  # ref voices are domain-independent and already built

    if args.self_test:
        self_test()
        return

    import pandas as pd

    load_dotenv()
    cfg = load_yaml(args.config)
    repo = _check_repo(cfg["hf_curated_repo"])
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[evalset] reading test metadata (no audio) ...")
    test = _read_split(repo, "test")
    sample = build_evalset(test, cfg, args.n, args.seed, args.domain)
    path = os.path.join(OUT_DIR, args.out)
    sample.to_csv(path, index=False)
    print(f"[evalset] {len(sample)} {args.domain} sentences -> {path}")
    print(sample["macro_accent"].value_counts().to_string())
    print(f"[evalset] text length: median "
          f"{int(sample['text'].str.len().median())} chars, "
          f"max {int(sample['text'].str.len().max())}")

    if not args.skip_refs:
        rows, shards = pick_ref_voices(cfg)
        rows = extract_ref_wavs(cfg, rows, shards)
        ref_path = os.path.join(OUT_DIR, "ref_voices.csv")
        pd.DataFrame(rows)[["speaker", "macro_accent", "gender", "duration",
                            "ref_text", "apath", "shard", "wav"]].to_csv(
            ref_path, index=False)
        print(f"[refs] -> {ref_path}")


if __name__ == "__main__":
    main()
