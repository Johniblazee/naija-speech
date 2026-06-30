"""EDA for the Nigerian subset of AfriSpeech-200 — metadata only, no audio.

Reads the dataset's transcript CSVs (a few MB total), filters country == NG, and
reports distribution stats across ALL Nigerian accents (incl. Ibibio/Efik) plus
the macro-accent grouping (Yoruba/Igbo/Hausa/Other) used for modeling.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from corpus import macro_accent

_RAW = "https://huggingface.co/datasets/{ds}/raw/main/transcripts/{split}.csv"
_SPLITS = ("train", "dev", "test")
_GROUP_COLS = ("macro_accent", "accent", "domain", "gender", "age_group", "split")


def _download(ds: str, split: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{split}.csv"
    if not dest.exists():
        urllib.request.urlretrieve(_RAW.format(ds=ds, split=split), dest)
    return dest


def load_ng_metadata(ds: str, cache_dir: str | Path, country: str = "NG",
                     macro_map: dict | None = None) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    frames = []
    for split in _SPLITS:
        df = pd.read_csv(_download(ds, split, cache_dir))
        df["split"] = "validation" if split == "dev" else split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["country"] == country].copy()
    df["accent"] = df["accent"].fillna("").str.lower()
    mm = {k.lower(): v for k, v in (macro_map or {}).items()}
    df["macro_accent"] = df["accent"].map(lambda a: macro_accent(a, mm))
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return df


def _hours(s: pd.Series) -> float:
    return round(s.sum() / 3600, 1)


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for col in _GROUP_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(col).agg(
            clips=("transcript", "size"),
            hours=("duration", _hours),
            speakers=("user_ids", "nunique"),
        ).sort_values("clips", ascending=False)
        tables[col] = g
    return tables


def make_charts(df: pd.DataFrame, fig_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for col in ("macro_accent", "accent", "domain", "gender", "split"):
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        ax = counts.plot(kind="bar", title=f"Clips by {col}", figsize=(8, 4))
        ax.set_ylabel("clips")
        ax.figure.tight_layout()
        p = fig_dir / f"clips_by_{col}.png"
        ax.figure.savefig(p)
        ax.figure.clf()
        paths.append(p)
    # duration distribution
    ax = df["duration"].dropna().plot(kind="hist", bins=50, title="Clip duration (s)", figsize=(8, 4))
    ax.set_xlabel("seconds")
    ax.figure.tight_layout()
    p = fig_dir / "duration_hist.png"
    ax.figure.savefig(p)
    ax.figure.clf()
    paths.append(p)
    return paths


def write_report(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = summary_tables(df)
    figs = make_charts(df, out_dir / "figures")

    def md_table(t: pd.DataFrame) -> str:
        try:
            return t.to_markdown()
        except Exception:  # tabulate missing
            return "```\n" + t.to_string() + "\n```"

    lines = [
        "# AfriSpeech-200 — Nigerian subset EDA",
        "",
        f"- Total Nigerian clips: **{len(df):,}**",
        f"- Total hours: **{_hours(df['duration'])}**",
        f"- Unique speakers: **{df['user_ids'].nunique():,}**",
        f"- Distinct accents: **{df['accent'].nunique()}**",
        "",
    ]
    for col, t in tables.items():
        lines += [f"## By {col}", "", md_table(t), ""]
    lines += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]

    report = out_dir / "eda_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def ng_accents(df: pd.DataFrame) -> list[str]:
    return sorted(a for a in df["accent"].unique() if a)
