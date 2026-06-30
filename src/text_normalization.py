"""Transcript normalization for Nigerian-accented English.

Design choices (see thesis §3.2 functional requirement 3):
- We normalize for FAIR scoring (casing, punctuation, whitespace) but we do NOT
  "correct" spelling toward US/UK English — legitimate Nigerian English lexical
  items (e.g. 'japa', 'naija') are preserved.
- We keep BOTH the raw transcript and a normalized transcript so we can report
  orthographic vs normalized WER, as shown on the deck's evaluation slide.

This module is pure-Python (no torch/datasets) so it can be unit-tested cheaply.
"""
from __future__ import annotations

import re
import unicodedata

# Punctuation we strip for normalized scoring. We deliberately KEEP the
# apostrophe so contractions ("don't") and elisions stay intact.
_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Apostrophes that appear at word edges (quotes) rather than inside contractions.
_EDGE_APOS_RE = re.compile(r"(?<!\w)'|'(?!\w)")


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize a transcript for fair WER/CER scoring.

    Steps: Unicode NFKC, optional lowercase, strip punctuation (keep in-word
    apostrophes), collapse whitespace. Spelling is left untouched on purpose.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if lowercase:
        text = text.lower()
    text = _PUNCT_RE.sub(" ", text)        # drop punctuation -> space
    text = _EDGE_APOS_RE.sub(" ", text)    # remove stray edge apostrophes/quotes
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_scorable(reference: str) -> bool:
    """True if a reference is non-empty after normalization (jiwer needs this)."""
    return bool(normalize_text(reference))
