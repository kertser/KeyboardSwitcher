#!/usr/bin/env python3
"""Build compact lexical-triage artifacts from the raw vocabulary lists.

Outputs (in model/triage_data/):
  lexicon_{lang}.txt    — filtered, normalized, deduped, sorted word list
                          (lowercase; Hebrew finals normalized; strictly
                          language-alphabet characters only)
  trigrams_{lang}.json  — character-trigram log10-prob table with boundary
                          markers, add-k smoothing floor, and calibration
                          stats (mean/std of per-word avg logprob) used to
                          map raw scores onto a [0,1] plausibility scale.
  meta.json             — build parameters + artifact stats.

These artifacts are consumed by model/triage.py (Python prototype) and are
the specification for the future C++ port (cpp/src/Triage.*).

Usage:  python build_triage_artifacts.py
"""
from __future__ import annotations

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "triage_data")

VOCAB_FILES = {
    "en": os.path.join(SCRIPT_DIR, "vocabulary", "english_vocabulary"),
    "ru": os.path.join(SCRIPT_DIR, "vocabulary", "russian_vocabulary"),
    "he": os.path.join(SCRIPT_DIR, "vocabulary", "hebrew_vocabulary"),
}

# Strict per-language alphabets (post-normalization).  Any word containing a
# character outside its language alphabet is dropped — the triage signal must
# be single-script by construction (mirrors the disjoint en/ru/he scripts).
ALPHABETS = {
    "en": set("abcdefghijklmnopqrstuvwxyz"),
    "ru": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "he": set(chr(c) for c in range(0x05D0, 0x05EB)),
}

MIN_LEN = 2
MAX_LEN = 24
SMOOTH_K = 0.5          # add-k smoothing for trigram probabilities
BOUNDARY_START = "^"    # word-boundary markers for trigram context
BOUNDARY_END = "$"

# Hebrew final forms → regular forms (mirrors cpp NormalizeHebrewFinals)
HEBREW_FINALS = str.maketrans({
    "\u05DA": "\u05DB",  # ך → כ
    "\u05DD": "\u05DE",  # ם → מ
    "\u05DF": "\u05E0",  # ן → נ
    "\u05E3": "\u05E4",  # ף → פ
    "\u05E5": "\u05E6",  # ץ → צ
})


def normalize_word(word: str, lang: str) -> str:
    w = word.strip().lower()
    if lang == "he":
        w = w.translate(HEBREW_FINALS)
    return w


def load_lexicon(lang: str) -> list[str]:
    alphabet = ALPHABETS[lang]
    seen: set[str] = set()
    with open(VOCAB_FILES[lang], encoding="utf-8") as f:
        for line in f:
            w = normalize_word(line, lang)
            if not (MIN_LEN <= len(w) <= MAX_LEN):
                continue
            if not w.isalpha():
                continue
            if any(c not in alphabet for c in w):
                continue
            seen.add(w)
    return sorted(seen)


def build_trigram_table(words: list[str], lang: str) -> dict:
    """Count boundary-marked trigrams and convert to log10 probabilities."""
    counts: dict[str, int] = {}
    total = 0
    for w in words:
        padded = BOUNDARY_START + w + BOUNDARY_END
        for i in range(len(padded) - 2):
            g = padded[i:i + 3]
            counts[g] = counts.get(g, 0) + 1
            total += 1

    # Possible-trigram universe for smoothing: (alphabet + 2 boundary chars)^3
    v = (len(ALPHABETS[lang]) + 2) ** 3
    denom = total + SMOOTH_K * v
    logprobs = {g: round(math.log10((c + SMOOTH_K) / denom), 4)
                for g, c in counts.items()}
    floor = round(math.log10(SMOOTH_K / denom), 4)

    # Calibration: per-word average trigram logprob over the whole lexicon.
    # Used by triage.py to map a raw avg-logprob onto [0,1] plausibility.
    n = 0
    mean = 0.0
    m2 = 0.0
    for w in words:
        padded = BOUNDARY_START + w + BOUNDARY_END
        ngrams = len(padded) - 2
        s = sum(logprobs.get(padded[i:i + 3], floor) for i in range(ngrams))
        avg = s / ngrams
        n += 1
        delta = avg - mean
        mean += delta / n
        m2 += delta * (avg - mean)
    std = math.sqrt(m2 / n) if n > 1 else 1.0

    return {
        "lang": lang,
        "smoothing_k": SMOOTH_K,
        "floor_logprob": floor,
        "calib_mean": round(mean, 4),
        "calib_std": round(std, 4),
        "trigrams": logprobs,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "smoothing_k": SMOOTH_K,
        "languages": {},
    }
    for lang in ("en", "ru", "he"):
        print(f"[{lang}] loading + filtering vocabulary ...", flush=True)
        words = load_lexicon(lang)
        lex_path = os.path.join(OUT_DIR, f"lexicon_{lang}.txt")
        with open(lex_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(words))
        print(f"[{lang}] {len(words)} words -> {os.path.basename(lex_path)}")

        print(f"[{lang}] building trigram table ...", flush=True)
        table = build_trigram_table(words, lang)
        tri_path = os.path.join(OUT_DIR, f"trigrams_{lang}.json")
        with open(tri_path, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False)
        print(f"[{lang}] {len(table['trigrams'])} trigrams, "
              f"calib mean={table['calib_mean']} std={table['calib_std']}")

        meta["languages"][lang] = {
            "words": len(words),
            "trigrams": len(table["trigrams"]),
            "calib_mean": table["calib_mean"],
            "calib_std": table["calib_std"],
        }

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done. Artifacts written to", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
