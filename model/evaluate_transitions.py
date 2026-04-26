#!/usr/bin/env python3
"""Evaluate detection quality for all language transitions using vocabulary lists.

Reports per directed pair (from->to):
- TP rate: target-language words typed on wrong source layout correctly detected as target.
- FP-to-target rate: source-language words typed correctly but incorrectly switched to target.

This is a lightweight offline harness to validate per-pair macro-parameters.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Languages
from Languages import (
    english_layout,
    russian_layout,
    hebrew_layout,
    convert_text_bidirectional,
    predict_language_with_confidence,
)

LANGS = ["en", "ru", "he"]
LAYOUTS = {
    "en": english_layout,
    "ru": russian_layout,
    "he": hebrew_layout,
}
VOCAB_FILES = {
    "en": os.path.join(SCRIPT_DIR, "vocabulary", "english_vocabulary"),
    "ru": os.path.join(SCRIPT_DIR, "vocabulary", "russian_vocabulary"),
    "he": os.path.join(SCRIPT_DIR, "vocabulary", "hebrew_vocabulary"),
}

# Must match cpp/src/Config.cpp
DEFAULT_PARAMS = (3, 10, 0.97, 0.55, 2, 0.85)
PAIR_OVERRIDES = {
    ("en", "ru"): (3, 10, 0.97, 0.55, 2, 0.85),
    ("ru", "en"): (3, 10, 0.97, 0.55, 2, 0.85),
    ("en", "he"): (4, 12, 0.99, 0.66, 2, 0.88),
    ("he", "en"): (4, 10, 0.98, 0.60, 2, 0.85),
    ("ru", "he"): (4, 12, 0.99, 0.66, 2, 0.88),
    ("he", "ru"): (4, 10, 0.98, 0.60, 2, 0.80),
}

LAYOUT_PAIRS = [
    (english_layout, russian_layout),
    (russian_layout, english_layout),
    (hebrew_layout, english_layout),
    (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),
    (hebrew_layout, russian_layout),
]


def load_vocab(lang: str, sample: int, seed: int, min_len: int, max_len: int) -> List[str]:
    words: List[str] = []
    with open(VOCAB_FILES[lang], encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.append(w)

    rng = random.Random(seed)
    return rng.sample(words, min(sample, len(words)))


def required_confidence(n: int, params: Tuple[int, int, float, float, int, float]) -> float:
    early_min, full_conf, conf_min, conf_max, _, _ = params
    if n < early_min:
        return 1.1
    if n >= full_conf:
        return conf_max
    t = (n - early_min) / (full_conf - early_min)
    return conf_min + t * (conf_max - conf_min)


def evaluate_pair(
    words_by_lang: Dict[str, List[str]],
    from_lang: str,
    to_lang: str,
    model_args: list,
) -> Tuple[int, int, int, int]:
    params = PAIR_OVERRIDES.get((from_lang, to_lang), DEFAULT_PARAMS)
    _, _, _, _, agreement_count, _ = params
    predict_cache: Dict[str, Tuple[str | None, float]] = {}

    def predict_best_lang(text: str, current_lang: str) -> str | None:
        last_lang = ""
        streak = 0

        for n in range(1, len(text) + 1):
            required = required_confidence(n, params)
            if required > 1.0:
                continue

            variants = []
            seen = set()
            for src_layout, dst_layout in LAYOUT_PAIRS:
                variant = convert_text_bidirectional(text[:n], src_layout, dst_layout)
                if variant not in seen:
                    seen.add(variant)
                    variants.append(variant)

            best_lang = None
            best_conf = 0.0
            for variant in variants:
                if variant in predict_cache:
                    lang, conf = predict_cache[variant]
                else:
                    result = predict_language_with_confidence(variant, *model_args)
                    if result is None:
                        lang, conf = None, 0.0
                    else:
                        lang, conf = result.language, result.confidence
                    predict_cache[variant] = (lang, conf)

                if conf > best_conf:
                    best_lang, best_conf = lang, conf

            if not best_lang:
                continue

            if best_lang == last_lang:
                streak += 1
            else:
                last_lang = best_lang
                streak = 1

            if (
                streak >= agreement_count
                and best_lang != current_lang
                and best_conf >= required
            ):
                return best_lang

        return None

    tp = 0
    for w in words_by_lang[to_lang]:
        mistyped = convert_text_bidirectional(w, LAYOUTS[to_lang], LAYOUTS[from_lang])
        if predict_best_lang(mistyped, from_lang) == to_lang:
            tp += 1

    fp = 0
    for w in words_by_lang[from_lang]:
        if predict_best_lang(w, from_lang) == to_lang:
            fp += 1

    return tp, len(words_by_lang[to_lang]), fp, len(words_by_lang[from_lang])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all directed language transitions")
    parser.add_argument("--sample", type=int, default=80, help="Words per language (default: 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum word length")
    parser.add_argument("--max-len", type=int, default=12, help="Maximum word length")
    args = parser.parse_args()

    print("Loading ONNX model...")
    model_args = Languages.load_model()

    words_by_lang = {
        lang: load_vocab(lang, args.sample, args.seed, args.min_len, args.max_len)
        for lang in LANGS
    }

    print("\nResults (all directed transitions):")
    for from_lang in LANGS:
        for to_lang in LANGS:
            if from_lang == to_lang:
                continue
            tp, tp_total, fp, fp_total = evaluate_pair(words_by_lang, from_lang, to_lang, model_args)
            print(
                f"{from_lang}->{to_lang}: "
                f"TP {tp}/{tp_total} ({tp/tp_total:.3f}), "
                f"FP-to-{to_lang} {fp}/{fp_total} ({fp/fp_total:.3f})"
            )


if __name__ == "__main__":
    main()
