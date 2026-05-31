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

# Must match cpp/src/Config.cpp  — keep in sync!
# Format (11): (EarlyDetectionMinChars, FullConfidenceChars,
#               ConfidenceAtMinChars, ConfidenceAtMaxChars,
#               ConsecutiveAgreementCount, BorderlineZoneFactor,
#               MinTop1Top2Margin, ShortInputExtraConf, PhraseConfScale,
#               HebrewScriptVirtualConf, HebrewScriptCoverageThreshold)
DEFAULT_PARAMS = (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90)
PAIR_OVERRIDES = {
    # Margin gate: 0.05 on robust pairs (cheap FP insurance), 0.10 on →he.
    ("en", "ru"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    ("ru", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    # en→he / ru→he: EarlyMin=3; signal-quality gates ported from 1.3.0
    #   margin=0.10, phraseScale=0.80, hebrewScriptVirtualConf=0.78
    ("en", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90),
    ("he", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    ("ru", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90),
    ("he", "ru"): (4, 15, 0.99, 0.70, 2, 0.80, 0.05, 0.02, 1.00, 0.00, 0.90),
}

LAYOUT_PAIRS = [
    (english_layout, russian_layout),
    (russian_layout, english_layout),
    (hebrew_layout, english_layout),
    (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),
    (hebrew_layout, russian_layout),
]

_CLASS_LANG = {1: "en", 2: "he", 3: "ru"}


def hebrew_script_coverage(text: str) -> float:
    """Fraction of alpha chars in the Hebrew Unicode block (U+05D0–U+05EA)."""
    alpha = hebrew = 0
    for c in text:
        if c == " ":
            continue
        if c.isalpha():
            alpha += 1
            if "\u05d0" <= c <= "\u05ea":
                hebrew += 1
    return (hebrew / alpha) if alpha else 0.0


def load_vocab(lang: str, sample: int, seed: int, min_len: int, max_len: int) -> List[str]:
    words: List[str] = []
    with open(VOCAB_FILES[lang], encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.append(w)

    rng = random.Random(seed)
    return rng.sample(words, min(sample, len(words)))


def required_confidence(n: int, params, is_phrase: bool = False) -> float:
    early_min, full_conf, conf_min, conf_max = params[0], params[1], params[2], params[3]
    phrase_scale = params[8]
    if n < early_min:
        return 1.1
    if n >= full_conf:
        req = conf_max
    else:
        t = (n - early_min) / (full_conf - early_min)
        req = conf_min + t * (conf_max - conf_min)
    if is_phrase and 0.0 < phrase_scale < 1.0:
        req *= phrase_scale
    return req


def evaluate_pair(
    words_by_lang: Dict[str, List[str]],
    from_lang: str,
    to_lang: str,
    model_args: list,
) -> Tuple[int, int, int, int]:
    params = PAIR_OVERRIDES.get((from_lang, to_lang), DEFAULT_PARAMS)
    agreement_count = params[4]
    early_min       = params[0]
    margin_min      = params[6]
    short_extra     = params[7]
    he_params       = PAIR_OVERRIDES.get((from_lang, "he"), DEFAULT_PARAMS)
    he_virtual_conf = he_params[9]
    he_cov_thresh   = he_params[10]
    # Cache: variant -> (lang, conf, scores)
    predict_cache: Dict[str, Tuple[str | None, float, list]] = {}

    def predict_best_lang(text: str, current_lang: str) -> str | None:
        last_lang = ""
        streak = 0

        for n in range(1, len(text) + 1):
            # Source-restricted variants (mirror cpp/src/main.cpp):
            # identity (text as typed) plus current_lang -> each other layout.
            variants = []
            seen = set()
            cur_layout = LAYOUTS[current_lang]
            variants.append(text[:n])
            seen.add(text[:n])
            for other_lang, dst_layout in LAYOUTS.items():
                if other_lang == current_lang:
                    continue
                variant = convert_text_bidirectional(text[:n], cur_layout, dst_layout)
                if variant not in seen:
                    seen.add(variant)
                    variants.append(variant)

            is_phrase = any(" " in v for v in variants)

            best_lang = None
            best_conf = 0.0
            best_scores: list = [0.0, 0.0, 0.0, 0.0]
            he_script_vc = 0.0
            for variant in variants:
                if variant in predict_cache:
                    lang, conf, scores = predict_cache[variant]
                else:
                    result = predict_language_with_confidence(variant, *model_args)
                    if result is None:
                        lang, conf, scores = None, 0.0, [0.0, 0.0, 0.0, 0.0]
                    else:
                        lang, conf, scores = result.language, result.confidence, result.scores
                    predict_cache[variant] = (lang, conf, scores)

                # Hebrew script coverage gate (→he pairs only)
                if (he_virtual_conf > 0.0 and current_lang != "he"):
                    cov = hebrew_script_coverage(variant)
                    if cov >= he_cov_thresh:
                        onnx_contradicts = (lang is not None and lang != "he" and conf > 0.80)
                        if not onnx_contradicts:
                            vc = cov * he_virtual_conf
                            if vc > he_script_vc:
                                he_script_vc = vc

                if conf > best_conf:
                    best_lang, best_conf, best_scores = lang, conf, scores

            # Apply Hebrew script gate if it beats ONNX
            script_fired = False
            if he_script_vc > best_conf:
                best_lang, best_conf = "he", he_script_vc
                best_scores = [0.0, 0.0, 0.0, 0.0]
                script_fired = True

            if not best_lang:
                continue

            # Runner-up / margin from softmax scores
            runner_up = 0.0
            for i in (1, 2, 3):
                if _CLASS_LANG[i] == best_lang:
                    continue
                if best_scores[i] > runner_up:
                    runner_up = best_scores[i]
            margin = best_conf - runner_up

            required = required_confidence(n, params, is_phrase)
            if required > 1.0:
                continue

            # Short-input extra confidence (FP guard)
            if (n <= early_min + 2) and (not is_phrase) and short_extra > 0.0:
                required = min(0.9999, required + short_extra)

            if best_lang == last_lang:
                streak += 1
            else:
                last_lang = best_lang
                streak = 1

            # Margin gate (skip when script gate fired — no real scores)
            if (not script_fired) and margin_min > 0.0 and margin < margin_min:
                continue

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
