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
# Format (17): (EarlyDetectionMinChars, FullConfidenceChars,
#               ConfidenceAtMinChars, ConfidenceAtMaxChars,
#               ConsecutiveAgreementCount, BorderlineZoneFactor,
#               MinTop1Top2Margin, ShortInputExtraConf, PhraseConfScale,
#               HebrewScriptVirtualConf, HebrewScriptCoverageThreshold,
#               SwitchBiasMargin, PersistentMinAvgConf, PersistentMinSteps,
#               WeakScoreClassIdx, WeakScoreMinAvg, WeakScoreWindow)
DEFAULT_PARAMS = (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7)
PAIR_OVERRIDES = {
    # Margin gate: 0.05 on robust pairs (cheap FP insurance), 0.10 on →he.
    # SwitchBiasMargin (idx 11): incumbent guard, 0.02 on →he only (after the
    #   v2.x retrain the incumbent EN signal rose, so 0.04 blocked ~4 genuine
    #   Hebrew phrases at zero single-word FP benefit), 0.0 on robust pairs.
    #   Persistent/weak gates (PMS=6, WSMA=0.40) for Hebrew flat-signal recovery
    #   with no FP cost.
    ("en", "ru"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("ru", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("en", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90, 0.02, 0.55, 6, 2, 0.40, 7),
    ("he", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("ru", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90, 0.02, 0.55, 6, 2, 0.40, 7),
    ("he", "ru"): (4, 15, 0.99, 0.70, 2, 0.80, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
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

# Global detection early-out: the C++ hook only calls detection once the cache
# reaches the smallest EarlyDetectionMinChars across all pairs.
GLOBAL_MIN = min(p[0] for p in PAIR_OVERRIDES.values())


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
    he_params       = PAIR_OVERRIDES.get((from_lang, "he"), DEFAULT_PARAMS)
    he_virtual_conf = he_params[9]
    he_cov_thresh   = he_params[10]
    # Cache: variant -> (lang, conf, scores)
    predict_cache: Dict[str, Tuple[str | None, float, list]] = {}

    def _is_phrase(variants) -> bool:
        # Any space → ≥2 words.  (Ablation showed a stricter rule costs
        # Hebrew-phrase recall for no measurable FP gain; see exp_phrase.py.)
        return any(" " in v for v in variants)

    def predict_best_lang(text: str, current_lang: str) -> str | None:
        last_lang = ""
        streak = 0
        frames: list = []  # (lang, conf, scores), newest at end, capped at 10
        cur_idx = next((i for i in (1, 2, 3) if _CLASS_LANG[i] == current_lang), 0)

        def update(lang, conf, scores):
            nonlocal last_lang, streak
            if lang == last_lang:
                streak += 1
            else:
                last_lang, streak = lang, 1
            frames.append((lang, conf, list(scores)))
            if len(frames) > 10:
                frames.pop(0)

        def is_persistent(lang, min_steps, min_avg) -> bool:
            if min_steps <= 0 or len(frames) < min_steps:
                return False
            s = 0.0
            for f in frames[-min_steps:]:
                if f[0] != lang:
                    return False
                s += f[1]
            return (s / min_steps) >= min_avg

        def weak_score_avg(cls_idx, window) -> float:
            if window <= 0 or len(frames) < window:
                return 0.0
            return sum(f[2][cls_idx] for f in frames[-window:]) / window

        for n in range(1, len(text) + 1):
            # Global early-out: detection only runs once enough chars are typed,
            # matching the C++ main.cpp fast-path (min EarlyDetectionMinChars
            # across all pairs).  Below this the hook never calls detection, so
            # the harness must not process or pre-seed history at those positions.
            if n < GLOBAL_MIN:
                continue
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

            is_phrase = _is_phrase(variants)

            best_lang = None
            best_conf = 0.0
            best_scores: list = [0.0, 0.0, 0.0, 0.0]
            incumbent_conf = 0.0
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

                if lang is not None:
                    # Incumbent (stay-on-current-language) strength, top-2 aware.
                    if cur_idx and scores[cur_idx] > incumbent_conf:
                        incumbent_conf = scores[cur_idx]
                    if conf > best_conf:
                        best_lang, best_conf, best_scores = lang, conf, scores

            # Apply Hebrew script gate if it beats ONNX
            script_fired = False
            if he_script_vc > best_conf:
                best_lang, best_conf = "he", he_script_vc
                best_scores = [0.0, 0.0, 0.0, 0.0]
                script_fired = True

            if not best_lang:
                update("", 0.0, [0.0, 0.0, 0.0, 0.0])
                continue

            det_params = PAIR_OVERRIDES.get((current_lang, best_lang), DEFAULT_PARAMS)
            early_min   = det_params[0]
            agreement   = det_params[4]
            margin_min  = det_params[6]
            short_extra = det_params[7]
            switch_bias = det_params[11]
            p_min_avg   = det_params[12]
            p_min_steps = det_params[13]
            ws_idx      = det_params[14]
            ws_min      = det_params[15]
            ws_win      = det_params[16]

            # Runner-up / margin from softmax scores
            runner_up = 0.0
            for i in (1, 2, 3):
                if _CLASS_LANG[i] == best_lang:
                    continue
                if best_scores[i] > runner_up:
                    runner_up = best_scores[i]
            margin = best_conf - runner_up

            required = required_confidence(n, det_params, is_phrase)
            no_scores = all(s == 0.0 for s in best_scores)
            if required > 1.0:
                # Below the pair's EarlyDetectionMinChars: pre-seed the
                # agreement streak (mirrors C++), then wait for more chars.
                update(best_lang, best_conf, [0.0, 0.0, 0.0, 0.0] if no_scores else best_scores)
                continue

            # Short-input extra confidence (FP guard)
            if (n <= early_min + 2) and (not is_phrase) and short_extra > 0.0:
                required = min(0.9999, required + short_extra)

            update(best_lang, best_conf, [0.0, 0.0, 0.0, 0.0] if no_scores else best_scores)

            # Incumbent-advantage gate (FP guard)
            if (best_lang != current_lang and switch_bias > 0.0
                    and best_conf < incumbent_conf + switch_bias):
                continue

            # Margin gate (skip when script gate fired — no real scores)
            if (not script_fired) and (not no_scores) and margin_min > 0.0 and margin < margin_min:
                continue

            # Firing decision
            fire = False
            if (last_lang == best_lang and streak >= agreement) and best_conf >= required:
                fire = True
            if (not fire) and best_lang == "he":
                if is_persistent("he", p_min_steps, p_min_avg):
                    fire = True
                elif weak_score_avg(ws_idx, ws_win) >= ws_min:
                    fire = True

            if fire and best_lang != current_lang:
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
