#!/usr/bin/env python3
"""
Synthetic tuning test for the adaptive-confidence-curve parameters.

Grid-searches over:
    EarlyDetectionMinChars   – earliest char count where detection fires
    FullConfidenceChars      – char count where the confidence floor kicks in
    ConfidenceAtMinChars     – required confidence at EarlyDetectionMinChars
    ConfidenceAtMaxChars     – confidence floor (at FullConfidenceChars+)

Test cases:
    TRUE POSITIVES  – a word from language A is typed on a different layout B,
                      producing garbled text.  The detector should fire and
                      identify language A.
    TRUE NEGATIVES  – a word from language A is typed on the correct layout A.
                      The detector should NOT fire (no switch needed).

Metrics per grid point:
    • TP rate          (correctly detected wrong layout)
    • FP rate          (incorrectly fired on correct layout)
    • Avg chars-to-detect (speed – lower is better, among TPs only)
    • Composite score  (TP_rate – 2·FP_rate – 0.02·avg_chars)

Usage (from the python/ folder, with the .venv activated):
    python tune_confidence.py                         # defaults
    python tune_confidence.py --sample 300            # more words per language
    python tune_confidence.py --sample 100 --phrases  # include 2-word phrases
    python tune_confidence.py --csv results.csv       # also dump a CSV
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Make sure the script can import project modules from the same directory
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------
VOCAB_DIR = os.path.join(SCRIPT_DIR, "vocabulary")

VOCAB_FILES = {
    "en": os.path.join(VOCAB_DIR, "english_vocabulary"),
    "ru": os.path.join(VOCAB_DIR, "russian_vocabulary"),
    "he": os.path.join(VOCAB_DIR, "hebrew_vocabulary"),
}

LAYOUT_MAP = {
    "en": english_layout,
    "ru": russian_layout,
    "he": hebrew_layout,
}


def load_vocabulary(lang: str, min_len: int = 3, max_len: int = 20) -> List[str]:
    """Load words from a vocabulary file, filtering by length."""
    path = VOCAB_FILES[lang]
    words: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.append(w)
    return words


# ---------------------------------------------------------------------------
# Test-case definitions
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    """One evaluation scenario."""
    text: str              # what the user *actually types* (on current layout)
    current_layout: str    # the keyboard layout that is active ("en", "ru", "he")
    expected_lang: str     # language the detector should identify (or "" for TN)
    is_positive: bool      # True → wrong layout (should detect); False → correct layout


def build_wrong_layout_text(word: str, native_lang: str, active_lang: str) -> Optional[str]:
    """Simulate typing *word* (native to native_lang) on active_lang's layout.

    Returns the garbled string that will appear on screen, or None if the
    conversion produces the same string (nothing to detect).
    """
    # The user's fingers hit keys for native_lang characters but the OS
    # interprets them through active_lang's layout.
    # So: map native_lang positions → active_lang characters.
    garbled = convert_text_bidirectional(
        word,
        LAYOUT_MAP[native_lang],
        LAYOUT_MAP[active_lang],
    )
    if garbled == word:
        return None  # no difference → skip
    return garbled


def build_test_cases(
    sample_per_lang: int = 200,
    include_phrases: bool = False,
    seed: int = 42,
) -> List[TestCase]:
    """Build a balanced set of true-positive and true-negative cases."""
    rng = random.Random(seed)
    cases: List[TestCase] = []

    langs = ["en", "ru", "he"]
    vocabs = {lang: load_vocabulary(lang) for lang in langs}

    for lang in langs:
        pool = vocabs[lang]
        if not pool:
            print(f"  [warn] No vocabulary for {lang}, skipping.")
            continue
        sampled = rng.sample(pool, min(sample_per_lang, len(pool)))

        # --- True negatives: word typed on correct layout ---
        for word in sampled:
            cases.append(TestCase(
                text=word,
                current_layout=lang,
                expected_lang="",
                is_positive=False,
            ))

        # --- True positives: word typed on each *wrong* layout ---
        other_langs = [l for l in langs if l != lang]
        for active_lang in other_langs:
            for word in sampled:
                garbled = build_wrong_layout_text(word, lang, active_lang)
                if garbled is None:
                    continue
                cases.append(TestCase(
                    text=garbled,
                    current_layout=active_lang,
                    expected_lang=lang,
                    is_positive=True,
                ))

    # --- Optional: 2-word phrases ---
    if include_phrases:
        for lang in langs:
            pool = vocabs[lang]
            if len(pool) < 2:
                continue
            phrase_count = min(sample_per_lang // 2, len(pool) // 2)
            pairs = [rng.sample(pool, 2) for _ in range(phrase_count)]

            for w1, w2 in pairs:
                phrase = f"{w1} {w2}"
                # TN
                cases.append(TestCase(
                    text=phrase,
                    current_layout=lang,
                    expected_lang="",
                    is_positive=False,
                ))
                # TPs on wrong layouts
                other_langs = [l for l in langs if l != lang]
                for active_lang in other_langs:
                    garbled = build_wrong_layout_text(phrase, lang, active_lang)
                    if garbled is None:
                        continue
                    cases.append(TestCase(
                        text=garbled,
                        current_layout=active_lang,
                        expected_lang=lang,
                        is_positive=True,
                    ))

    rng.shuffle(cases)
    return cases


# ---------------------------------------------------------------------------
# Detection helpers (mirrors main.py logic, but parameterized)
# ---------------------------------------------------------------------------
LAYOUT_PAIRS = [
    (english_layout, russian_layout),
    (russian_layout, english_layout),
    (hebrew_layout, english_layout),
    (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),
    (hebrew_layout, russian_layout),
]


def _deduplicate_ordered(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _should_skip_detection(text: str) -> bool:
    if "://" in text or "www." in text or "http" in text:
        return True
    if len(text) > 2 and text[1] == ":" and text[2] in ("\\", "/"):
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < len(text) / 2:
        return True
    return False


def get_required_confidence(
    num_chars: int,
    early_min: int,
    full_conf: int,
    conf_at_min: float,
    conf_at_max: float,
) -> float:
    """Parameterized version of config.get_required_confidence."""
    if num_chars < early_min:
        return 1.1  # impossible
    if num_chars >= full_conf:
        return conf_at_max
    t = (num_chars - early_min) / (full_conf - early_min)
    return conf_at_min + t * (conf_at_max - conf_at_min)


def simulate_detection(
    text: str,
    model_args: list,
    early_min: int,
    full_conf: int,
    conf_at_min: float,
    conf_at_max: float,
) -> Tuple[Optional[str], int]:
    """Simulate incremental typing and return (detected_lang, chars_at_detection).

    Returns (None, len(text)) if no detection fires before end-of-input.
    """
    for n in range(1, len(text) + 1):
        partial = text[:n]

        if n < early_min:
            continue

        if _should_skip_detection(partial):
            continue

        required = get_required_confidence(n, early_min, full_conf, conf_at_min, conf_at_max)

        # Generate all 6 layout variants
        variants = [convert_text_bidirectional(partial, src, dst) for src, dst in LAYOUT_PAIRS]
        variants = _deduplicate_ordered(variants)

        best_lang = None
        best_conf = 0.0

        for variant in variants:
            result = predict_language_with_confidence(variant, *model_args)
            if result is not None and result.confidence > best_conf:
                best_conf = result.confidence
                best_lang = result.language

        if best_lang and best_conf >= required:
            return best_lang, n

    return None, len(text)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    tp_rate: float
    fp_rate: float
    avg_chars_to_detect: float
    composite: float
    tp_count: int
    fp_count: int
    tn_count: int
    fn_count: int
    total_positives: int
    total_negatives: int


def evaluate(
    cases: List[TestCase],
    model_args: list,
    early_min: int,
    full_conf: int,
    conf_at_min: float,
    conf_at_max: float,
) -> Metrics:
    """Run all test cases under a parameter set and compute metrics."""
    tp = fp = tn = fn = 0
    chars_to_detect_sum = 0
    tp_detected = 0
    total_positives = sum(1 for c in cases if c.is_positive)
    total_negatives = sum(1 for c in cases if not c.is_positive)

    for case in cases:
        det_lang, det_chars = simulate_detection(
            case.text, model_args, early_min, full_conf, conf_at_min, conf_at_max,
        )

        if case.is_positive:
            # Should fire with expected_lang
            if det_lang == case.expected_lang:
                tp += 1
                tp_detected += 1
                chars_to_detect_sum += det_chars
            elif det_lang is not None:
                # Fired with wrong language → count as FP and FN
                fp += 1
                fn += 1
            else:
                fn += 1
        else:
            # Should NOT fire (true negative)
            if det_lang is None:
                tn += 1
            else:
                fp += 1

    tp_rate = tp / total_positives if total_positives else 0.0
    fp_rate = fp / total_negatives if total_negatives else 0.0
    avg_chars = chars_to_detect_sum / tp_detected if tp_detected else float("inf")

    # Composite: reward TP rate, heavily penalise FP rate, mildly penalise slow detection
    composite = tp_rate - 2.0 * fp_rate - 0.02 * (avg_chars if avg_chars != float("inf") else 20)

    return Metrics(
        tp_rate=tp_rate,
        fp_rate=fp_rate,
        avg_chars_to_detect=avg_chars,
        composite=composite,
        tp_count=tp,
        fp_count=fp,
        tn_count=tn,
        fn_count=fn,
        total_positives=total_positives,
        total_negatives=total_negatives,
    )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def build_grid() -> list:
    """Return list of (early_min, full_conf, conf_at_min, conf_at_max) tuples."""
    early_mins = [2, 3, 4, 5]
    full_confs = [6, 8, 10, 12, 15]
    conf_at_mins = [0.90, 0.93, 0.95, 0.97, 0.99]
    conf_at_maxs = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    grid = []
    for em, fc, cm, cx in itertools.product(early_mins, full_confs, conf_at_mins, conf_at_maxs):
        if em >= fc:
            continue  # EarlyDetection must be < FullConfidence
        if cx >= cm:
            continue  # floor must be lower than initial confidence
        grid.append((em, fc, cm, cx))
    return grid


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_results(results: list, top_n: int = 25):
    """Print the top-N parameter combos in a table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = [
        "Rank",
        "EarlyMin",
        "FullConf",
        "ConfAtMin",
        "ConfAtMax",
        "TP Rate",
        "FP Rate",
        "Avg Chars",
        "Composite",
        "TP",
        "FP",
        "FN",
    ]
    rows = []
    for i, (params, m) in enumerate(results[:top_n], 1):
        rows.append([
            i,
            params[0],
            params[1],
            f"{params[2]:.2f}",
            f"{params[3]:.2f}",
            f"{m.tp_rate:.4f}",
            f"{m.fp_rate:.4f}",
            f"{m.avg_chars_to_detect:.2f}" if m.avg_chars_to_detect != float("inf") else "N/A",
            f"{m.composite:.4f}",
            m.tp_count,
            m.fp_count,
            m.fn_count,
        ])

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        # Fallback: simple column printing
        print("  ".join(f"{h:>10}" for h in headers))
        print("-" * (12 * len(headers)))
        for row in rows:
            print("  ".join(f"{str(v):>10}" for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tune adaptive-confidence-curve parameters")
    parser.add_argument("--sample", type=int, default=200,
                        help="Words to sample per language (default: 200)")
    parser.add_argument("--phrases", action="store_true",
                        help="Also test 2-word phrases")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to write full results CSV")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top results to display (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print("  Keyboard Switcher — Adaptive Confidence Curve Tuning")
    print("=" * 70)

    # Load model once
    # The ONNX model may live in ../cpp/ rather than the python/ folder.
    print("\n[1/4] Loading ONNX model …")
    onnx_in_python = os.path.join(SCRIPT_DIR, "lang_model.onnx")
    onnx_in_cpp = os.path.join(SCRIPT_DIR, "..", "cpp", "lang_model.onnx")
    if not os.path.exists(onnx_in_python) and os.path.exists(onnx_in_cpp):
        import shutil
        shutil.copy2(onnx_in_cpp, onnx_in_python)
        print(f"       (copied lang_model.onnx from cpp/ into python/)")
    model_args = Languages.load_model()
    print("       Model loaded.\n")

    # Build test cases
    print(f"[2/4] Building test cases (sample={args.sample}, phrases={args.phrases}) …")
    cases = build_test_cases(
        sample_per_lang=args.sample,
        include_phrases=args.phrases,
        seed=args.seed,
    )
    n_pos = sum(1 for c in cases if c.is_positive)
    n_neg = sum(1 for c in cases if not c.is_positive)
    print(f"       {len(cases)} total cases  ({n_pos} positives, {n_neg} negatives)\n")

    # Build grid
    grid = build_grid()
    print(f"[3/4] Grid search over {len(grid)} parameter combinations …")
    print(f"       (this may take a while)\n")

    results: list = []
    t0 = time.time()

    for idx, (em, fc, cm, cx) in enumerate(grid, 1):
        m = evaluate(cases, model_args, em, fc, cm, cx)
        results.append(((em, fc, cm, cx), m))

        # Progress
        if idx % 10 == 0 or idx == len(grid):
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (len(grid) - idx)
            print(f"       [{idx:>4}/{len(grid)}]  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"current best composite={max(r[1].composite for r in results):.4f}",
                  end="\r")

    elapsed_total = time.time() - t0
    print(f"\n       Done in {elapsed_total:.1f}s.\n")

    # Sort by composite score (descending)
    results.sort(key=lambda r: r[1].composite, reverse=True)

    # Display top results
    print(f"[4/4] Top {args.top} parameter sets by composite score:\n")
    print_results(results, top_n=args.top)

    # Best result summary
    best_params, best_m = results[0]
    print("\n" + "=" * 70)
    print("  RECOMMENDED PARAMETERS")
    print("=" * 70)
    print(f"  EarlyDetectionMinChars = {best_params[0]}")
    print(f"  FullConfidenceChars    = {best_params[1]}")
    print(f"  ConfidenceAtMinChars   = {best_params[2]:.2f}")
    print(f"  ConfidenceAtMaxChars   = {best_params[3]:.2f}")
    print()
    print(f"  TP rate:           {best_m.tp_rate:.4f}  ({best_m.tp_count}/{best_m.total_positives})")
    print(f"  FP rate:           {best_m.fp_rate:.4f}  ({best_m.fp_count}/{best_m.total_negatives})")
    avg_str = f"{best_m.avg_chars_to_detect:.2f}" if best_m.avg_chars_to_detect != float("inf") else "N/A"
    print(f"  Avg chars-to-det:  {avg_str}")
    print(f"  Composite score:   {best_m.composite:.4f}")
    print("=" * 70)

    # Optional: compare with current defaults
    print("\n  Comparison with CURRENT defaults (3, 10, 0.97, 0.55):")
    current_m = evaluate(cases, model_args, 3, 10, 0.97, 0.55)
    print(f"  TP rate:           {current_m.tp_rate:.4f}  ({current_m.tp_count}/{current_m.total_positives})")
    print(f"  FP rate:           {current_m.fp_rate:.4f}  ({current_m.fp_count}/{current_m.total_negatives})")
    avg_str_curr = f"{current_m.avg_chars_to_detect:.2f}" if current_m.avg_chars_to_detect != float("inf") else "N/A"
    print(f"  Avg chars-to-det:  {avg_str_curr}")
    print(f"  Composite score:   {current_m.composite:.4f}")
    print()

    # CSV export
    if args.csv:
        csv_path = os.path.join(SCRIPT_DIR, args.csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "EarlyMin", "FullConf", "ConfAtMin", "ConfAtMax",
                "TP_Rate", "FP_Rate", "Avg_Chars", "Composite",
                "TP", "FP", "TN", "FN",
                "Total_Pos", "Total_Neg",
            ])
            for (em, fc, cm, cx), m in results:
                writer.writerow([
                    em, fc, f"{cm:.2f}", f"{cx:.2f}",
                    f"{m.tp_rate:.6f}", f"{m.fp_rate:.6f}",
                    f"{m.avg_chars_to_detect:.4f}" if m.avg_chars_to_detect != float("inf") else "",
                    f"{m.composite:.6f}",
                    m.tp_count, m.fp_count, m.tn_count, m.fn_count,
                    m.total_positives, m.total_negatives,
                ])
        print(f"  Full results written to: {csv_path}")


if __name__ == "__main__":
    main()


