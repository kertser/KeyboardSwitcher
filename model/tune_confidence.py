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


def load_vocabulary(lang: str, min_len: int = 2, max_len: int = 20) -> List[str]:
    """Load words from a vocabulary file, filtering by length.

    min_len defaults to 2 so that common short words (e.g. Hebrew מה/כן/לא)
    are included in the test set and phrase combinations.
    """
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
    """Build a balanced set of true-positive and true-negative cases.

    Always includes short-Hebrew-phrase cases (2-char first word + longer
    second word) because these are a known blind-spot: the first word alone
    never reaches EarlyDetectionMinChars=3, so detection can only fire once
    the user starts typing the second word.  These must be covered explicitly
    so the grid-search finds parameters that handle them.
    """
    rng = random.Random(seed)
    cases: List[TestCase] = []

    langs = ["en", "ru", "he"]
    # Load with min_len=2 so 2-char words (מה, כן, לא, גם …) are in the pool
    vocabs = {lang: load_vocabulary(lang, min_len=2) for lang in langs}

    for lang in langs:
        pool = vocabs[lang]
        if not pool:
            print(f"  [warn] No vocabulary for {lang}, skipping.")
            continue
        # For single-word test cases keep min_len=3 to avoid trivially short tests
        pool_single = [w for w in pool if len(w) >= 3]
        sampled = rng.sample(pool_single, min(sample_per_lang, len(pool_single)))

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

    # --- Short-phrase Hebrew cases (always included, not behind --phrases) ---
    # Hebrew has many common 2-letter words (מה, כן, לא, גם, כי, אם, אך, …).
    # When typed on the wrong layout the total alpha count is only 2 until the
    # space and the second word arrive.  These cases stress-test that the
    # detection curve fires quickly once more characters follow the space.
    he_short_pool = [w for w in vocabs["he"] if len(w) == 2]
    he_long_pool  = [w for w in vocabs["he"] if len(w) >= 4]
    if he_short_pool and he_long_pool:
        short_phrase_count = min(sample_per_lang // 2, len(he_short_pool),
                                  len(he_long_pool))
        short_words = rng.sample(he_short_pool, short_phrase_count)
        long_words  = rng.sample(he_long_pool,  short_phrase_count)
        for w1, w2 in zip(short_words, long_words):
            phrase = f"{w1} {w2}"
            # TN: phrase typed on correct Hebrew layout
            cases.append(TestCase(
                text=phrase,
                current_layout="he",
                expected_lang="",
                is_positive=False,
            ))
            # TP: phrase typed on wrong layouts (en, ru)
            for active_lang in ["en", "ru"]:
                garbled = build_wrong_layout_text(phrase, "he", active_lang)
                if garbled is None:
                    continue
                cases.append(TestCase(
                    text=garbled,
                    current_layout=active_lang,
                    expected_lang="he",
                    is_positive=True,
                ))

    # --- Optional: generic 2-word phrases for all languages ---
    if include_phrases:
        for lang in langs:
            pool = vocabs[lang]
            pool_phrases = [w for w in pool if len(w) >= 3]
            if len(pool_phrases) < 2:
                continue
            phrase_count = min(sample_per_lang // 2, len(pool_phrases) // 2)
            pairs = [rng.sample(pool_phrases, 2) for _ in range(phrase_count)]

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
    """Skip detection for URLs, drive paths, or similarly non-language input.

    This mirrors the URL/path guard in main.cpp.  The alpha-count / detection-
    text-length checks that used to live here have been moved into
    simulate_detection() so that they use alpha_count (not raw len) consistent
    with the C++ implementation.
    """
    if "://" in text or "www." in text or "http" in text:
        return True
    if len(text) > 2 and text[1] == ":" and text[2] in ("\\", "/"):
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
    consecutive_required: int = 2,
) -> Tuple[Optional[str], int]:
    """Simulate incremental typing and return (detected_lang, chars_at_detection).

    Returns (None, len(text)) if no detection fires before end-of-input.

    Key alignment with C++ main.cpp:
      • alpha_count  – only alphabetic characters count toward EarlyDetectionMinChars
                       and toward the confidence-threshold interpolation (not raw len).
      • detection_text – alpha chars + spaces, trimmed; its *length* must also reach
                         EarlyDetectionMinChars before detection is attempted.
      • consecutive_required – mirrors ConsecutiveAgreementCount: the model must
                         predict the same language on this many consecutive keystrokes
                         before the detection fires.
    """
    # Consecutive-agreement state (mirrors DetectionHistory in C++)
    last_lang: str = ""
    streak: int = 0

    for n in range(1, len(text) + 1):
        partial = text[:n]

        # Build detection_text: alpha chars + spaces, trimmed (mirrors C++ detectionText)
        detection_text = "".join(c for c in partial if c.isalpha() or c == " ").strip()
        alpha_count = sum(1 for c in partial if c.isalpha())

        # Skip check: matches C++
        #   alphaCount < EarlyDetectionMinChars  OR
        #   detectionText.size() < EarlyDetectionMinChars
        if alpha_count < early_min or len(detection_text) < early_min:
            continue

        # URL / path guard (unchanged)
        if _should_skip_detection(partial):
            continue

        # Confidence threshold keyed on alpha_count (not raw n)
        required = get_required_confidence(alpha_count, early_min, full_conf, conf_at_min, conf_at_max)

        # Generate all 6 layout variants from detection_text
        variants = [convert_text_bidirectional(detection_text, src, dst) for src, dst in LAYOUT_PAIRS]
        variants = _deduplicate_ordered(variants)

        best_lang = None
        best_conf = 0.0

        for variant in variants:
            result = predict_language_with_confidence(variant, *model_args)
            if result is not None and result.confidence > best_conf:
                best_conf = result.confidence
                best_lang = result.language

        # History update (mirrors DetectionHistory::Update; runs even if conf < threshold)
        if best_lang:
            if best_lang == last_lang:
                streak += 1
            else:
                last_lang = best_lang
                streak = 1
        else:
            last_lang = ""
            streak = 0
            continue

        # Consecutive-agreement gate (mirrors ConsecutiveAgreementCount check in C++)
        if streak < consecutive_required:
            continue

        # Confidence gate
        if best_conf >= required:
            return best_lang, n

    return None, len(text)


# ---------------------------------------------------------------------------
# Precomputed per-character detection state
# ---------------------------------------------------------------------------
# Model inference is independent of the confidence parameters.
# We precompute (alpha_count, det_text_len, best_lang, best_conf) once per
# (test-case, char-position) and then the grid search merely applies thresholds —
# this reduces runtime from O(grid_size × cases × chars × model_calls) to
# O(cases × chars × model_calls)  +  O(grid_size × cases × chars).
@dataclass
class PerCharState:
    n: int                   # 1-based character position (raw text length)
    alpha_count: int         # number of alphabetic chars in text[:n]
    det_text_len: int        # len of detection_text (alpha+space trimmed)
    best_lang: Optional[str] # top predicted language (or None if N/A wins)
    best_conf: float         # softmax confidence of top language


def precompute_case(text: str, model_args: list) -> List[PerCharState]:
    """Compute per-char detection states for a single test-case text.

    Only characters at or beyond the minimum possible early_min (= 2) are
    processed; earlier positions are skipped to save time.
    """
    # Smallest early_min in the grid = 2; anything below that is always skipped
    MIN_EARLY_MIN = 2

    # Cache model outputs keyed on detection_text (avoid re-running the model
    # on identical strings when the same detection_text appears across positions)
    cache: dict = {}

    states: List[PerCharState] = []
    for n in range(1, len(text) + 1):
        partial = text[:n]

        detection_text = "".join(c for c in partial if c.isalpha() or c == " ").strip()
        alpha_count = sum(1 for c in partial if c.isalpha())

        # Always record state (grid loop needs to skip based on early_min)
        if alpha_count < MIN_EARLY_MIN or len(detection_text) < MIN_EARLY_MIN:
            states.append(PerCharState(n, alpha_count, len(detection_text), None, 0.0))
            continue

        if _should_skip_detection(partial):
            states.append(PerCharState(n, alpha_count, len(detection_text), None, 0.0))
            continue

        if detection_text in cache:
            best_lang, best_conf = cache[detection_text]
        else:
            variants = [convert_text_bidirectional(detection_text, src, dst)
                        for src, dst in LAYOUT_PAIRS]
            variants = _deduplicate_ordered(variants)

            best_lang = None
            best_conf = 0.0
            for variant in variants:
                result = predict_language_with_confidence(variant, *model_args)
                if result is not None and result.confidence > best_conf:
                    best_conf = result.confidence
                    best_lang = result.language

            cache[detection_text] = (best_lang, best_conf)

        states.append(PerCharState(n, alpha_count, len(detection_text), best_lang, best_conf))

    return states




def simulate_detection_precomputed(
    states: List[PerCharState],
    current_layout: str,
    early_min: int,
    full_conf: int,
    conf_at_min: float,
    conf_at_max: float,
    consecutive_required: int = 2,
) -> Tuple[Optional[str], int]:
    """Fast variant of simulate_detection using pre-computed model outputs.

    current_layout: the active keyboard layout (mirrors currentLangId in C++).
    Only returns a non-None language when the detected language DIFFERS from
    current_layout — this mirrors the C++ guard `if (currentLangId != bestLang)`.
    """
    last_lang: str = ""
    streak: int = 0

    for state in states:
        if state.alpha_count < early_min or state.det_text_len < early_min:
            continue

        if state.best_lang is None:
            last_lang = ""
            streak = 0
            continue

        # History update (always, regardless of current_layout — mirrors C++)
        if state.best_lang == last_lang:
            streak += 1
        else:
            last_lang = state.best_lang
            streak = 1

        # Consecutive-agreement gate
        if streak < consecutive_required:
            continue

        # Mirror C++: only trigger a switch when bestLang != currentLangId
        if state.best_lang == current_layout:
            continue

        # Confidence gate
        required = get_required_confidence(state.alpha_count, early_min, full_conf,
                                           conf_at_min, conf_at_max)
        if state.best_conf >= required:
            return state.best_lang, state.n

    return None, states[-1].n if states else 0


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
    consecutive_required: int = 2,
) -> Metrics:
    """Run all test cases (no precomputation) — kept for quick ad-hoc calls."""
    tp = fp = tn = fn = 0
    chars_to_detect_sum = 0
    tp_detected = 0
    total_positives = sum(1 for c in cases if c.is_positive)
    total_negatives = sum(1 for c in cases if not c.is_positive)

    for case in cases:
        det_lang, det_chars = simulate_detection(
            case.text, model_args, early_min, full_conf, conf_at_min, conf_at_max,
            consecutive_required=consecutive_required,
        )

        if case.is_positive:
            if det_lang == case.expected_lang:
                tp += 1
                tp_detected += 1
                chars_to_detect_sum += det_chars
            elif det_lang is not None:
                fp += 1
                fn += 1
            else:
                fn += 1
        else:
            if det_lang is None:
                tn += 1
            else:
                fp += 1

    tp_rate = tp / total_positives if total_positives else 0.0
    fp_rate = fp / total_negatives if total_negatives else 0.0
    avg_chars = chars_to_detect_sum / tp_detected if tp_detected else float("inf")
    composite = tp_rate - 2.0 * fp_rate - 0.02 * (avg_chars if avg_chars != float("inf") else 20)

    return Metrics(
        tp_rate=tp_rate, fp_rate=fp_rate, avg_chars_to_detect=avg_chars,
        composite=composite, tp_count=tp, fp_count=fp, tn_count=tn, fn_count=fn,
        total_positives=total_positives, total_negatives=total_negatives,
    )


def evaluate_precomputed(
    cases: List[TestCase],
    precomputed: List[List[PerCharState]],
    early_min: int,
    full_conf: int,
    conf_at_min: float,
    conf_at_max: float,
    consecutive_required: int = 2,
) -> Metrics:
    """Fast evaluation using precomputed per-char model states."""
    tp = fp = tn = fn = 0
    chars_to_detect_sum = 0
    tp_detected = 0
    total_positives = sum(1 for c in cases if c.is_positive)
    total_negatives = sum(1 for c in cases if not c.is_positive)

    for case, states in zip(cases, precomputed):
        det_lang, det_chars = simulate_detection_precomputed(
            states, case.current_layout,
            early_min, full_conf, conf_at_min, conf_at_max,
            consecutive_required=consecutive_required,
        )

        if case.is_positive:
            # Should switch to expected_lang
            if det_lang == case.expected_lang:
                tp += 1
                tp_detected += 1
                chars_to_detect_sum += det_chars
            elif det_lang is not None:
                # Fired with wrong (3rd) language → both a false positive and a miss
                fp += 1
                fn += 1
            else:
                # No switch → missed detection
                fn += 1
        else:
            # Should NOT switch (text is in the current layout language)
            if det_lang is None:
                tn += 1  # correctly no switch
            else:
                fp += 1  # incorrectly switched to a different language

    tp_rate = tp / total_positives if total_positives else 0.0
    fp_rate = fp / total_negatives if total_negatives else 0.0
    avg_chars = chars_to_detect_sum / tp_detected if tp_detected else float("inf")
    composite = tp_rate - 2.0 * fp_rate - 0.02 * (avg_chars if avg_chars != float("inf") else 20)

    return Metrics(
        tp_rate=tp_rate, fp_rate=fp_rate, avg_chars_to_detect=avg_chars,
        composite=composite, tp_count=tp, fp_count=fp, tn_count=tn, fn_count=fn,
        total_positives=total_positives, total_negatives=total_negatives,
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
                        help="Also test generic 2-word phrases (short Hebrew phrases are always included)")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to write full results CSV")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top results to display (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--consecutive", type=int, default=2,
                        help="ConsecutiveAgreementCount to simulate (default: 2, mirrors C++)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Keyboard Switcher — Adaptive Confidence Curve Tuning")
    print("=" * 70)
    print(f"\n  ConsecutiveAgreementCount = {args.consecutive}  (--consecutive to change)")

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

    # Precompute per-character model states (done once; the slow step)
    print(f"[3/4] Precomputing model states for all test cases …")
    t_pre = time.time()
    precomputed: List[List[PerCharState]] = []
    for i, case in enumerate(cases, 1):
        precomputed.append(precompute_case(case.text, model_args))
        if i % 100 == 0 or i == len(cases):
            elapsed_pre = time.time() - t_pre
            eta_pre = (elapsed_pre / i) * (len(cases) - i)
            print(f"       [{i:>4}/{len(cases)}]  "
                  f"elapsed {elapsed_pre:.0f}s  ETA {eta_pre:.0f}s",
                  end="\r")
    print(f"\n       Precomputation done in {time.time() - t_pre:.1f}s.\n")

    # Grid search (fast — just threshold comparisons on precomputed states)
    grid = build_grid()
    print(f"[4/4] Grid search over {len(grid)} parameter combinations …")
    results: list = []
    t0 = time.time()

    for idx, (em, fc, cm, cx) in enumerate(grid, 1):
        m = evaluate_precomputed(cases, precomputed, em, fc, cm, cx,
                                 consecutive_required=args.consecutive)
        results.append(((em, fc, cm, cx), m))

        if idx % 50 == 0 or idx == len(grid):
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (len(grid) - idx)
            print(f"       [{idx:>4}/{len(grid)}]  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s  "
                  f"best composite={max(r[1].composite for r in results):.4f}",
                  end="\r")

    print(f"\n       Grid search done in {time.time() - t0:.1f}s.\n")

    # Sort by composite score (descending)
    results.sort(key=lambda r: r[1].composite, reverse=True)

    # Display top results
    print(f"  Top {args.top} parameter sets by composite score:\n")
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

    # Compare with current defaults
    print("\n  Comparison with CURRENT defaults (3, 10, 0.97, 0.55):")
    current_m = evaluate_precomputed(cases, precomputed, 3, 10, 0.97, 0.55,
                                     consecutive_required=args.consecutive)
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


