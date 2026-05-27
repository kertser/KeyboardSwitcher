#!/usr/bin/env python3
"""Evaluate detection quality for all language transitions using vocabulary lists.

Reports per directed pair (from->to):
- TP / FP / FN / TN counts
- Precision, Recall, F1
- FP rate, FN rate
- Avg chars-to-detect (TPs only)
- Short-word breakdown (3-5 chars): same metrics

This is a lightweight offline harness to validate per-pair macro-parameters.
It mirrors ALL gates present in TypoResilientDetect (C++) including:
  - margin gate (MinTop1Top2Margin)
  - short-input extra confidence (ShortInputExtraConf)
  - variant consensus gate (VariantAgreementCount)
  - trend gate (TrendWindowSize / MinStableSteps)
  - consecutive-agreement gate

Usage:
    python evaluate_transitions.py
    python evaluate_transitions.py --sample 200 --csv baseline.csv
    python evaluate_transitions.py --short-only   # only 3-5 char words
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# Must match cpp/src/Config.cpp PairOverrides (all 12 fields)
# (EarlyMin, FullConf, ConfAtMin, ConfAtMax, ConsecAgree, BorderlineFactor,
#  MinMargin, VariantAgree, TrendWindow, MinSteps, MinSlope, ShortExtraConf)
DEFAULT_PARAMS = (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02)
PAIR_OVERRIDES = {
    ("en", "ru"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    ("ru", "en"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    # →he: PhraseConfScale=0.80 — phrases fire at 80% of normal threshold
    #       ConfAtMin=0.97, MinMargin=0.10 guard against FP,
    #       VariantAgree=0 (only 1 variant produces Hebrew per word)
    ("en", "he"): (3, 15, 0.97, 0.65, 2, 0.88, 0.10, 0, 4, 2, 0.0, 0.0,  0.80),
    ("he", "en"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    ("ru", "he"): (3, 15, 0.97, 0.65, 2, 0.88, 0.10, 0, 4, 2, 0.0, 0.0,  0.80),
    ("he", "ru"): (3, 15, 0.99, 0.70, 2, 0.80, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
}

LAYOUT_PAIRS = [
    (english_layout, russian_layout),
    (russian_layout, english_layout),
    (hebrew_layout, english_layout),
    (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),
    (hebrew_layout, russian_layout),
]


def load_vocab(lang: str, sample: int, seed: int,
               min_len: int, max_len: int) -> List[str]:
    words: List[str] = []
    with open(VOCAB_FILES[lang], encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if min_len <= len(w) <= max_len and w.isalpha():
                words.append(w)
    rng = random.Random(seed)
    return rng.sample(words, min(sample, len(words)))


def required_confidence(n: int, params: tuple, is_phrase: bool = False) -> float:
    early_min, full_conf, conf_min, conf_max = params[0], params[1], params[2], params[3]
    phrase_scale = params[12] if len(params) > 12 else 1.0
    if n < early_min:
        return 1.1
    if n >= full_conf:
        base = conf_max
    else:
        t = (n - early_min) / (full_conf - early_min)
        base = conf_min + t * (conf_max - conf_min)
    if is_phrase and phrase_scale < 1.0:
        base *= phrase_scale
    return base


def get_runner_up(result) -> Tuple[str, float]:
    """Return (runner_up_lang, runner_up_conf) from a DetectionResult."""
    if result is None:
        return "", 0.0
    scores = result.scores  # [N/A, en, he, ru]
    class_langs = ["", "en", "he", "ru"]
    best_lang = result.language
    runner_conf = 0.0
    runner_lang = ""
    for i in range(1, 4):
        lang_i = class_langs[i]
        if lang_i != best_lang and scores[i] > runner_conf:
            runner_conf = scores[i]
            runner_lang = lang_i
    return runner_lang, runner_conf


@dataclass
class PairMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    tp_total: int = 0   # total positive (TP) samples
    fp_total: int = 0   # total negative (TN) samples
    chars_sum: int = 0  # sum of chars-to-detect for TPs

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.tp_total if self.tp_total > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        return self.fp / self.fp_total if self.fp_total > 0 else 0.0

    @property
    def fn_rate(self) -> float:
        return self.fn / self.tp_total if self.tp_total > 0 else 0.0

    @property
    def avg_chars(self) -> float:
        return self.chars_sum / self.tp if self.tp > 0 else float("inf")


def _simulate_word(
    text: str,
    current_lang: str,
    params: tuple,
    model_args: list,
    predict_cache: dict,
) -> Tuple[Optional[str], int]:
    """Simulate incremental typing and return (detected_lang, chars_at_detection).

    Mirrors ALL gates from TypoResilientDetect in C++:
      - per-pair min-chars
      - short-input extra confidence
      - margin gate
      - variant consensus gate
      - consecutive-agreement gate
      - trend gate (OR-alternative)
      - confidence threshold
    """
    (early_min, full_conf, conf_min, conf_max,
     agreement_count, _border,
     min_margin, variant_agree,
     trend_window, min_steps, min_slope, short_extra_conf,
     phrase_conf_scale) = (*params[:12], params[12] if len(params) > 12 else 1.0)

    last_lang = ""
    streak = 0
    window: List[tuple] = []   # (top1lang, top1conf, runner_conf, margin)

    for n in range(1, len(text) + 1):
        partial = text[:n]
        det_text = "".join(c for c in partial if c.isalpha() or c == " ").strip()
        alpha_count = sum(1 for c in partial if c.isalpha())

        if alpha_count < early_min or len(det_text) < early_min:
            continue

        # Confidence threshold with optional short-input boost
        is_phrase = " " in det_text
        req = required_confidence(alpha_count, params, is_phrase=is_phrase)
        is_short = (alpha_count <= early_min + 2)
        # Short-input extra conf not applied in phrase mode (PhraseConfScale handles it)
        if is_short and not is_phrase and short_extra_conf > 0:
            req = min(0.9999, req + short_extra_conf)

        # Build layout variants
        seen: set = set()
        variants = []
        for src, dst in LAYOUT_PAIRS:
            v = convert_text_bidirectional(det_text, src, dst)
            if v not in seen:
                seen.add(v)
                variants.append(v)

        best_lang_step: Optional[str] = None
        best_conf_step = 0.0
        best_result = None
        lang_votes: Dict[str, int] = {}

        for v in variants:
            r = predict_cache.get(v)
            if r is None:
                r = predict_language_with_confidence(v, *model_args)
                predict_cache[v] = r
            if r is not None:
                lang_votes[r.language] = lang_votes.get(r.language, 0) + 1
                if r.confidence > best_conf_step:
                    best_conf_step = r.confidence
                    best_lang_step = r.language
                    best_result = r

        if not best_lang_step:
            last_lang = ""
            streak = 0
            window = []
            continue

        runner_lang, runner_conf = get_runner_up(best_result)
        margin = best_conf_step - runner_conf

        # Update streak (always)
        if best_lang_step == last_lang:
            streak += 1
        else:
            last_lang = best_lang_step
            streak = 1

        # Update sliding window
        window.append((best_lang_step, best_conf_step, runner_conf, margin))
        if len(window) > trend_window:
            window = window[-trend_window:]

        # --- Margin gate ---
        if min_margin > 0 and margin < min_margin:
            continue

        # --- Variant consensus gate (borderline only) ---
        v_agree = lang_votes.get(best_lang_step, 0)
        if variant_agree > 0 and best_conf_step < conf_min * 0.99:
            if v_agree < variant_agree:
                continue

        # --- Only switch when detected != current ---
        if best_lang_step == current_lang:
            continue

        # --- Agreement gate ---
        agreement_ok = (streak >= agreement_count)

        # --- Trend gate (OR-alternative) ---
        trend_ok = False
        if trend_window > 0 and min_steps > 0:
            to_frames = [f for f in window if f[0] == best_lang_step]
            if len(to_frames) >= min_steps:
                if min_slope > 0 and len(to_frames) >= 2:
                    slope = ((to_frames[-1][1] - to_frames[0][1])
                             / max(1, len(to_frames) - 1))
                    if slope >= min_slope:
                        trend_ok = True
                else:
                    trend_ok = True

        if not agreement_ok and not trend_ok:
            continue

        # --- Confidence gate ---
        if best_conf_step >= req:
            return best_lang_step, n

    return None, len(text)


def evaluate_pair(
    words_by_lang: Dict[str, List[str]],
    from_lang: str,
    to_lang: str,
    model_args: list,
) -> Tuple[PairMetrics, PairMetrics]:
    """Return (full_metrics, short_metrics) for a directed pair.

    short_metrics covers only words of length 3-5.
    """
    params = PAIR_OVERRIDES.get((from_lang, to_lang), DEFAULT_PARAMS)
    predict_cache: dict = {}

    full_m  = PairMetrics()
    short_m = PairMetrics()

    # True positives: to_lang words typed on from_lang keyboard
    for w in words_by_lang[to_lang]:
        mistyped = convert_text_bidirectional(w, LAYOUTS[to_lang], LAYOUTS[from_lang])
        det, chars = _simulate_word(mistyped, from_lang, params, model_args, predict_cache)
        is_short = (3 <= len(w) <= 5)

        full_m.tp_total += 1
        if is_short:
            short_m.tp_total += 1

        if det == to_lang:
            full_m.tp += 1
            full_m.chars_sum += chars
            if is_short:
                short_m.tp += 1
                short_m.chars_sum += chars
        else:
            full_m.fn += 1
            if is_short:
                short_m.fn += 1

    # False positives: from_lang words typed correctly on from_lang keyboard
    for w in words_by_lang[from_lang]:
        det, _ = _simulate_word(w, from_lang, params, model_args, predict_cache)
        is_short = (3 <= len(w) <= 5)

        full_m.fp_total += 1
        if is_short:
            short_m.fp_total += 1

        if det == to_lang:
            full_m.fp += 1
            if is_short:
                short_m.fp += 1
        else:
            full_m.tn += 1
            if is_short:
                short_m.tn += 1

    return full_m, short_m


def print_metrics_table(rows: list, title: str) -> None:
    print(f"\n{title}")
    header = (f"{'Pair':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'FPrate':>7} {'FNrate':>7} {'AvgChr':>7}")
    print(header)
    print("-" * len(header))
    for pair, m in rows:
        avg = f"{m.avg_chars:.1f}" if m.avg_chars != float("inf") else "  N/A"
        print(
            f"{pair:<10} {m.tp:>5} {m.fp:>5} {m.fn:>5} {m.tn:>5} "
            f"{m.precision:>7.3f} {m.recall:>7.3f} {m.f1:>7.3f} "
            f"{m.fp_rate:>7.4f} {m.fn_rate:>7.4f} {avg:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all directed language transitions with full metrics")
    parser.add_argument("--sample", type=int, default=150,
                        help="Words per language (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--short-only", action="store_true",
                        help="Only test words of length 3-5")
    parser.add_argument("--csv", type=str, default="",
                        help="Write full results to CSV file")
    args = parser.parse_args()

    print("=" * 60)
    print("  KeyboardSwitcher — Directed-Pair Transition Evaluation")
    print("=" * 60)

    print("\nLoading ONNX model...")
    model_args = Languages.load_model()

    min_len = 3 if args.short_only else args.min_len
    max_len = 5 if args.short_only else args.max_len
    words_by_lang = {
        lang: load_vocab(lang, args.sample, args.seed, min_len, max_len)
        for lang in LANGS
    }
    print(f"Vocab: " + "  ".join(
        f"{lang}={len(w)} words" for lang, w in words_by_lang.items()))

    full_rows:  List[Tuple[str, PairMetrics]] = []
    short_rows: List[Tuple[str, PairMetrics]] = []

    for from_lang in LANGS:
        for to_lang in LANGS:
            if from_lang == to_lang:
                continue
            pair = f"{from_lang}->{to_lang}"
            print(f"  {pair} ...", end=" ", flush=True)
            full_m, short_m = evaluate_pair(
                words_by_lang, from_lang, to_lang, model_args)
            full_rows.append((pair, full_m))
            short_rows.append((pair, short_m))
            print(f"TP={full_m.tp}/{full_m.tp_total}  "
                  f"FP={full_m.fp}/{full_m.fp_total}  "
                  f"F1={full_m.f1:.3f}")

    print_metrics_table(full_rows, "=== ALL WORD LENGTHS ===")
    print_metrics_table(short_rows, "=== SHORT WORDS (3-5 chars) ===")

    # Overall summary
    total_fp  = sum(m.fp  for _, m in full_rows)
    total_neg = sum(m.fp_total for _, m in full_rows)
    total_tp  = sum(m.tp  for _, m in full_rows)
    total_pos = sum(m.tp_total for _, m in full_rows)
    wfp = total_fp / total_neg if total_neg else 0.0
    rec = total_tp / total_pos if total_pos else 0.0
    print(f"\n{'='*60}")
    print(f"  Weighted FP rate : {total_fp}/{total_neg} = {wfp:.4f}")
    print(f"  Overall recall   : {total_tp}/{total_pos} = {rec:.4f}")
    print(f"{'='*60}")

    if args.csv:
        csv_path = os.path.join(SCRIPT_DIR, args.csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pair", "subset",
                "TP", "FP", "FN", "TN",
                "precision", "recall", "f1",
                "fp_rate", "fn_rate", "avg_chars",
                "tp_total", "fp_total",
            ])
            for (pair, m), (_, sm) in zip(full_rows, short_rows):
                for subset, met in [("all", m), ("short_3_5", sm)]:
                    writer.writerow([
                        pair, subset,
                        met.tp, met.fp, met.fn, met.tn,
                        f"{met.precision:.4f}", f"{met.recall:.4f}",
                        f"{met.f1:.4f}",
                        f"{met.fp_rate:.4f}", f"{met.fn_rate:.4f}",
                        f"{met.avg_chars:.2f}" if met.avg_chars != float("inf") else "",
                        met.tp_total, met.fp_total,
                    ])
        print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
