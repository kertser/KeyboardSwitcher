#!/usr/bin/env python3
"""Baseline vs hybrid (LSTM + lexical triage) pipeline comparison.

Runs the two offline harnesses — evaluate_transitions (per-directed-pair
TP/FP on vocabulary words) and test_phrases_lite (curated multi-word
phrases) — twice on identical samples/seeds: once with the pure-LSTM
baseline and once with the lexical triage stage enabled.  Reports deltas,
triage latency, and a hard regression verdict:

  FAIL if any pair's FP count increases, aggregate TP decreases, or the
  curated phrase recall decreases versus baseline.

Usage:
  python compare_pipelines.py --sample 200
  python compare_pipelines.py --sample 200 --lsh          # include LSH
  python compare_pipelines.py --sample 200 --ablation     # component ablation
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Languages
import evaluate_transitions as ET
import test_phrases_lite as TP
from triage import LexicalTriage, TriageParams

LANGS = ["en", "ru", "he"]


def run_transitions(words_by_lang, model_args, triage):
    """Returns {(from,to): (tp, tp_total, fp, fp_total)}."""
    out = {}
    for f in LANGS:
        for t in LANGS:
            if f == t:
                continue
            out[(f, t)] = ET.evaluate_pair(words_by_lang, f, t, model_args,
                                           triage=triage)
    return out


def measure_triage_latency(triage, words_by_lang) -> tuple[float, float]:
    """Avg / p99 evidence-call latency in microseconds on realistic inputs
    (cold cache)."""
    samples = []
    for lang in LANGS:
        for w in words_by_lang[lang][:150]:
            samples.append((w, lang))
            # also a cross-script miss (cheap path) and a phrase
            samples.append((w, "en" if lang != "en" else "ru"))
            samples.append((w + " " + w, lang))
    triage.clear_cache()
    times = []
    for text, lang in samples:
        t0 = time.perf_counter()
        triage.evidence(text, lang)
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return (statistics.fmean(times), times[int(len(times) * 0.99)])


def print_comparison(base, hyb, phrase_base, phrase_hyb, phrase_total,
                     label="triage"):
    print(f"\n{'pair':<8} {'TP base':>12} {'TP ' + label:>12} {'dTP':>5}   "
          f"{'FP base':>10} {'FP ' + label:>10} {'dFP':>5}")
    tp_b = tp_h = fp_b = fp_h = 0
    fp_regressed = []
    for pair in base:
        b_tp, tp_tot, b_fp, fp_tot = base[pair]
        h_tp, _, h_fp, _ = hyb[pair]
        tp_b += b_tp
        tp_h += h_tp
        fp_b += b_fp
        fp_h += h_fp
        if h_fp > b_fp:
            fp_regressed.append(pair)
        print(f"{pair[0]}->{pair[1]:<4} "
              f"{b_tp:>6}/{tp_tot:<5} {h_tp:>6}/{tp_tot:<5} {h_tp - b_tp:>+5}   "
              f"{b_fp:>4}/{fp_tot:<5} {h_fp:>4}/{fp_tot:<5} {h_fp - b_fp:>+5}")
    print(f"{'TOTAL':<8} {tp_b:>12} {tp_h:>12} {tp_h - tp_b:>+5}   "
          f"{fp_b:>10} {fp_h:>10} {fp_h - fp_b:>+5}")
    print(f"\nphrases : base {phrase_base}/{phrase_total}  "
          f"{label} {phrase_hyb}/{phrase_total}  ({phrase_hyb - phrase_base:+d})")

    ok = True
    if fp_regressed:
        print(f"REGRESSION: FP increased on pairs: "
              f"{', '.join(f'{a}->{b}' for a, b in fp_regressed)}")
        ok = False
    if tp_h < tp_b:
        print(f"REGRESSION: aggregate TP dropped ({tp_h} < {tp_b})")
        ok = False
    if phrase_hyb < phrase_base:
        print(f"REGRESSION: phrase recall dropped ({phrase_hyb} < {phrase_base})")
        ok = False
    return ok, (tp_h - tp_b, fp_h - fp_b, phrase_hyb - phrase_base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline vs hybrid comparison")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--lsh", action="store_true",
                        help="Enable the LSH component in the hybrid run")
    parser.add_argument("--fuzzy", action="store_true",
                        help="Enable the BK-tree component in the hybrid run "
                             "(off by default; costs TP on dense lexicons)")
    parser.add_argument("--ablation", action="store_true",
                        help="Additionally run per-component ablations")
    args = parser.parse_args()

    print("Loading ONNX model...")
    model_args = Languages.load_model()
    words_by_lang = {
        lang: ET.load_vocab(lang, args.sample, args.seed, args.min_len, args.max_len)
        for lang in LANGS
    }

    print("Running BASELINE (pure LSTM pipeline) ...")
    t0 = time.perf_counter()
    base = run_transitions(words_by_lang, model_args, triage=None)
    phrase_base, phrase_total = TP.run(True, model_args)
    base_s = time.perf_counter() - t0

    print("Building lexical triage (dictionaries"
          + (" / BK-trees" if args.fuzzy else "")
          + (" / LSH" if args.lsh else "") + ") ...")
    t0 = time.perf_counter()
    triage = LexicalTriage(enable_fuzzy=args.fuzzy, enable_lsh=args.lsh)
    build_s = time.perf_counter() - t0

    print("Running HYBRID (LSTM + lexical triage) ...")
    t0 = time.perf_counter()
    hyb = run_transitions(words_by_lang, model_args, triage=triage)
    phrase_hyb, _ = TP.run(True, model_args, triage=triage)
    hyb_s = time.perf_counter() - t0

    avg_us, p99_us = measure_triage_latency(triage, words_by_lang)

    print("\n" + "=" * 70)
    print(f"COMPARISON  (sample={args.sample}, seed={args.seed}, "
          f"fuzzy={args.fuzzy}, lsh={args.lsh})")
    print("=" * 70)
    ok, _ = print_comparison(base, hyb, phrase_base, phrase_hyb, phrase_total)

    print(f"\ntiming  : baseline harness {base_s:.1f}s | hybrid harness {hyb_s:.1f}s"
          f" | triage build {build_s:.1f}s")
    print(f"latency : triage evidence() avg {avg_us:.1f} us, p99 {p99_us:.1f} us"
          f" (cold cache; per-keystroke budget ~1000 us)")
    verdict = ("PASS — hybrid is not worse than baseline" if ok
               else "FAIL — hybrid regressed")
    print(f"\nVERDICT : {verdict}")

    if args.ablation:
        print("\n" + "=" * 70)
        print("ABLATIONS (each component alone; exact always includes prefix)")
        print("=" * 70)
        configs = {
            "exact-only":   dict(params=TriageParams(w_fuzzy=0, w_trigram=0, w_lsh=0),
                                 enable_fuzzy=False, enable_lsh=False),
            "exact+fuzzy":  dict(params=TriageParams(w_trigram=0, w_lsh=0),
                                 enable_fuzzy=True, enable_lsh=False),
            "exact+trigram": dict(params=TriageParams(w_fuzzy=0, w_lsh=0),
                                  enable_fuzzy=False, enable_lsh=False),
            "exact+lsh":    dict(params=TriageParams(w_fuzzy=0, w_trigram=0),
                                 enable_fuzzy=False, enable_lsh=True),
        }
        for name, cfg in configs.items():
            print(f"\n--- {name} ---")
            tri = LexicalTriage(**cfg)
            res = run_transitions(words_by_lang, model_args, triage=tri)
            ph, _ = TP.run(True, model_args, triage=tri)
            print_comparison(base, res, phrase_base, ph, phrase_total, label=name)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
