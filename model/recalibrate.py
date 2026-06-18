#!/usr/bin/env python3
"""recalibrate.py — re-derive per-pair detection thresholds for the v2 model.

After retraining (train_model.py) the model's confidence behaviour changes, so
the hand-tuned PairOverrides in cpp/src/Config.cpp must be re-derived on a LARGE
vocabulary sample.  This script sweeps, per directed pair, the parameters that
shape the confidence gate and prints:
    • a TP/FP table per candidate,
    • the best FP-bounded setting,
    • ready-to-paste C++ SwitchingParams lines (Config.cpp) and the matching
      Python tuples (evaluate_transitions.py / test_phrases_lite.py).

It reuses evaluate_transitions.evaluate_pair so the detection logic under test
is exactly the production mirror (incumbent gate, margin gate, weak-signal
gates, etc.).  Whatever lang_model.onnx + dictionary.json are present are used,
so run it AFTER copying the retrained model into model/.

Usage:
    python recalibrate.py --sample 1500 --fp-cap 0.005
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import Languages
import evaluate_transitions as ET

PAIRS = [("en", "ru"), ("ru", "en"), ("en", "he"),
         ("he", "en"), ("ru", "he"), ("he", "ru")]

# Field indices into the 17-field params tuple (see Config.cpp / evaluate_transitions)
I_EMIN, I_CX, I_MRG = 0, 3, 6

# Search grid (kept modest so the full run is tractable on a large sample).
GRID_EMIN = [2, 3, 4]
GRID_CX = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GRID_MRG_HE = [0.05, 0.10, 0.15]     # →he pairs: HE/EN softmax compete
GRID_MRG_ROBUST = [0.0, 0.05]        # en↔ru / he→en / he→ru


def with_pair(base_params, emin, cx, mrg):
    p = list(base_params)
    p[I_EMIN] = emin
    p[I_CX] = cx
    p[I_MRG] = mrg
    return tuple(p)


def gather_variant_strings(fr, to, words):
    """Reproduce every string the harness will query for this directed pair so
    they can be batch-inferred (prewarmed) in one shot.  Mirrors the variant
    generation inside evaluate_transitions.predict_best_lang: for each prefix of
    each (mistyped TP word / native FP word) typed on `fr`, the identity text
    plus the `fr`-layout → every-other-layout conversions."""
    cur_layout = ET.LAYOUTS[fr]
    out = []

    def emit_prefixes(text):
        for n in range(1, len(text) + 1):
            if n < ET.GLOBAL_MIN:
                continue
            prefix = text[:n]
            out.append(prefix)
            for other_lang, dst_layout in ET.LAYOUTS.items():
                if other_lang == fr:
                    continue
                out.append(ET.convert_text_bidirectional(prefix, cur_layout, dst_layout))

    # TP words: target-language words typed on the wrong (`fr`) layout.
    for w in words[to]:
        emit_prefixes(ET.convert_text_bidirectional(w, ET.LAYOUTS[to], ET.LAYOUTS[fr]))
    # FP words: source-language words typed correctly on `fr`.
    for w in words[fr]:
        emit_prefixes(w)
    return out


def prewarm_all(words, model_args):
    """Batch-infer every variant string across all pairs up front, so the grid
    sweep that follows runs entirely against the in-process prediction cache."""
    strings = []
    for fr, to in PAIRS:
        strings.extend(gather_variant_strings(fr, to, words))
    n = Languages.prewarm_cache(strings, *model_args)
    print(f"Prewarmed {n} unique variant strings "
          f"(of {len(strings)} requested) into the inference cache.")


def sweep_pair(fr, to, words, model_args, fp_cap, fp_total):
    base = ET.PAIR_OVERRIDES[(fr, to)]
    margins = GRID_MRG_HE if to == "he" else GRID_MRG_ROBUST
    results = []
    for emin in GRID_EMIN:
        for cx in GRID_CX:
            for mrg in margins:
                ET.PAIR_OVERRIDES[(fr, to)] = with_pair(base, emin, cx, mrg)
                tp, tp_n, fp, fp_n = ET.evaluate_pair(words, fr, to, model_args)
                results.append((emin, cx, mrg, tp, tp_n, fp, fp_n))
    ET.PAIR_OVERRIDES[(fr, to)] = base  # restore

    # Choose: max TP subject to fp_rate <= fp_cap; tie-break smaller EarlyMin,
    # then higher ConfAtMax (more conservative), then higher margin.
    fp_allow = max(0, int(round(fp_cap * fp_n)))
    feasible = [r for r in results if r[5] <= fp_allow]
    pool = feasible if feasible else results
    best = max(pool, key=lambda r: (r[3], -r[0], r[1], r[2]))
    return best, results, fp_allow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500,
                    help="words per language (large sample for stable calibration)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=14)
    ap.add_argument("--fp-cap", type=float, default=0.005,
                    help="max tolerated false-positive rate per pair")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("Loading ONNX model ...")
    model_args = Languages.load_model()
    words = {l: ET.load_vocab(l, args.sample, args.seed, args.min_len, args.max_len)
             for l in ET.LANGS}
    fp_total = args.sample

    print(f"\nRecalibrating on {args.sample} words/lang, FP cap {args.fp_cap:.1%}\n")
    prewarm_all(words, model_args)
    print()
    chosen = {}
    for fr, to in PAIRS:
        best, results, fp_allow = sweep_pair(fr, to, words, model_args,
                                             args.fp_cap, fp_total)
        emin, cx, mrg, tp, tp_n, fp, fp_n = best
        chosen[(fr, to)] = (emin, cx, mrg)
        print(f"{fr}->{to}: EarlyMin={emin} ConfAtMax={cx:.2f} Margin={mrg:.2f}"
              f"  -> TP {tp}/{tp_n} ({tp/tp_n:.3f})  FP {fp}/{fp_n} ({fp/fp_n:.3f})"
              f"   [fp_allow={fp_allow}]")
        if args.verbose:
            for r in sorted(results, key=lambda x: -x[3])[:8]:
                print(f"      EMin={r[0]} CX={r[1]:.2f} M={r[2]:.2f}"
                      f"  TP {r[3]}/{r[4]}  FP {r[5]}/{r[6]}")

    # Emit ready-to-paste config.
    print("\n" + "=" * 70)
    print("  C++  (cpp/src/Config.cpp — update EMin / ConfAtMax / Margin fields)")
    print("=" * 70)
    for fr, to in PAIRS:
        emin, cx, mrg = chosen[(fr, to)]
        base = ET.PAIR_OVERRIDES[(fr, to)]
        # Show the full 17-field line with recalibrated EMin/CX/Margin spliced in.
        p = list(base); p[I_EMIN], p[I_CX], p[I_MRG] = emin, cx, mrg
        fields = ", ".join(
            (f"{v:.2f}f" if isinstance(v, float) else str(v)) for v in p)
        print(f'    {{ {{"{fr}", "{to}"}}, {{ {fields} }} }},')

    print("\n" + "=" * 70)
    print("  Python harness tuples (evaluate_transitions.py / test_phrases_lite.py)")
    print("=" * 70)
    for fr, to in PAIRS:
        emin, cx, mrg = chosen[(fr, to)]
        base = list(ET.PAIR_OVERRIDES[(fr, to)])
        base[I_EMIN], base[I_CX], base[I_MRG] = emin, cx, mrg
        print(f'    ("{fr}", "{to}"): {tuple(base)},')


if __name__ == "__main__":
    main()

