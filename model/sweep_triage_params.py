#!/usr/bin/env python3
"""Sweep lexical-triage integration parameters on the exact-only config.

Phase-4 tuning helper: varies boost_scale / boost_threshold / suppress_extra
and reports TP/FP/phrase deltas vs the pure-LSTM baseline on the same sample.

Usage:  python sweep_triage_params.py --sample 200
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import Languages
import evaluate_transitions as ET
import test_phrases_lite as TP
from triage import LexicalTriage, TriageParams

LANGS = ["en", "ru", "he"]


def run_all(words, model_args, triage):
    tp = fp = 0
    per_pair = {}
    for f in LANGS:
        for t in LANGS:
            if f == t:
                continue
            r = ET.evaluate_pair(words, f, t, model_args, triage=triage)
            per_pair[(f, t)] = r
            tp += r[0]
            fp += r[2]
    ph, ph_total = TP.run(True, model_args, triage=triage)
    return tp, fp, ph, ph_total, per_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading ONNX model...")
    model_args = Languages.load_model()
    words = {l: ET.load_vocab(l, args.sample, args.seed, 2, 12) for l in LANGS}

    base_tp, base_fp, base_ph, ph_total, base_pairs = run_all(words, model_args, None)
    print(f"baseline: TP {base_tp}  FP {base_fp}  phrases {base_ph}/{ph_total}\n")

    # One shared exact-only triage; params object is swapped per config.
    triage = LexicalTriage(enable_fuzzy=False, enable_lsh=False)

    grid = []
    for boost_scale in (0.95, 0.90, 0.85, 0.80):
        for boost_threshold in (0.99, 0.90, 0.80):
            for suppress_extra in (0.04, 0.08, 0.15):
                grid.append(TriageParams(
                    w_fuzzy=0.0, w_trigram=0.0, w_lsh=0.0,
                    boost_scale=boost_scale,
                    boost_threshold=boost_threshold,
                    suppress_extra=suppress_extra))

    print(f"{'bScale':>6} {'bThr':>5} {'sExtra':>6} | "
          f"{'TP':>5} {'dTP':>5} {'FP':>3} {'dFP':>4} {'phr':>4} {'dPh':>4}")
    best = None
    for p in grid:
        triage.params = p
        tp, fp, ph, _, pairs = run_all(words, model_args, triage)
        fp_reg = any(pairs[k][2] > base_pairs[k][2] for k in pairs)
        flag = " FP-REG" if fp_reg else ""
        print(f"{p.boost_scale:>6.2f} {p.boost_threshold:>5.2f} "
              f"{p.suppress_extra:>6.2f} | {tp:>5} {tp - base_tp:>+5} "
              f"{fp:>3} {fp - base_fp:>+4} {ph:>4} {ph - base_ph:>+4}{flag}")
        score = (tp + 2 * ph, -fp)
        if not fp_reg and ph >= base_ph and (best is None or score > best[0]):
            best = (score, p)

    if best:
        p = best[1]
        print(f"\nbest (no FP regression, phrases kept): "
              f"boost_scale={p.boost_scale} boost_threshold={p.boost_threshold} "
              f"suppress_extra={p.suppress_extra}")


if __name__ == "__main__":
    main()
