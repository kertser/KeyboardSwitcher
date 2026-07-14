#!/usr/bin/env python3
"""sweep_word_aware.py — grid search for word-aware aggregation parameters.

Reuses the firing simulation and datasets from eval_word_aware.py.  Prints a
table of (recall, FP) for each config, plus the whole-string baseline.  Goal:
find a config that raises recall while holding FP <= baseline.
"""
from __future__ import annotations

import os
import random
import sys
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import Languages
import eval_word_aware as E
from Languages import predict_language_with_confidence
import word_aware


def run_all(scorer, model_args):
    hits = total_r = 0
    for native, lang in E.CURATED:
        for src in ("en", "ru", "he"):
            if src == lang:
                continue
            mistyped = E.convert_text_bidirectional(
                native, E.LAYOUTS[lang], E.LAYOUTS[src])
            total_r += 1
            if E.predict(mistyped, src, model_args, scorer) == lang:
                hits += 1
    fp, total_fp, _ = E.run_fp_with_scorer(scorer, model_args, random.Random(42))
    return hits, total_r, fp, total_fp


def main():
    print("Loading ONNX model...")
    model_args = Languages.load_model()

    # Baseline: whole-string.
    whole = lambda v: predict_language_with_confidence(v, *model_args)
    h, tr, fp, tfp = run_all(whole, model_args)
    print(f"\nBASELINE whole : recall {h}/{tr}={h/tr:.3f}  FP {fp}/{tfp}={fp/tfp:.4f}\n")

    grid = {
        "conf_floor": [0.0, 0.5, 0.6, 0.7],
        "sharpen":    [1.0, 1.5, 2.0],
        "short_w":    [0.35],
        "stop_w":     [0.25],
    }
    keys = list(grid)
    best = None
    print(f"{'conf_floor':>10} {'sharpen':>8} {'recall':>14} {'FP':>14}")
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        scorer = (lambda cfg: lambda v: word_aware.predict_word_aware(
            v, *model_args, short_w=cfg["short_w"], stop_w=cfg["stop_w"],
            conf_floor=cfg["conf_floor"], sharpen=cfg["sharpen"]))(cfg)
        h, tr, fp, tfp = run_all(scorer, model_args)
        rec, fpr = h / tr, fp / tfp
        print(f"{cfg['conf_floor']:>10.2f} {cfg['sharpen']:>8.1f} "
              f"{h:>4}/{tr} ={rec:.3f} {fp:>6}/{tfp} ={fpr:.4f}")
        score = (rec, -fpr)
        if best is None or score > best[0]:
            best = (score, cfg, (h, tr, fp, tfp))
    print("\nBEST:", best[1], "->", best[2])


if __name__ == "__main__":
    main()

