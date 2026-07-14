#!/usr/bin/env python3
"""eval_consensus.py — whole-string vs word-aware vs consensus (geo-mean).

The consensus scorer multiplies the whole-string softmax by the word-aware
softmax element-wise and re-normalises (a geometric-mean-style AND): a class
scores high only when BOTH views agree, which suppresses false positives where
the two disagree while preserving recall where they agree.
"""
from __future__ import annotations

import os
import random
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import Languages
import eval_word_aware as E
from Languages import predict_language_with_confidence, DetectionResult
import word_aware


def make_consensus(model_args):
    def scorer(v):
        whole = predict_language_with_confidence(v, *model_args)
        wa = word_aware.predict_word_aware(v, *model_args)
        if whole is None and wa is None:
            return None
        ws = whole.scores if whole else [1.0, 0.0, 0.0, 0.0]
        as_ = wa.scores if wa else [1.0, 0.0, 0.0, 0.0]
        comb = [math.sqrt(max(ws[i], 0.0) * max(as_[i], 0.0)) for i in range(4)]
        z = sum(comb) or 1.0
        comb = [c / z for c in comb]
        best_c, best_p = 0, 0.0
        for i in range(1, 4):
            if comb[i] > best_p:
                best_p, best_c = comb[i], i
        if comb[0] >= best_p:
            return None
        return DetectionResult(language={1: "en", 2: "he", 3: "ru"}[best_c],
                               confidence=best_p, scores=comb)
    return scorer


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
    fp, total_fp, ex = E.run_fp_with_scorer(scorer, model_args, random.Random(42))
    return hits, total_r, fp, total_fp, ex


def main():
    print("Loading ONNX model...")
    model_args = Languages.load_model()

    scorers = {
        "whole":     lambda v: predict_language_with_confidence(v, *model_args),
        "word":      lambda v: word_aware.predict_word_aware(v, *model_args),
        "consensus": make_consensus(model_args),
    }
    for name, sc in scorers.items():
        h, tr, fp, tfp, ex = run_all(sc, model_args)
        print(f"  {name:10s}: recall {h:>3}/{tr}={h/tr:.3f}   FP {fp:>2}/{tfp}={fp/tfp:.4f}")
        if name == "consensus" and ex:
            print("     FP examples:", ", ".join(f"{e[0]}[{e[1]}->{e[2]}]" for e in ex[:8]))


if __name__ == "__main__":
    main()

