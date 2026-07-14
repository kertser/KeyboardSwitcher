#!/usr/bin/env python3
"""ablate_stopwords.py — does the stop-word list matter for the consensus scorer?

If short-word length weighting alone (language-agnostic) matches the full
stop-word list, the C++ port can drop the error-prone Hebrew/Russian literals.
"""
from __future__ import annotations
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Languages, word_aware
import eval_word_aware as E
from Languages import predict_language_with_confidence, DetectionResult

ma = Languages.load_model()


def make_cons(stop_w, short_w):
    def sc(v):
        whole = predict_language_with_confidence(v, *ma)
        wa = word_aware.predict_word_aware(v, *ma, stop_w=stop_w, short_w=short_w)
        if whole is None and wa is None:
            return None
        ws = whole.scores if whole else [1.0, 0, 0, 0]
        as_ = wa.scores if wa else [1.0, 0, 0, 0]
        comb = [math.sqrt(max(ws[i], 0) * max(as_[i], 0)) for i in range(4)]
        z = sum(comb) or 1.0
        comb = [c / z for c in comb]
        bc, bp = 0, 0.0
        for i in range(1, 4):
            if comb[i] > bp:
                bp, bc = comb[i], i
        if comb[0] >= bp:
            return None
        return DetectionResult(language={1: "en", 2: "he", 3: "ru"}[bc],
                               confidence=bp, scores=comb)
    return sc


def recall(sc):
    h = t = 0
    for nat, lang in E.CURATED:
        for src in ("en", "ru", "he"):
            if src == lang:
                continue
            m = E.convert_text_bidirectional(nat, E.LAYOUTS[lang], E.LAYOUTS[src])
            t += 1
            if E.predict(m, src, ma, sc) == lang:
                h += 1
    return h, t


def main():
    configs = [
        ("stop=0.25 short=0.35 (full)", 0.25, 0.35),
        ("stop=1.0  short=0.35 (no stopwords)", 1.0, 0.35),
        ("stop=1.0  short=1.0  (plain mean)", 1.0, 1.0),
    ]
    for label, stop_w, short_w in configs:
        sc = make_cons(stop_w, short_w)
        h, t = recall(sc)
        fp, tf, _ = E.run_fp_with_scorer(sc, ma, random.Random(42), n_phrases=100)
        print(f"{label:38s}: recall {h}/{t}={h/t:.3f}  FP {fp}/{tf}={fp/tf:.4f}")


if __name__ == "__main__":
    main()

