#!/usr/bin/env python3
"""Quick sweep of ConfidenceAtMaxChars for Hebrew-target pairs.

Finds the optimal floor confidence that maximises TP rate while keeping FP = 0.
"""
from __future__ import annotations
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Languages
from Languages import (
    english_layout, russian_layout, hebrew_layout,
    convert_text_bidirectional, predict_language_with_confidence,
)

LAYOUTS = {"en": english_layout, "ru": russian_layout, "he": hebrew_layout}
LAYOUT_PAIRS = [
    (english_layout, russian_layout), (russian_layout, english_layout),
    (hebrew_layout, english_layout),  (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),  (hebrew_layout, russian_layout),
]


def load_vocab(lang: str, sample: int = 200, seed: int = 42,
               min_len: int = 3, max_len: int = 14) -> list[str]:
    name = {"en": "english", "ru": "russian", "he": "hebrew"}[lang]
    path = os.path.join(os.path.dirname(__file__), "vocabulary", f"{name}_vocabulary")
    words = [w.strip() for w in open(path, encoding="utf-8")
             if min_len <= len(w.strip()) <= max_len and w.strip().isalpha()]
    return random.Random(seed).sample(words, min(sample, len(words)))


def req_conf(n: int, em: int, fc: int, cm: float, cx: float) -> float:
    if n < em:
        return 1.1
    if n >= fc:
        return cx
    t = (n - em) / (fc - em)
    return cm + t * (cx - cm)


def eval_pair(words_by_lang: dict, from_lang: str, to_lang: str,
              em: int, fc: int, cm: float, cx: float, ag: int,
              model_args: list) -> tuple[int, int]:
    cache: dict = {}

    def run(text: str, cur: str) -> str | None:
        last, streak = "", 0
        for n in range(1, len(text) + 1):
            r = req_conf(n, em, fc, cm, cx)
            if r > 1.0:
                continue
            variants = list(dict.fromkeys(
                convert_text_bidirectional(text[:n], s, d) for s, d in LAYOUT_PAIRS
            ))
            bl, bc = None, 0.0
            for v in variants:
                if v not in cache:
                    res = predict_language_with_confidence(v, *model_args)
                    cache[v] = (res.language, res.confidence) if res else (None, 0.0)
                lang_v, conf_v = cache[v]
                if conf_v > bc:
                    bl, bc = lang_v, conf_v
            if not bl:
                last, streak = "", 0
                continue
            streak = (streak + 1) if bl == last else 1
            last = bl
            if streak >= ag and bl != cur and bc >= r:
                return bl
        return None

    n = len(words_by_lang[to_lang])
    tp = sum(
        1 for w in words_by_lang[to_lang]
        if run(convert_text_bidirectional(w, LAYOUTS[to_lang], LAYOUTS[from_lang]), from_lang) == to_lang
    )
    fp = sum(1 for w in words_by_lang[from_lang] if run(w, from_lang) == to_lang)
    return tp, fp


def main() -> None:
    print("Loading ONNX model...")
    model_args = Languages.load_model()
    words = {lang: load_vocab(lang) for lang in ["en", "ru", "he"]}
    total = len(words["en"])

    # ── Sweep ConfAtMax for →he pairs ────────────────────────────────
    print()
    print("Sweeping ConfAtMax for →he pairs  (EarlyMin=4, FullConf=15, ConfAtMin=0.99, Agreement=2)")
    print(f"  {'ConfAtMax':>9} | {'en→he TP':>8} | {'ru→he TP':>8} | {'en→he FP':>8} | {'ru→he FP':>8}")
    print(f"  {'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for cx in [0.75, 0.72, 0.70, 0.68, 0.65, 0.62, 0.60, 0.55]:
        tp_en, fp_en = eval_pair(words, "en", "he", 4, 15, 0.99, cx, 2, model_args)
        tp_ru, fp_ru = eval_pair(words, "ru", "he", 4, 15, 0.99, cx, 2, model_args)
        marker = " ←" if fp_en == 0 and fp_ru == 0 else "  FP!"
        print(f"  {cx:>9.2f} | {tp_en:>4}/{total}   | {tp_ru:>4}/{total}   | {fp_en:>4}/{total}   | {fp_ru:>4}/{total}{marker}")

    # ── Sweep ConfAtMax for en↔ru / he↔ru ────────────────────────────
    print()
    print("Sweeping ConfAtMax for en↔ru / he→ru  (EarlyMin=4, FullConf=15, ConfAtMin=0.99, Agreement=2)")
    print(f"  {'ConfAtMax':>9} | {'en→ru TP':>8} | {'ru→en TP':>8} | {'he→ru TP':>8} | {'he→en TP':>8} | any FP?")
    print(f"  {'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+--------")
    for cx in [0.70, 0.68, 0.65, 0.62, 0.60, 0.55]:
        tp_er, fp_er = eval_pair(words, "en", "ru", 4, 15, 0.99, cx, 2, model_args)
        tp_re, fp_re = eval_pair(words, "ru", "en", 4, 15, 0.99, cx, 2, model_args)
        tp_hr, fp_hr = eval_pair(words, "he", "ru", 4, 15, 0.99, cx, 2, model_args)
        tp_he, fp_he = eval_pair(words, "he", "en", 4, 15, 0.99, cx, 2, model_args)
        any_fp = fp_er + fp_re + fp_hr + fp_he
        marker = "  FP!" if any_fp else " ←"
        print(f"  {cx:>9.2f} | {tp_er:>4}/{total}   | {tp_re:>4}/{total}   | {tp_hr:>4}/{total}   | {tp_he:>4}/{total}  | {any_fp}{marker}")


if __name__ == "__main__":
    main()

