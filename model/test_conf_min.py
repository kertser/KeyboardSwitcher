#!/usr/bin/env python3
"""Test ConfAtMin tradeoff for Hebrew-targeted pairs."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Languages
from Languages import *

LAYOUT_PAIRS = [
    (english_layout, russian_layout), (russian_layout, english_layout),
    (hebrew_layout, english_layout),  (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),  (hebrew_layout, russian_layout),
]
model_args = Languages.load_model()
random.seed(42)

def load_v(lang, n=300):
    fmap = {"en":"vocabulary/english_vocabulary","ru":"vocabulary/russian_vocabulary","he":"vocabulary/hebrew_vocabulary"}
    words = []
    with open(os.path.join(os.path.dirname(__file__), fmap[lang]), encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if 2 <= len(w) <= 12 and w.isalpha():
                words.append(w)
    return random.sample(words, min(n, len(words)))

def sim(text, current, conf_min=0.99, conf_max=0.72, min_margin=0.10, early_min=3, agreement=2):
    def req(n):
        if n < early_min: return 1.1
        if n >= 15: return conf_max
        return conf_min + (n - early_min) / 12.0 * (conf_max - conf_min)
    ll = ""; streak = 0
    for n in range(1, len(text) + 1):
        part = text[:n]
        det = "".join(c for c in part if c.isalpha() or c == " ").strip()
        ac = sum(1 for c in part if c.isalpha())
        if ac < early_min or len(det) < early_min: continue
        seen = set(); variants = []
        for src, dst in LAYOUT_PAIRS:
            v = convert_text_bidirectional(det, src, dst)
            if v not in seen: seen.add(v); variants.append(v)
        bl = None; bc = 0.0; br = None
        for v in variants:
            r = predict_language_with_confidence(v, *model_args)
            if r and r.confidence > bc:
                bc = r.confidence; bl = r.language; br = r
        if not bl: ll = ""; streak = 0; continue
        if bl == ll: streak += 1
        else: ll = bl; streak = 1
        if streak < agreement: continue
        if bl == current: continue
        if br and min_margin > 0:
            scores = br.scores; cl = ["", "en", "he", "ru"]
            rc = max(scores[i] for i in range(1, 4) if cl[i] != bl)
            if bc - rc < min_margin: continue
        if bc >= req(ac): return bl, n
    return None, len(text)

words = {lang: load_v(lang) for lang in ["en", "ru", "he"]}
n_he = len(words["he"]); n_en = len(words["en"]); n_ru = len(words["ru"])

print("conf_min  recall_enhe  FP_enhe  recall_ruhe  FP_ruhe")
print("-" * 60)
for cm in [0.99, 0.97, 0.95, 0.93, 0.90]:
    r_enhe = fp_enhe = r_ruhe = fp_ruhe = 0
    for w in words["he"]:
        m1 = convert_text_bidirectional(w, hebrew_layout, english_layout)
        det, _ = sim(m1, "en", conf_min=cm)
        if det == "he": r_enhe += 1
        m2 = convert_text_bidirectional(w, hebrew_layout, russian_layout)
        det2, _ = sim(m2, "ru", conf_min=cm)
        if det2 == "he": r_ruhe += 1
    for w in words["en"]:
        det, _ = sim(w, "en", conf_min=cm)
        if det == "he": fp_enhe += 1
    for w in words["ru"]:
        det, _ = sim(w, "ru", conf_min=cm)
        if det == "he": fp_ruhe += 1
    print(f"{cm:.2f}      {r_enhe}/{n_he}={r_enhe/n_he:.3f}    {fp_enhe}/{n_en}={fp_enhe/n_en:.4f}  "
          f"{r_ruhe}/{n_he}={r_ruhe/n_he:.3f}    {fp_ruhe}/{n_ru}={fp_ruhe/n_ru:.4f}")

