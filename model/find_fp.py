#!/usr/bin/env python3
"""Find false-positive cases for Hebrew-targeted pairs."""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
import Languages
from Languages import *

LAYOUT_PAIRS = [
    (english_layout, russian_layout), (russian_layout, english_layout),
    (hebrew_layout, english_layout),  (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),  (hebrew_layout, russian_layout),
]

model_args = Languages.load_model()
random.seed(42)

def load_vocab(lang, n=300):
    fmap = {"en": "english_vocabulary", "ru": "russian_vocabulary", "he": "hebrew_vocabulary"}
    words = []
    with open(os.path.join(os.path.dirname(__file__), "vocabulary", fmap[lang]), encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if 2 <= len(w) <= 12 and w.isalpha():
                words.append(w)
    return random.sample(words, min(n, len(words)))

def simulate(text, current, early_min=3, conf_min=0.99, conf_max=0.72,
             agreement_count=2, min_margin=0.10, full_conf=15):
    def req(n):
        if n < early_min: return 1.1
        if n >= full_conf: return conf_max
        return conf_min + (n - early_min) / (full_conf - early_min) * (conf_max - conf_min)

    last_lang = ""; streak = 0
    for n in range(1, len(text) + 1):
        part = text[:n]
        det = "".join(c for c in part if c.isalpha() or c == " ").strip()
        ac = sum(1 for c in part if c.isalpha())
        if ac < early_min or len(det) < early_min:
            continue

        seen = set(); variants = []
        for src, dst in LAYOUT_PAIRS:
            v = convert_text_bidirectional(det, src, dst)
            if v not in seen:
                seen.add(v); variants.append(v)

        bl = None; bc = 0.0; br = None
        for v in variants:
            r = predict_language_with_confidence(v, *model_args)
            if r and r.confidence > bc:
                bc = r.confidence; bl = r.language; br = r
        if not bl:
            last_lang = ""; streak = 0; continue

        if bl == last_lang:
            streak += 1
        else:
            last_lang = bl; streak = 1

        if streak < agreement_count: continue
        if bl == current: continue

        if br and min_margin > 0:
            scores = br.scores
            cl = ["", "en", "he", "ru"]
            rc = max((scores[i] for i in range(1, 4) if cl[i] != bl), default=0.0)
            if bc - rc < min_margin: continue

        if bc >= req(ac):
            return bl, n
    return None, len(text)

print("=== FALSE POSITIVES en->he ===")
for w in load_vocab("en"):
    det, ch = simulate(w, "en")
    if det == "he":
        print(f"  {w!r:20} chars_at_det={ch}")

print("\n=== FALSE POSITIVES ru->he ===")
for w in load_vocab("ru"):
    det, ch = simulate(w, "ru")
    if det == "he":
        print(f"  {w!r:20} chars_at_det={ch}")

print("\n=== FALSE POSITIVES en->ru ===")
for w in load_vocab("en"):
    det, ch = simulate(w, "en", conf_max=0.70, min_margin=0.05)
    if det == "ru":
        print(f"  {w!r:20} chars_at_det={ch}")

print("\n=== FALSE POSITIVES ru->en ===")
for w in load_vocab("ru"):
    det, ch = simulate(w, "ru", conf_max=0.70, min_margin=0.05)
    if det == "en":
        print(f"  {w!r:20} chars_at_det={ch}")

