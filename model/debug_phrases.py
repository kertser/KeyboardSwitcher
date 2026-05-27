#!/usr/bin/env python3
"""Debug why specific phrases fail detection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Languages
from Languages import *

LAYOUT_PAIRS = [
    (english_layout, russian_layout), (russian_layout, english_layout),
    (hebrew_layout, english_layout),  (english_layout, hebrew_layout),
    (russian_layout, hebrew_layout),  (hebrew_layout, russian_layout),
]
model_args = Languages.load_model()

failing = [
    ("כך רציתי",   "he", "en",  "fl rmh,h"),
    ("תודה רבה",   "he", "en",  ",usv rcv"),
    ("אני לא יודע","he", "en",  "tbh kt husg"),
    ("עד מחר",     "he", "en",  "gs njr"),
    ("בסדר גמור",  "he", "en",  "cxsr dnur"),
    ("בוקר טוב",   "he", "ru",  "сгук нгс"),
]

def analyze(mistyped, current_lang, native):
    print(f"\n{'='*65}")
    print(f"  Native: {native!r}  ({current_lang} keyboard → he)")
    print(f"  Typed:  {mistyped!r}")
    print()

    # Show incremental detections
    for n in range(1, len(mistyped)+1):
        part = mistyped[:n]
        det_text = "".join(c for c in part if c.isalpha() or c==" ").strip()
        ac = sum(1 for c in part if c.isalpha())
        if ac < 3: continue

        seen=set(); variants=[]
        for src,dst in LAYOUT_PAIRS:
            v = convert_text_bidirectional(det_text,src,dst)
            if v not in seen: seen.add(v); variants.append(v)

        bl=None; bc=0.0; br=None
        for v in variants:
            r = predict_language_with_confidence(v, *model_args)
            if r and r.confidence > bc: bc=r.confidence; bl=r.language; br=r
        if not bl: continue

        scores = br.scores
        runner = max(((scores[i], ["","en","he","ru"][i]) for i in range(1,4) if ["","en","he","ru"][i]!=bl), default=(0,""))
        margin = bc - runner[0]

        # What does the he-variant look like?
        he_raw = convert_text_bidirectional(det_text, english_layout, hebrew_layout) if current_lang=="en" else convert_text_bidirectional(det_text, russian_layout, hebrew_layout)
        r_he = predict_language_with_confidence(he_raw, *model_args)
        he_conf = r_he.confidence if r_he else 0.0
        he_det = r_he.language if r_he else "None"

        he_short = he_raw[:20]
        print(f"  n={n:2d} α={ac:2d}  best={bl}({bc:.3f}) margin={margin:.3f}"
              f"  | he-variant={he_short!r:20s} -> {he_det}({he_conf:.3f})")

for native, nat_lang, kb, mistyped in failing:
    analyze(mistyped, kb, native)


