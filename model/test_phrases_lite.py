#!/usr/bin/env python3
"""Lightweight phrase-detection validator for the ported 1.3.0 signal-quality gates.

Tests curated Hebrew/Russian phrases typed on the wrong keyboard layout and
reports how many are correctly detected — WITH the gates (PhraseConfScale +
Hebrew script gate + margin) vs WITHOUT them.  This isolates the value of the
gates, which only manifests on multi-word phrases (the single-word
evaluate_transitions harness cannot show it).

Usage:  py test_phrases_lite.py
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import Languages
from Languages import (
    english_layout, russian_layout, hebrew_layout,
    convert_text_bidirectional, predict_language_with_confidence,
)

LAYOUTS = {"en": english_layout, "ru": russian_layout, "he": hebrew_layout}
_CLASS_LANG = {1: "en", 2: "he", 3: "ru"}

# Field order matches cpp/src/Config.cpp SwitchingParams (11):
#   EMin FConf CAt0 CAt1 Agr BLF  Mrg  SXC  PCS  HeVC HeCT
PAIR_OVERRIDES = {
    ("en", "ru"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    ("ru", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    ("en", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90),
    ("he", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90),
    ("ru", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90),
    ("he", "ru"): (4, 15, 0.99, 0.70, 2, 0.80, 0.05, 0.02, 1.00, 0.00, 0.90),
}
DEFAULT_PARAMS = (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90)

CURATED = [
    ("אני רוצה", "he"), ("כך רציתי", "he"), ("שלום לכולם", "he"),
    ("תודה רבה", "he"), ("מה שלומך", "he"), ("אני לא יודע", "he"),
    ("בסדר גמור", "he"), ("עד מחר", "he"), ("לילה טוב", "he"),
    ("בוקר טוב", "he"), ("אני עייף", "he"), ("יש לי שאלה", "he"),
    ("איפה אתה", "he"), ("מתי נפגש", "he"), ("אני מסכים", "he"),
    ("זה לא נכון", "he"), ("כמה זה עולה", "he"), ("אני אוהב אותך", "he"),
    ("שנה טובה", "he"), ("חג שמח", "he"), ("ברוך הבא", "he"),
    ("תכתוב לי", "he"), ("אני חוזר עוד רגע", "he"), ("כן בטח", "he"),
    ("לא עכשיו", "he"), ("קח את הזמן", "he"), ("הכל בסדר", "he"),
    ("נשמע טוב", "he"), ("אנחנו מחכים", "he"), ("תהיה בריא", "he"),
    ("אני צריך עזרה", "he"), ("אתה מבין", "he"), ("בוא נדבר", "he"),
    ("זה מאוד חשוב", "he"), ("תודה לך", "he"), ("מה קורה", "he"),
    ("הכל טוב", "he"), ("אני בדרך", "he"), ("עוד מעט", "he"),
    ("קצת יותר מאוחר", "he"),
    ("привет как дела", "ru"), ("я не знаю", "ru"), ("спасибо большое", "ru"),
    ("до свидания", "ru"), ("хорошо понял", "ru"), ("я согласен", "ru"),
    ("не могу сейчас", "ru"), ("позвони мне", "ru"), ("всё в порядке", "ru"),
    ("увидимся завтра", "ru"), ("мне нужна помощь", "ru"), ("что случилось", "ru"),
]


def hebrew_script_coverage(text: str) -> float:
    alpha = hebrew = 0
    for c in text:
        if c == " ":
            continue
        if c.isalpha():
            alpha += 1
            if "\u05d0" <= c <= "\u05ea":
                hebrew += 1
    return (hebrew / alpha) if alpha else 0.0


def required_confidence(n, params, is_phrase, gates_on):
    early_min, full_conf, conf_min, conf_max = params[0], params[1], params[2], params[3]
    phrase_scale = params[8] if gates_on else 1.00
    if n < early_min:
        return 1.1
    if n >= full_conf:
        req = conf_max
    else:
        t = (n - early_min) / (full_conf - early_min)
        req = conf_min + t * (conf_max - conf_min)
    if is_phrase and 0.0 < phrase_scale < 1.0:
        req *= phrase_scale
    return req


def predict(text, current_lang, model_args, gates_on):
    params = PAIR_OVERRIDES.get((current_lang, "he"), DEFAULT_PARAMS)
    he_vc = params[9] if gates_on else 0.0
    he_ct = params[10]
    last_lang, streak = "", 0
    cache = {}

    for n in range(1, len(text) + 1):
        variants, seen = [text[:n]], {text[:n]}
        cur_layout = LAYOUTS[current_lang]
        for other, dst in LAYOUTS.items():
            if other == current_lang:
                continue
            v = convert_text_bidirectional(text[:n], cur_layout, dst)
            if v not in seen:
                seen.add(v)
                variants.append(v)

        is_phrase = any(" " in v for v in variants)
        best_lang, best_conf, best_scores = None, 0.0, [0.0, 0.0, 0.0, 0.0]
        he_script_vc = 0.0
        for v in variants:
            if v in cache:
                lang, conf, scores = cache[v]
            else:
                r = predict_language_with_confidence(v, *model_args)
                lang, conf, scores = (None, 0.0, [0, 0, 0, 0]) if r is None \
                    else (r.language, r.confidence, r.scores)
                cache[v] = (lang, conf, scores)
            if he_vc > 0.0 and current_lang != "he":
                cov = hebrew_script_coverage(v)
                if cov >= he_ct and not (lang and lang != "he" and conf > 0.80):
                    he_script_vc = max(he_script_vc, cov * he_vc)
            if conf > best_conf:
                best_lang, best_conf, best_scores = lang, conf, scores

        script_fired = False
        if he_script_vc > best_conf:
            best_lang, best_conf, best_scores = "he", he_script_vc, [0, 0, 0, 0]
            script_fired = True
        if not best_lang:
            continue

        det_params = PAIR_OVERRIDES.get((current_lang, best_lang), DEFAULT_PARAMS)
        margin_min = det_params[6] if gates_on else 0.0
        runner = max((best_scores[i] for i in (1, 2, 3)
                      if _CLASS_LANG[i] != best_lang), default=0.0)
        margin = best_conf - runner

        required = required_confidence(n, det_params, is_phrase, gates_on)
        if required > 1.0:
            continue

        if best_lang == last_lang:
            streak += 1
        else:
            last_lang, streak = best_lang, 1

        if (not script_fired) and margin_min > 0.0 and margin < margin_min:
            continue
        if streak >= det_params[4] and best_lang != current_lang and best_conf >= required:
            return best_lang
    return None


def run(gates_on, model_args):
    hits = 0
    total = 0
    for native, lang in CURATED:
        # Type the phrase on each *other* layout (mistype), expect detection==lang
        for src in ("en", "ru", "he"):
            if src == lang:
                continue
            mistyped = convert_text_bidirectional(native, LAYOUTS[lang], LAYOUTS[src])
            total += 1
            if predict(mistyped, src, model_args, gates_on) == lang:
                hits += 1
    return hits, total


def main():
    print("Loading ONNX model...")
    model_args = Languages.load_model()
    off_hits, total = run(False, model_args)
    on_hits, _ = run(True, model_args)
    print(f"\nCurated phrase detection ({len(CURATED)} phrases x 2 wrong layouts = {total} cases):")
    print(f"  WITHOUT gates : {off_hits}/{total} ({off_hits/total:.3f})")
    print(f"  WITH    gates : {on_hits}/{total} ({on_hits/total:.3f})")
    print(f"  delta         : {on_hits - off_hits:+d} phrases")


if __name__ == "__main__":
    main()

