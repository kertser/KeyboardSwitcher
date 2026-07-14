#!/usr/bin/env python3
"""eval_word_aware.py — does word-aware aggregation beat whole-string scoring?

Compares two per-variant scorers inside an identical firing simulation:

  * WHOLE : predict_language_with_confidence  (current production behaviour)
  * WORD  : predict_word_aware                (this change)

Metrics
-------
  RECALL : curated Hebrew/Russian phrases typed on the wrong layout should be
           detected as their true language (higher = better).
  FP     : genuine phrases/words typed on the CORRECT layout must NOT trigger a
           switch away (lower = better).

The simulation mirrors the essential C++ gates: incremental keystrokes,
consecutive-agreement, adaptive confidence (with PhraseConfScale), margin gate,
and switch-bias.  It is intentionally the same skeleton as test_phrases_lite.py.
"""
from __future__ import annotations

import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import Languages
from Languages import (
    english_layout, russian_layout, hebrew_layout,
    convert_text_bidirectional, predict_language_with_confidence,
)
from word_aware import predict_word_aware

LAYOUTS = {"en": english_layout, "ru": russian_layout, "he": hebrew_layout}
_CLASS_LANG = {1: "en", 2: "he", 3: "ru"}

# Field order matches cpp/src/Config.cpp SwitchingParams (17)
PAIR_OVERRIDES = {
    ("en", "ru"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("ru", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("en", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90, 0.02, 0.55, 6, 2, 0.40, 7),
    ("he", "en"): (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
    ("ru", "he"): (3, 15, 0.99, 0.60, 2, 0.88, 0.10, 0.00, 0.80, 0.78, 0.90, 0.02, 0.55, 6, 2, 0.40, 7),
    ("he", "ru"): (4, 15, 0.99, 0.70, 2, 0.80, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7),
}
DEFAULT_PARAMS = (4, 15, 0.99, 0.70, 2, 0.85, 0.05, 0.02, 1.00, 0.00, 0.90, 0.00, 0.55, 6, 2, 0.40, 7)

# Curated phrases (native rendering, true language) — same set as test_phrases_lite.
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
    ("i have a pen", "en"), ("this is a test", "en"), ("see you later", "en"),
    ("what are you doing", "en"), ("i am on my way", "en"), ("it is a good day", "en"),
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


def required_confidence(n, params, is_phrase):
    early_min, full_conf, conf_min, conf_max = params[0], params[1], params[2], params[3]
    phrase_scale = params[8]
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


def predict(text, current_lang, model_args, scorer):
    """Incremental firing simulation.  Returns the language switched to, or None.

    scorer(variant) -> (lang, conf, scores)
    """
    params = PAIR_OVERRIDES.get((current_lang, "he"), DEFAULT_PARAMS)
    he_vc = params[9]
    he_ct = params[10]
    last_lang, streak = "", 0
    cache = {}
    frames = []
    cur_idx = next((i for i in (1, 2, 3) if _CLASS_LANG[i] == current_lang), 0)

    def update(lang, conf, scores):
        nonlocal last_lang, streak
        if lang == last_lang:
            streak += 1
        else:
            last_lang, streak = lang, 1
        frames.append((lang, conf, list(scores)))
        if len(frames) > 10:
            frames.pop(0)

    def is_persistent(lang, min_steps, min_avg):
        if min_steps <= 0 or len(frames) < min_steps:
            return False
        s = 0.0
        for f in frames[-min_steps:]:
            if f[0] != lang:
                return False
            s += f[1]
        return (s / min_steps) >= min_avg

    def weak_score_avg(cls_idx, window):
        if window <= 0 or len(frames) < window:
            return 0.0
        return sum(f[2][cls_idx] for f in frames[-window:]) / window

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
        incumbent_conf = 0.0
        he_script_vc = 0.0
        for v in variants:
            if v in cache:
                lang, conf, scores = cache[v]
            else:
                r = scorer(v)
                lang, conf, scores = (None, 0.0, [0, 0, 0, 0]) if r is None \
                    else (r.language, r.confidence, r.scores)
                cache[v] = (lang, conf, scores)
            if he_vc > 0.0 and current_lang != "he":
                cov = hebrew_script_coverage(v)
                if cov >= he_ct and not (lang and lang != "he" and conf > 0.80):
                    he_script_vc = max(he_script_vc, cov * he_vc)
            if lang is not None:
                if cur_idx and scores[cur_idx] > incumbent_conf:
                    incumbent_conf = scores[cur_idx]
                if conf > best_conf:
                    best_lang, best_conf, best_scores = lang, conf, scores

        script_fired = False
        if he_script_vc > best_conf:
            best_lang, best_conf, best_scores = "he", he_script_vc, [0, 0, 0, 0]
            script_fired = True
        if not best_lang:
            update("", 0.0, [0, 0, 0, 0])
            continue

        det_params = PAIR_OVERRIDES.get((current_lang, best_lang), DEFAULT_PARAMS)
        margin_min  = det_params[6]
        switch_bias = det_params[11]
        p_min_avg   = det_params[12]
        p_min_steps = det_params[13]
        ws_idx      = det_params[14]
        ws_min      = det_params[15]
        ws_win      = det_params[16]
        runner = max((best_scores[i] for i in (1, 2, 3)
                      if _CLASS_LANG[i] != best_lang), default=0.0)
        margin = best_conf - runner

        required = required_confidence(n, det_params, is_phrase)
        if required > 1.0:
            continue

        no_scores = all(s == 0.0 for s in best_scores)
        update(best_lang, best_conf, [0, 0, 0, 0] if no_scores else best_scores)

        if (best_lang != current_lang and switch_bias > 0.0
                and best_conf < incumbent_conf + switch_bias):
            continue
        if (not script_fired) and margin_min > 0.0 and margin < margin_min:
            continue

        fire = (last_lang == best_lang and streak >= det_params[4]
                and best_conf >= required)
        if (not fire) and best_lang == "he":
            if is_persistent("he", p_min_steps, p_min_avg):
                fire = True
            elif weak_score_avg(ws_idx, ws_win) >= ws_min:
                fire = True
        if fire and best_lang != current_lang:
            return best_lang
    return None


def make_scorer(kind, model_args):
    if kind == "whole":
        return lambda v: predict_language_with_confidence(v, *model_args)
    return lambda v: predict_word_aware(v, *model_args)


def run_recall(kind, model_args):
    scorer = make_scorer(kind, model_args)
    hits = total = 0
    misses = []
    for native, lang in CURATED:
        for src in ("en", "ru", "he"):
            if src == lang:
                continue
            mistyped = convert_text_bidirectional(native, LAYOUTS[lang], LAYOUTS[src])
            total += 1
            if predict(mistyped, src, model_args, scorer) == lang:
                hits += 1
            else:
                misses.append((native, lang, src))
    return hits, total, misses


def load_words(lang, sample, rng, min_len=3, max_len=14):
    name = {"en": "english", "ru": "russian", "he": "hebrew"}[lang]
    path = os.path.join(SCRIPT_DIR, "vocabulary", f"{name}_vocabulary")
    words = [w.strip() for w in open(path, encoding="utf-8")
             if min_len <= len(w.strip()) <= max_len and w.strip().isalpha()]
    return rng.sample(words, min(sample, len(words)))


def run_fp(kind, model_args, rng, n_phrases=200):
    """Genuine multi-word text typed on the CORRECT layout must NOT switch away."""
    scorer = make_scorer(kind, model_args)
    return run_fp_with_scorer(scorer, model_args, rng, n_phrases)


def run_fp_with_scorer(scorer, model_args, rng, n_phrases=200):
    fp = total = 0
    examples = []
    # Native curated phrases on their own layout.
    corpus = [(native, lang) for native, lang in CURATED]
    # Random 2-4 word phrases per language from the vocabulary.
    for lang in ("en", "ru", "he"):
        pool = load_words(lang, 600, rng)
        for _ in range(n_phrases):
            k = rng.choice([2, 3, 4])
            corpus.append((" ".join(rng.sample(pool, k)), lang))
    for phrase, lang in corpus:
        total += 1
        out = predict(phrase, lang, model_args, scorer)
        if out is not None and out != lang:
            fp += 1
            examples.append((phrase, lang, out))
    return fp, total, examples


def main():
    print("Loading ONNX model...")
    model_args = Languages.load_model()
    rng = random.Random(42)

    print("\n=== RECALL (mistyped phrases -> detect true language) ===")
    for kind in ("whole", "word"):
        hits, total, misses = run_recall(kind, model_args)
        print(f"  {kind:5s}: {hits}/{total} ({hits/total:.3f})")
        if kind == "word" and misses:
            print("    word-aware misses:",
                  ", ".join(f"{m[0]}[{m[1]}<{m[2]}]" for m in misses[:12]))

    print("\n=== FALSE POSITIVES (genuine text on correct layout -> no switch) ===")
    for kind in ("whole", "word"):
        fp, total, ex = run_fp(kind, model_args, random.Random(42))
        print(f"  {kind:5s}: {fp}/{total} ({fp/total:.4f})")
        if ex:
            print("    examples:", ", ".join(f"{e[0]}[{e[1]}->{e[2]}]" for e in ex[:8]))


if __name__ == "__main__":
    main()


