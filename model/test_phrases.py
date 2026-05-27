#!/usr/bin/env python3
"""Phrase-level detection test for KeyboardSwitcher.

Tests that detection fires correctly on full phrases typed on wrong keyboards.
Generates hundreds of phrase examples automatically from vocabulary and also
includes a curated set of idioms and common sentences.

Examples from user:
  "еир кгьм"   → Hebrew  אני רוצה  (typed on Russian keyboard)
  "fl rmh,h"   → Hebrew  כך רציתי  (typed on English keyboard)

Usage:
    python test_phrases.py
    python test_phrases.py --verbose      # show each phrase result
    python test_phrases.py --csv out.csv  # save full results
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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

# ── Must match Config.cpp PairOverrides exactly ─────────────────────────────
PAIR_OVERRIDES = {
    ("en", "ru"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    ("ru", "en"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    ("en", "he"): (3, 15, 0.97, 0.65, 2, 0.88, 0.10, 0, 4, 2, 0.0, 0.0,  0.80),
    ("he", "en"): (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
    ("ru", "he"): (3, 15, 0.97, 0.65, 2, 0.88, 0.10, 0, 4, 2, 0.0, 0.0,  0.80),
    ("he", "ru"): (3, 15, 0.99, 0.70, 2, 0.80, 0.05, 0, 4, 2, 0.0, 0.02, 1.00),
}

# ── Curated phrase sets ─────────────────────────────────────────────────────
# Common phrases that users actually type.  Each entry: (native_text, lang)
CURATED_PHRASES = [
    # Hebrew phrases — typed on English keyboard (user's examples + more)
    ("אני רוצה",         "he"),   # I want
    ("כך רציתי",         "he"),   # this is how I wanted
    ("שלום לכולם",       "he"),   # hello everyone
    ("תודה רבה",         "he"),   # thank you very much
    ("מה שלומך",         "he"),   # how are you
    ("אני לא יודע",      "he"),   # I don't know
    ("בסדר גמור",        "he"),   # absolutely fine
    ("עד מחר",           "he"),   # until tomorrow
    ("לילה טוב",         "he"),   # good night
    ("בוקר טוב",         "he"),   # good morning
    ("אני עייף",         "he"),   # I'm tired
    ("יש לי שאלה",       "he"),   # I have a question
    ("איפה אתה",         "he"),   # where are you
    ("מתי נפגש",         "he"),   # when shall we meet
    ("אני מסכים",        "he"),   # I agree
    ("זה לא נכון",       "he"),   # this is not right
    ("כמה זה עולה",      "he"),   # how much does it cost
    ("אני אוהב אותך",    "he"),   # I love you
    ("שנה טובה",         "he"),   # happy new year
    ("חג שמח",           "he"),   # happy holiday
    ("ברוך הבא",         "he"),   # welcome
    ("תכתוב לי",         "he"),   # write to me
    ("אני חוזר עוד רגע", "he"),   # I'll be right back
    ("כן בטח",           "he"),   # yes of course
    ("לא עכשיו",         "he"),   # not now
    ("קח את הזמן",       "he"),   # take your time
    ("הכל בסדר",         "he"),   # everything is fine
    ("נשמע טוב",         "he"),   # sounds good
    ("אנחנו מחכים",      "he"),   # we are waiting
    ("תהיה בריא",        "he"),   # stay healthy

    # Hebrew phrases — typed on Russian keyboard
    ("אני צריך עזרה",    "he"),   # I need help
    ("אתה מבין",         "he"),   # do you understand
    ("בוא נדבר",         "he"),   # let's talk
    ("זה מאוד חשוב",     "he"),   # this is very important
    ("תודה לך",          "he"),   # thank you (to you)
    ("מה קורה",          "he"),   # what's happening
    ("הכל טוב",          "he"),   # all good
    ("אני בדרך",         "he"),   # I'm on the way
    ("עוד מעט",          "he"),   # soon
    ("קצת יותר מאוחר",   "he"),   # a little later

    # Russian phrases — typed on English keyboard
    ("привет как дела",  "ru"),   # hello how are you
    ("я не знаю",        "ru"),   # I don't know
    ("спасибо большое",  "ru"),   # thank you very much
    ("до свидания",      "ru"),   # goodbye
    ("хорошо понял",     "ru"),   # understood well
    ("я согласен",       "ru"),   # I agree
    ("не могу сейчас",   "ru"),   # can't right now
    ("позвони мне",      "ru"),   # call me
    ("всё в порядке",    "ru"),   # everything is fine
    ("увидимся завтра",  "ru"),   # see you tomorrow
    ("мне нужна помощь", "ru"),   # I need help
    ("что случилось",    "ru"),   # what happened
    ("я уже иду",        "ru"),   # I'm already going
    ("когда встретимся", "ru"),   # when shall we meet
    ("пока что всё",     "ru"),   # that's all for now
    ("отличная идея",    "ru"),   # great idea
    ("буду через час",   "ru"),   # I'll be there in an hour
    ("не понимаю",       "ru"),   # I don't understand
    ("спокойной ночи",   "ru"),   # good night
    ("доброе утро",      "ru"),   # good morning
    ("хорошего дня",     "ru"),   # have a good day
    ("ты прав",          "ru"),   # you are right
    ("конечно да",       "ru"),   # of course yes
    ("подожди немного",  "ru"),   # wait a little
    ("я тебя люблю",     "ru"),   # I love you
    ("всё понятно",      "ru"),   # all clear
    ("очень хорошо",     "ru"),   # very good
    ("давай потом",      "ru"),   # let's do it later
    ("это здорово",      "ru"),   # this is great
    ("уже почти готово", "ru"),   # almost done

    # English phrases — typed on Russian keyboard
    ("hello how are you",     "en"),
    ("thank you very much",   "en"),
    ("see you tomorrow",      "en"),
    ("I need your help",      "en"),
    ("good morning everyone", "en"),
    ("what do you think",     "en"),
    ("lets meet tonight",     "en"),
    ("sounds good to me",     "en"),
    ("I will be right back",  "en"),
    ("have a nice day",       "en"),
    ("please let me know",    "en"),
    ("I agree with you",      "en"),
    ("not sure about this",   "en"),
    ("can you call me",       "en"),
    ("just a moment please",  "en"),
    ("of course why not",     "en"),
    ("I understand now",      "en"),
    ("talk to you later",     "en"),
    ("happy to help you",     "en"),
    ("good job everyone",     "en"),
    ("please confirm this",   "en"),
    ("where are you now",     "en"),
    ("on my way there",       "en"),
    ("almost done here",      "en"),
    ("hope you are well",     "en"),
    ("nice to meet you",      "en"),
    ("I am not sure yet",     "en"),
    ("let me think about",    "en"),
    ("will do right away",    "en"),
    ("thank you for that",    "en"),

    # English phrases — typed on Hebrew keyboard
    ("hello my friend",       "en"),
    ("good night everyone",   "en"),
    ("I love this place",     "en"),
    ("see you next week",     "en"),
    ("very well done",        "en"),
    ("please wait for me",    "en"),
    ("I am coming now",       "en"),
    ("what time is it",       "en"),
    ("where do we go",        "en"),
    ("yes that is right",     "en"),
]


def required_conf_phrase(n: int, params: tuple, is_phrase: bool) -> float:
    if result is None:
        return "", 0.0
    scores = result.scores
    cl = ["", "en", "he", "ru"]
    best_lang = result.language
    runner_conf = 0.0; runner_lang = ""
    for i in range(1, 4):
        if cl[i] != best_lang and scores[i] > runner_conf:
            runner_conf = scores[i]; runner_lang = cl[i]
    return runner_lang, runner_conf


def required_conf_phrase(n: int, params: tuple, is_phrase: bool) -> float:
    early_min, full_conf, conf_min, conf_max = params[0], params[1], params[2], params[3]
    phrase_scale = params[12] if len(params) > 12 else 1.0
    if n < early_min: return 1.1
    if n >= full_conf: base = conf_max
    else:
        t = (n - early_min) / (full_conf - early_min)
        base = conf_min + t * (conf_max - conf_min)
    if is_phrase and phrase_scale < 1.0:
        base *= phrase_scale
    return base


def get_runner_up(result) -> Tuple[str, float]:
    if result is None:
        return "", 0.0
    scores = result.scores
    cl = ["", "en", "he", "ru"]
    best_lang = result.language
    runner_conf = 0.0; runner_lang = ""
    for i in range(1, 4):
        if cl[i] != best_lang and scores[i] > runner_conf:
            runner_conf = scores[i]; runner_lang = cl[i]
    return runner_lang, runner_conf


def simulate_phrase(
    text: str,
    current_lang: str,
    model_args: list,
    cache: dict,
    verbose: bool = False,
) -> Tuple[Optional[str], int, float, float]:
    """Simulate incremental typing of a phrase on wrong keyboard.

    Returns (detected_lang, chars_at_detection, final_conf, final_margin).
    Mirrors ALL gates from TypoResilientDetect.
    """
    # Use the pair params that would fire if detection succeeds with any target
    # We don't know the target a priori, so we try all pairs.
    # Use the most common case's params for candidates.

    last_lang = ""; streak = 0
    window: List[tuple] = []
    last_conf = 0.0; last_margin = 0.0

    for n in range(1, len(text) + 1):
        partial = text[:n]
        det_text = "".join(c for c in partial if c.isalpha() or c == " ").strip()
        alpha_count = sum(1 for c in partial if c.isalpha())
        if alpha_count < 3 or len(det_text) < 3:
            continue

        seen: set = set(); variants = []
        for src, dst in LAYOUT_PAIRS:
            v = convert_text_bidirectional(det_text, src, dst)
            if v not in seen: seen.add(v); variants.append(v)

        bl = None; bc = 0.0; br = None; lang_votes: Dict[str, int] = {}
        for v in variants:
            r = cache.get(v)
            if r is None:
                r = predict_language_with_confidence(v, *model_args)
                cache[v] = r
            if r is not None:
                lang_votes[r.language] = lang_votes.get(r.language, 0) + 1
                if r.confidence > bc: bc = r.confidence; bl = r.language; br = r

        if not bl: last_lang = ""; streak = 0; window = []; continue

        runner_lang, runner_conf = get_runner_up(br)
        margin = bc - runner_conf; last_conf = bc; last_margin = margin

        if bl == last_lang: streak += 1
        else: last_lang = bl; streak = 1

        window.append((bl, bc, runner_conf, margin))
        if len(window) > 4: window = window[-4:]

        # Look up per-pair params
        params = PAIR_OVERRIDES.get((current_lang, bl), (3, 15, 0.99, 0.70, 2, 0.85, 0.05, 0, 4, 2, 0.0, 0.02, 1.00))
        (early_min, full_conf, conf_min, conf_max,
         agreement_count, _border, min_margin, variant_agree,
         trend_window, min_steps, min_slope, short_extra_conf,
         phrase_scale) = (*params[:12], params[12] if len(params) > 12 else 1.0)

        is_phrase = " " in det_text
        req = required_conf_phrase(alpha_count, params, is_phrase)
        is_short = (alpha_count <= early_min + 2)
        if is_short and not is_phrase and short_extra_conf > 0:
            req = min(0.9999, req + short_extra_conf)
        if alpha_count < early_min: continue

        # Margin gate
        if min_margin > 0 and margin < min_margin: continue

        # Variant consensus gate
        v_agree = lang_votes.get(bl, 0)
        if variant_agree > 0 and bc < conf_min * 0.99:
            if v_agree < variant_agree: continue

        # Only switch when differs from current
        if bl == current_lang: continue

        # Agreement gate
        agreement_ok = (streak >= agreement_count)

        # Trend gate (OR-alternative)
        trend_ok = False
        if trend_window > 0 and min_steps > 0:
            to_frames = [f for f in window if f[0] == bl]
            if len(to_frames) >= min_steps:
                trend_ok = True

        if not agreement_ok and not trend_ok: continue

        # Confidence gate
        if bc >= req:
            if verbose:
                print(f"      [char {n:3d}] lang={bl} conf={bc:.3f} margin={margin:.3f} "
                      f"streak={streak} α={alpha_count}")
            return bl, n, bc, margin

    return None, len(text), last_conf, last_margin


@dataclass
class PhraseResult:
    native_text: str
    native_lang: str
    keyboard_lang: str       # wrong keyboard
    mistyped: str
    expected_lang: str       # same as native_lang
    detected_lang: Optional[str]
    chars_at_det: int
    conf: float
    margin: float
    correct: bool


def build_phrase_tests(
    curated: List[Tuple[str, str]],
    vocab_words: Dict[str, List[str]],
    n_synthetic: int,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """Build list of (native_phrase, native_lang, keyboard_lang) test cases.

    Includes:
    1. Curated phrases on ALL wrong keyboards
    2. Synthetic 2-3 word phrases built from vocabulary
    """
    cases: List[Tuple[str, str, str]] = []
    all_keyboards = ["en", "ru", "he"]

    # Curated phrases on each wrong keyboard
    for phrase, lang in curated:
        for kb in all_keyboards:
            if kb != lang:
                cases.append((phrase, lang, kb))

    # Synthetic 2-word phrases
    for native_lang in ["en", "ru", "he"]:
        pool = vocab_words[native_lang]
        if len(pool) < 4:
            continue
        pool_long = [w for w in pool if len(w) >= 4]
        if len(pool_long) < 2:
            continue
        count = min(n_synthetic, len(pool_long) // 2)
        for _ in range(count):
            w1, w2 = rng.sample(pool_long, 2)
            phrase = f"{w1} {w2}"
            for kb in all_keyboards:
                if kb != native_lang:
                    cases.append((phrase, native_lang, kb))

    # Synthetic 3-word phrases (longer, easier to detect)
    for native_lang in ["en", "ru", "he"]:
        pool = vocab_words[native_lang]
        pool_med = [w for w in pool if 3 <= len(w) <= 8]
        if len(pool_med) < 3:
            continue
        count = min(n_synthetic // 2, len(pool_med) // 3)
        for _ in range(count):
            w1, w2, w3 = rng.sample(pool_med, 3)
            phrase = f"{w1} {w2} {w3}"
            for kb in all_keyboards:
                if kb != native_lang:
                    cases.append((phrase, native_lang, kb))

    return cases


def load_vocab(lang: str, n: int = 200) -> List[str]:
    fmap = {
        "en": "vocabulary/english_vocabulary",
        "ru": "vocabulary/russian_vocabulary",
        "he": "vocabulary/hebrew_vocabulary",
    }
    words = []
    path = os.path.join(SCRIPT_DIR, fmap[lang])
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if 3 <= len(w) <= 10 and w.isalpha():
                words.append(w)
    return random.sample(words, min(n, len(words)))


def run_tests(
    cases: List[Tuple[str, str, str]],
    model_args: list,
    verbose: bool = False,
) -> List[PhraseResult]:
    cache: dict = {}
    results: List[PhraseResult] = []

    for i, (native, native_lang, kb_lang) in enumerate(cases):
        mistyped = convert_text_bidirectional(native, LAYOUTS[native_lang], LAYOUTS[kb_lang])
        if mistyped == native:
            continue  # No change — skip (layout maps identical for this pair)

        det, chars_det, conf, margin = simulate_phrase(
            mistyped, kb_lang, model_args, cache, verbose=verbose)
        correct = (det == native_lang)
        results.append(PhraseResult(
            native_text=native,
            native_lang=native_lang,
            keyboard_lang=kb_lang,
            mistyped=mistyped,
            expected_lang=native_lang,
            detected_lang=det,
            chars_at_det=chars_det,
            conf=conf,
            margin=margin,
            correct=correct,
        ))

        if verbose:
            mark = "✓" if correct else "✗"
            pair = f"{kb_lang}->{native_lang}"
            native_short = native[:30]
            print(f"  [{mark}] {pair:8s}  {mistyped!r:35s} → {native_short!r}")
            if not correct:
                print(f"       detected={det}  conf={conf:.3f}  margin={margin:.3f}")

    return results


def print_summary(results: List[PhraseResult], title: str) -> None:
    from collections import defaultdict
    per_pair: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "chars": []})

    for r in results:
        pair = f"{r.keyboard_lang}->{r.native_lang}"
        if r.correct:
            per_pair[pair]["tp"] += 1
            per_pair[pair]["chars"].append(r.chars_at_det)
        else:
            per_pair[pair]["fn"] += 1

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {'Pair':<12} {'Detected':>10} {'Total':>7} {'Recall':>8}  {'AvgChar':>8}")
    print(f"  {'-'*55}")

    total_tp = total_tot = 0
    for pair in sorted(per_pair.keys()):
        d = per_pair[pair]
        tp = d["tp"]; fn = d["fn"]; tot = tp + fn
        rec = tp / tot if tot > 0 else 0.0
        avg = sum(d["chars"]) / len(d["chars"]) if d["chars"] else 0
        mark = "✓" if rec >= 0.80 else ("~" if rec >= 0.60 else "✗")
        print(f"  [{mark}] {pair:<10}  {tp:>10}  {tot:>6}  {rec:>7.1%}  {avg:>7.1f}")
        total_tp += tp; total_tot += tot

    overall = total_tp / total_tot if total_tot > 0 else 0.0
    print(f"  {'-'*55}")
    print(f"      {'TOTAL':<10}  {total_tp:>10}  {total_tot:>6}  {overall:>7.1%}")
    print(f"{'='*65}")


def print_failures(results: List[PhraseResult], max_per_pair: int = 5) -> None:
    from collections import defaultdict
    by_pair: Dict[str, List[PhraseResult]] = defaultdict(list)
    for r in results:
        if not r.correct:
            by_pair[f"{r.keyboard_lang}->{r.native_lang}"].append(r)

    if not by_pair:
        print("\n  ✓ No failures!")
        return

    print(f"\n  Failed detections (up to {max_per_pair} per pair):")
    for pair in sorted(by_pair.keys()):
        print(f"\n  {pair}:")
        for r in by_pair[pair][:max_per_pair]:
            det_str = r.detected_lang or "None"
            print(f"    native:   {r.native_text!r}")
            print(f"    mistyped: {r.mistyped!r}")
            print(f"    detected: {det_str}  conf={r.conf:.3f}  margin={r.margin:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phrase-level detection test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-phrase results")
    parser.add_argument("--csv", type=str, default="",
                        help="Save results to CSV")
    parser.add_argument("--synthetic", type=int, default=40,
                        help="Synthetic 2-word phrases per language to generate (default: 40)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failures", action="store_true",
                        help="Print detailed failure cases")
    args = parser.parse_args()

    print("=" * 65)
    print("  KeyboardSwitcher — Phrase-Level Detection Test")
    print("=" * 65)
    print(f"\n  Curated phrases: {len(CURATED_PHRASES)}")
    print(f"  Synthetic phrases per lang: {args.synthetic}")

    print("\nLoading ONNX model...")
    model_args = Languages.load_model()

    rng = random.Random(args.seed)
    vocab = {lang: load_vocab(lang) for lang in ["en", "ru", "he"]}
    print(f"Vocab loaded: " + "  ".join(f"{l}={len(v)}" for l, v in vocab.items()))

    # Build test suite
    cases = build_phrase_tests(CURATED_PHRASES, vocab, args.synthetic, rng)
    print(f"\nTotal test cases: {len(cases)}")
    print("Running detection simulation...\n")

    if args.verbose:
        print("─" * 65)
    results = run_tests(cases, model_args, verbose=args.verbose)
    if args.verbose:
        print("─" * 65)

    # Split curated vs synthetic
    curated_keys = set(
        (p, l, kb)
        for p, l in CURATED_PHRASES
        for kb in ["en", "ru", "he"] if kb != l
    )
    curated_results = [r for r in results
                       if (r.native_text, r.native_lang, r.keyboard_lang) in curated_keys]
    synth_results   = [r for r in results if r not in curated_results]  # type: ignore

    print_summary(curated_results, "CURATED PHRASES")
    print_summary(synth_results,   "SYNTHETIC 2-3 WORD PHRASES")
    print_summary(results,         "OVERALL (CURATED + SYNTHETIC)")

    if args.failures:
        print("\n── Curated phrase failures ──")
        print_failures(curated_results)
        print("\n── Synthetic phrase failures ──")
        print_failures(synth_results)

    # Print user's specific examples
    print("\n── User example phrases ──")
    user_examples = [
        ("אני רוצה",  "he", "ru"),   # еир кгьм
        ("כך רציתי",  "he", "en"),   # fl rmh,h
        ("שלום לכולם","he", "en"),
        ("תודה רבה",  "he", "en"),
        ("תודה רבה",  "he", "ru"),
        ("привет как дела", "ru", "en"),
        ("спасибо большое", "ru", "en"),
        ("hello how are you", "en", "ru"),
        ("I need your help",  "en", "ru"),
        ("I need your help",  "en", "he"),
    ]
    cache: dict = {}
    for native, nat_lang, kb in user_examples:
        mistyped = convert_text_bidirectional(native, LAYOUTS[nat_lang], LAYOUTS[kb])
        det, chars, conf, margin = simulate_phrase(mistyped, kb, model_args, cache)
        mark = "✓" if det == nat_lang else "✗"
        print(f"  [{mark}] {kb}->{nat_lang}  "
              f"'{mistyped}' → '{native}' "
              f"(det={det} char={chars} conf={conf:.3f})")

    if args.csv:
        csv_path = os.path.join(SCRIPT_DIR, args.csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "native_text", "native_lang", "keyboard_lang", "mistyped",
                "detected_lang", "correct", "chars_at_det", "conf", "margin"
            ])
            for r in results:
                writer.writerow([
                    r.native_text, r.native_lang, r.keyboard_lang, r.mistyped,
                    r.detected_lang or "", r.correct,
                    r.chars_at_det, f"{r.conf:.4f}", f"{r.margin:.4f}",
                ])
        print(f"\n  Results saved to: {csv_path}")


if __name__ == "__main__":
    main()







