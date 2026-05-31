#!/usr/bin/env python3
"""
Validate the case-signal Hebrew exclusion introduced in Iteration 5-A.

Rationale
---------
Hebrew has no uppercase letters. Whenever the user types text with:
  • ≥ CaseExclusionMinCaps alpha characters typed with Shift (ALL-CAPS: FPS, USB)
  • any internal capital (CamelCase: iPhone, myVar, OAuth)

…it is physically impossible for that text to be Hebrew.  The C++ code
adds "he" to the exclusion set before running inference.

This script measures:
  1. RAW FP rate: how often does the ONNX model (without exclusion) incorrectly
     classify capitalised English/Russian text — typed on a wrong layout — as Hebrew?
  2. BLOCKED by case: how many of those FPs would be eliminated by the rule?
  3. SAFE: how many true Hebrew–from-en / Hebrew–from-ru cases would be incorrectly
     blocked? (Expected: 0, because real Hebrew has no capitals.)

Test corpus
-----------
  • English ALL-CAPS abbreviations typed on the Hebrew layout → garbled latin text
    that the model might confuse with Hebrew.
  • English CamelCase words typed on the Hebrew layout.
  • Russian ALL-CAPS (Cyrillic) typed on the Hebrew layout.
  • True Hebrew words typed on the English layout → must NOT be blocked.
    (Hebrew text will never have uppercase-intent characters, so UpperCount=0
    and HasInternalCapital=false → exclusion never fires → correct.)

Case-exclusion rule (mirrors C++ Config defaults):
  Exclude "he" when UpperCount >= CASE_EXCL_MIN_CAPS  OR  has internal capital.
  CASE_EXCL_MIN_CAPS = 2  (default from Config.cpp)
"""
from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Languages
from Languages import (
    english_layout, russian_layout, hebrew_layout,
    convert_text_bidirectional, predict_language_with_confidence,
)

# ── Mirror C++ Config defaults ────────────────────────────────────────────────
CASE_EXCL_MIN_CAPS = 2        # Config::CaseExclusionMinCaps
ENABLE_CASE_EXCL   = True     # Config::EnableCaseBasedHeExclusion


# ── Test word lists ───────────────────────────────────────────────────────────
# English ALL-CAPS abbreviations (≥ CASE_EXCL_MIN_CAPS uppercase chars)
EN_ALLCAPS = [
    "FBI", "CIA", "NASA", "NATO", "OPEC", "BMW", "USB", "FPS", "GPU", "CPU",
    "API", "SDK", "HTTP", "HTML", "JSON", "XML", "PDF", "RGB", "RAM", "ROM",
    "VPN", "SSD", "LCD", "LED", "GPS", "LAN", "WAN", "DNS", "TCP", "IP",
    "MVP", "CTO", "CEO", "CFO", "HR", "PR", "AI", "ML", "CV", "NLP",
    "OCR", "TTS", "STT", "QR", "AR", "VR", "UI", "UX", "DB", "ORM",
    "MVC", "MVP", "OOP", "IDE", "CI", "CD", "TDD", "DDD", "SLA", "KPI",
]

# English CamelCase / internal-capital words (HasInternalCapital = True)
EN_CAMELCASE = [
    "iPhone", "iPad", "iMac", "macOS", "macBook", "AirPods",
    "YouTube", "GitHub", "LinkedIn", "PayPal", "eBay", "eBook",
    "JavaScript", "TypeScript", "OpenAI", "ChatGPT", "WordPress",
    "OAuth", "MyApp", "getUserInfo", "onClick", "useEffect",
    "myVar", "firstName", "lastName", "emailAddress", "phoneNumber",
    "CamelCase", "PascalCase", "camelCase", "mixedCase",
]

# Russian ALL-CAPS (Cyrillic) typed on Hebrew layout → garbled Hebrew-range or Latin
RU_ALLCAPS = [
    "СССР", "РФ", "МВД", "ФСБ", "МЧС", "ЦБ", "ООН", "США", "ВВС", "ВМФ",
    "НДС", "ИНН", "КПП", "ООО", "АО", "ЗАО", "ИП", "СМИ", "ТВ", "РТ",
]

# True Hebrew words that MUST NOT be blocked (typed on English layout → garbled Latin)
HE_TRUE_POSITIVE = [
    "שלום", "תודה", "בבקשה", "כן", "לא", "מה", "איפה", "מתי", "למה", "איך",
    "ישראל", "ירושלים", "תל", "אביב", "חיפה", "בארץ", "עברית", "דברים",
    "אנחנו", "הולכים", "לאכול", "ארוחת", "ערב", "מחר", "בערב", "יחד",
    "בסדר", "נהדר", "מצוין", "רגע", "אחד", "שתיים", "שלוש", "ארבע",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def upper_count(text: str) -> int:
    """Count alphabetic chars with uppercase intent (mirrors InputCache::UpperCount).
    In Python we detect upper intent by isupper() on the character itself,
    since we don't have per-keystroke shift state; this is equivalent for
    latin/cyrillic text where case is preserved in the string.
    For Hebrew characters, isupper() always returns False → correct."""
    return sum(1 for c in text if c.isalpha() and c.isupper())


def has_internal_capital(text: str) -> bool:
    """True when any alpha char AFTER the first alpha char is uppercase.
    Mirrors InputCache::HasInternalCapital."""
    found_first = False
    for c in text:
        if not c.isalpha():
            continue
        if not found_first:
            found_first = True
            continue  # skip first alpha char (sentence-initial cap is OK)
        if c.isupper():
            return True
    return False


def would_be_excluded(text: str, min_caps: int = CASE_EXCL_MIN_CAPS) -> bool:
    """Mirror of the C++ exclusion decision for 'he'."""
    if not ENABLE_CASE_EXCL:
        return False
    return upper_count(text) >= min_caps or has_internal_capital(text)


LAYOUT_PAIRS = [
    (english_layout, russian_layout), (russian_layout, english_layout),
    (hebrew_layout,  english_layout), (english_layout,  hebrew_layout),
    (russian_layout, hebrew_layout),  (hebrew_layout,   russian_layout),
]


def best_prediction(text: str, model_args: list) -> Optional[str]:
    """Return the top predicted language across all 6 layout variants, or None."""
    variants = list(dict.fromkeys(
        convert_text_bidirectional(text, s, d) for s, d in LAYOUT_PAIRS
    ))
    bl, bc = None, 0.0
    for v in variants:
        res = predict_language_with_confidence(v, *model_args)
        if res and res.confidence > bc:
            bl, bc = res.language, res.confidence
    return bl


@dataclass
class CaseResult:
    word: str
    garbled: str
    raw_pred: Optional[str]          # model prediction without exclusion
    was_he_fp: bool                   # model said "he" (false positive)
    excluded_by_case: bool            # would case-rule have blocked "he"?
    tp_blocked: bool                  # was a TRUE Hebrew TP incorrectly blocked?


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate_group(
    words: List[str],
    native_layout: str,
    wrong_layout: str,
    label: str,
    model_args: list,
    expect_he: bool = False,
) -> List[CaseResult]:
    results = []
    for word in words:
        garbled = convert_text_bidirectional(word, native_layout, wrong_layout)
        if garbled == word:
            continue  # no conversion

        raw_pred       = best_prediction(garbled, model_args)
        was_he_fp      = (raw_pred == "he") and not expect_he
        excluded       = would_be_excluded(word)           # exclusion on the ORIGINAL word
        # For true-Hebrew case: exclusion must NOT fire (Hebrew text has no uppercase)
        tp_blocked     = expect_he and would_be_excluded(word)

        results.append(CaseResult(
            word=word, garbled=garbled, raw_pred=raw_pred,
            was_he_fp=was_he_fp, excluded_by_case=excluded, tp_blocked=tp_blocked,
        ))
    return results


def print_group_summary(label: str, results: List[CaseResult], expect_he: bool = False):
    if not results:
        print(f"  {label}: (no results)")
        return

    if expect_he:
        blocked = sum(1 for r in results if r.tp_blocked)
        print(f"  {label}: {len(results)} words — Hebrew TP blocked by case rule: {blocked}/{len(results)}", end="")
        if blocked:
            print("  ← PROBLEM", end="")
            for r in results:
                if r.tp_blocked:
                    print(f"\n    BLOCKED: '{r.word}' → '{r.garbled}' (upper={upper_count(r.word)}, internalCap={has_internal_capital(r.word)})", end="")
        print()
        return

    fp_raw   = sum(1 for r in results if r.was_he_fp)
    fp_blocked = sum(1 for r in results if r.was_he_fp and r.excluded_by_case)
    fp_remain  = fp_raw - fp_blocked
    rule_would_block = sum(1 for r in results if r.excluded_by_case)

    print(f"  {label}: {len(results)} words")
    print(f"    Model 'he' FPs (raw):   {fp_raw}/{len(results)}")
    print(f"    Blocked by case rule:   {fp_blocked}/{fp_raw if fp_raw else '-'}")
    print(f"    Remaining FPs:          {fp_remain}")
    print(f"    Rule would have blocked:{rule_would_block}/{len(results)} (includes non-FP too)")
    if fp_remain > 0:
        print(f"    Unblocked FP examples:")
        for r in results:
            if r.was_he_fp and not r.excluded_by_case:
                print(f"      '{r.word}' → '{r.garbled}' pred={r.raw_pred} uc={upper_count(r.word)}")


def main():
    print("Loading ONNX model...")
    model_args = Languages.load_model()

    print()
    print("=" * 70)
    print("  Case-signal Hebrew exclusion — validation")
    print(f"  Config: EnableCaseBasedHeExclusion={ENABLE_CASE_EXCL}, "
          f"CaseExclusionMinCaps={CASE_EXCL_MIN_CAPS}")
    print("=" * 70)

    # ── Group A: English ALL-CAPS typed on Hebrew layout → should not be Hebrew
    print("\n[A] English ALL-CAPS typed on Hebrew layout (FP guard):")
    res_a = evaluate_group(EN_ALLCAPS, english_layout, hebrew_layout,
                           "en-ALLCAPS → he layout", model_args, expect_he=False)
    print_group_summary("en ALL-CAPS → Hebrew layout", res_a)

    # ── Group B: English CamelCase typed on Hebrew layout → should not be Hebrew
    print("\n[B] English CamelCase typed on Hebrew layout (FP guard):")
    res_b = evaluate_group(EN_CAMELCASE, english_layout, hebrew_layout,
                           "en-CamelCase → he layout", model_args, expect_he=False)
    print_group_summary("en CamelCase → Hebrew layout", res_b)

    # ── Group C: Russian ALL-CAPS typed on Hebrew layout → should not be Hebrew
    print("\n[C] Russian ALL-CAPS typed on Hebrew layout (FP guard):")
    res_c = evaluate_group(RU_ALLCAPS, russian_layout, hebrew_layout,
                           "ru-ALLCAPS → he layout", model_args, expect_he=False)
    print_group_summary("ru ALL-CAPS → Hebrew layout", res_c)

    # ── Group D: True Hebrew typed on English layout — exclusion MUST NOT fire
    # Hebrew chars have no uppercase, so UpperCount=0 and HasInternalCapital=False.
    print("\n[D] True Hebrew words typed on English layout (TP safety check):")
    res_d = evaluate_group(HE_TRUE_POSITIVE, hebrew_layout, english_layout,
                           "he → en layout (garbled)", model_args, expect_he=True)
    print_group_summary("true Hebrew → English layout", res_d, expect_he=True)

    # ── Summary ─────────────────────────────────────────────────────────���─────
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    total_fps = (sum(1 for r in res_a if r.was_he_fp) +
                 sum(1 for r in res_b if r.was_he_fp) +
                 sum(1 for r in res_c if r.was_he_fp))
    blocked_fps = (sum(1 for r in res_a if r.was_he_fp and r.excluded_by_case) +
                   sum(1 for r in res_b if r.was_he_fp and r.excluded_by_case) +
                   sum(1 for r in res_c if r.was_he_fp and r.excluded_by_case))
    safe_blocked = sum(1 for r in res_d if r.tp_blocked)

    print(f"  Total Hebrew FPs in A+B+C:   {total_fps}")
    print(f"  Blocked by case-rule:         {blocked_fps}")
    if total_fps:
        print(f"  Block rate:                  {blocked_fps/total_fps:.1%}")
    print(f"  True-Hebrew TPs blocked:      {safe_blocked}  (must be 0)")
    if safe_blocked == 0:
        print("  → Safety check PASSED: no real Hebrew text is blocked. ✓")
    else:
        print("  → Safety check FAILED: some real Hebrew was incorrectly excluded!")


if __name__ == "__main__":
    main()

