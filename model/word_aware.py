#!/usr/bin/env python3
"""word_aware.py — word-aware (per-word) aggregation of language detection.

Rationale
---------
The production detector feeds the WHOLE accumulated buffer (e.g. the mistyped
render of "I have a pen") into the model as a single string and takes one
softmax over the entire phrase.  Short words, prepositions and particles get
diluted in that single verdict, and one odd token can flip the whole phrase.

This module segments the buffer into words (on spaces), scores each word
independently, and combines the per-word softmax vectors with weights that
reflect how much language signal each word actually carries:

  * longer words are more reliable   -> weight grows with (known-char) length,
    capped so a single very long word cannot dominate;
  * very short words (<= SHORT_WORD_MAX_LEN) carry little signal
    -> down-weighted by SHORT_WORD_WEIGHT;
  * closed-class stop-words (prepositions / particles / articles) are the
    same across typo-noise and are ambiguous between languages
    -> down-weighted by STOP_WORD_WEIGHT (kept as a weak tie-breaker, not
       ignored, per the agreed design).

The aggregated softmax is renormalised and returned as a DetectionResult, so
the rest of the pipeline (thresholds, margin gate, agreement gate) is
unchanged — only the *per-variant score* becomes word-aware.

Both the weighting scheme and the constants here are mirrored 1:1 in the C++
implementation (cpp/src/Languages.cpp :: PredictWordAware).
"""
from __future__ import annotations

from typing import Optional, Iterable

from Languages import (
    DetectionResult,
    predict_language_with_confidence,
)

# ── Tunable weighting parameters (mirrored in cpp/include/Config.h) ──────────
# The SHIPPED behaviour (and the C++ port) uses ONLY length-based short-word
# down-weighting: it was the geometric-mean consensus in predict_consensus that
# drove the offline win, while stop-word / sharpen / conf-floor refinements
# proved neutral on the benchmark (model/ablate_stopwords.py).  The extra knobs
# are kept here (defaulting to INERT values) purely for offline experimentation.
SHORT_WORD_MAX_LEN = 2      # words this short or shorter are "short"
SHORT_WORD_WEIGHT  = 0.35   # weight multiplier for short words
STOP_WORD_WEIGHT   = 1.00   # inert by default (proved neutral); <1 to experiment
LEN_CAP            = 8      # cap on the length-based base weight
MIN_WORDS_FOR_AGG  = 2      # only aggregate when >= this many words

# Confidence-shaping (experimental, INERT by default — proved neutral offline).
WORD_CONF_FLOOR    = 0.0    # 0 disables; >0 diverts unsure mass to N/A
SHARPEN            = 1.0    # 1.0 = no sharpening; >1 sharpens confident words

# Closed-class stop words (prepositions / particles / articles / pronouns)
# that carry little discriminative language signal on their own.  Kept small
# and high-frequency; matched case-insensitively on the *native* rendering.
# Note: because a word is only ever scored on its own layout render, we list
# the native forms for each language.
_STOP_WORDS = {
    # English
    "a", "an", "the", "to", "of", "in", "on", "at", "is", "am", "are", "be",
    "i", "it", "as", "or", "and", "if", "so", "we", "he", "my", "me", "up",
    "do", "no", "by", "for", "you", "our",
    # Russian (prepositions / particles / pronouns)
    "и", "в", "во", "не", "на", "я", "с", "со", "как", "а", "то", "все",
    "он", "но", "да", "ты", "к", "у", "же", "за", "бы", "по", "ее", "мне",
    "их", "чтобы", "был", "до", "вы", "их", "из", "ли", "если", "или",
    # Hebrew (prepositions / particles / conjunctions)
    "אני", "לא", "זה", "של", "את", "כי", "עם", "יש", "או", "גם", "כל",
    "מה", "לי", "הוא", "היא", "אם", "כן", "רק", "עוד", "אבל", "הכל",
}


def _known_len(word: str, char_to_index) -> int:
    return sum(1 for c in word if c in char_to_index)


def word_weight(word: str, char_to_index) -> float:
    """Weight this word's contribution to the aggregate softmax."""
    n = _known_len(word, char_to_index)
    if n == 0:
        return 0.0
    w = float(min(n, LEN_CAP))
    if n <= SHORT_WORD_MAX_LEN:
        w *= SHORT_WORD_WEIGHT
    if word.lower() in _STOP_WORDS:
        w *= STOP_WORD_WEIGHT
    return w


def _scores_to_result(scores) -> Optional[DetectionResult]:
    best_class, best_prob = 0, 0.0
    for i in range(1, 4):
        if scores[i] > best_prob:
            best_prob = scores[i]
            best_class = i
    if scores[0] >= best_prob:
        return None
    lang = {1: "en", 2: "he", 3: "ru"}[best_class]
    return DetectionResult(language=lang, confidence=best_prob, scores=scores)


def predict_word_aware(text: str, ort_session, char_to_index, max_length,
                       *, short_len=SHORT_WORD_MAX_LEN, short_w=SHORT_WORD_WEIGHT,
                       stop_w=STOP_WORD_WEIGHT, len_cap=LEN_CAP,
                       conf_floor=WORD_CONF_FLOOR, sharpen=SHARPEN
                       ) -> Optional[DetectionResult]:
    """Word-aware analogue of predict_language_with_confidence.

    Splits *text* on spaces, scores each word, and returns the weighted-mean
    softmax as a DetectionResult.  Falls back to the whole-string prediction
    when the input is a single word.

    Each word contributes a *confidence-shaped* softmax:
      * the unsure fraction (1 - top1conf) of a low-confidence word (top1 <
        conf_floor) is diverted to N/A, so diffuse weak leanings do not add up;
      * kept per-class mass is raised to `sharpen` and renormalised, so
        confident words dominate the aggregate.
    """
    words = [w for w in text.split(" ") if w]
    if len(words) < MIN_WORDS_FOR_AGG:
        return predict_language_with_confidence(
            text, ort_session, char_to_index, max_length)

    def weight(word: str) -> float:
        n = _known_len(word, char_to_index)
        if n == 0:
            return 0.0
        w = float(min(n, len_cap))
        if n <= short_len:
            w *= short_w
        if word.lower() in _STOP_WORDS:
            w *= stop_w
        return w

    agg = [0.0, 0.0, 0.0, 0.0]
    total_w = 0.0
    for w in words:
        wt = weight(w)
        if wt <= 0.0:
            continue
        r = predict_language_with_confidence(
            w, ort_session, char_to_index, max_length)
        if r is None:
            agg[0] += wt
        else:
            s = list(r.scores)
            if sharpen != 1.0:
                s = [x ** sharpen for x in s]
                z = sum(s) or 1.0
                s = [x / z for x in s]
            # Divert the unsure fraction of a low-confidence word to N/A.
            top1 = max(s[1:4])
            if top1 < conf_floor and conf_floor > 0.0:
                keep = top1 / conf_floor            # 0..1 confidence-scaled
                s = [s[0] + (1.0 - keep)] + [x * keep for x in s[1:4]]
            for i in range(4):
                agg[i] += wt * s[i]
        total_w += wt

    if total_w <= 0.0:
        return None
    agg = [a / total_w for a in agg]
    return _scores_to_result(agg)




