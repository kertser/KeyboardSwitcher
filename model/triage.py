#!/usr/bin/env python3
"""Lexical triage stage — fast dictionary/n-gram evidence for language detection.

Implements the hybrid-pipeline triage signals:
  * exact dictionary membership (hash-set lookup, prefix-aware for the word
    being typed — bisect over a sorted lexicon, mirroring the C++
    FeedbackLogger prefix matching style),
  * fuzzy matching via a BK-tree (Levenshtein edit distance <= 1),
  * character-trigram log-likelihood scoring (per-language tables built by
    build_triage_artifacts.py, calibrated to a [0,1] plausibility),
  * optional LSH (minhash over character trigrams, banded index) as an
    approximate-fuzzy alternative, kept behind a flag for ablation.

The triage is ADVISORY ONLY: it returns per-language evidence in [0,1] that
the detection pipeline uses to bias its confidence requirements (boost a
well-attested switch candidate, suppress a switch away from text that is
valid in the current language).  The LSTM always runs.

Scripts are disjoint (Latin/Cyrillic/Hebrew), so a variant can only carry
evidence for the single language whose script it is written in; evidence for
any other language is 0.

Designed for a 1:1 port to C++ (cpp/src/Triage.*).
"""
from __future__ import annotations

import bisect
import os
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz.distance import Levenshtein as _RF_Lev

    def _levenshtein(a: str, b: str) -> int:
        return _RF_Lev.distance(a, b)

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pure-python fallback (slower BK-tree build)
    _HAVE_RAPIDFUZZ = False

    def _levenshtein(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "triage_data")

# Hebrew final forms -> regular forms (mirrors cpp NormalizeHebrewFinals)
HEBREW_FINALS = str.maketrans({
    "\u05DA": "\u05DB", "\u05DD": "\u05DE", "\u05DF": "\u05E0",
    "\u05E3": "\u05E4", "\u05E5": "\u05E6",
})


# ============================================================
# Tunable parameters (swept by sweep_triage_params.py)
#
# Tuned defaults (sample 200, seeds 42/7/123): exact+prefix evidence
# only — fuzzy/LSH/trigram contribute nothing at safe thresholds and
# ED<=1 fuzzy is noisy on these dense lexicons (costs TP). boost_scale
# 0.80 is the conservative plateau (+42..+59 TP, FP -> 0, phrases kept).
# ============================================================
@dataclass
class TriageParams:
    # Evidence component weights — evidence = max(weight_i * component_i)
    w_exact: float = 1.00     # exact/prefix dictionary membership
    w_fuzzy: float = 0.85     # BK-tree edit-distance<=1 membership (if enabled)
    w_trigram: float = 0.60   # calibrated trigram plausibility
    w_lsh: float = 0.50       # LSH approximate membership (ablation)

    # Integration thresholds (advisory gating in the detection pipeline)
    suppress_threshold: float = 0.999  # identity evidence >= this => suppress
    suppress_extra: float = 0.08       # added to required confidence
    suppress_blocks_gates: bool = True # also veto Tier-3 (->he) gates
    boost_threshold: float = 0.80      # switch-candidate evidence >= this
    boost_scale: float = 0.80          # multiplies required confidence

    # Matching behavior
    min_prefix_len: int = 3    # last (partial) word needs this many chars
    fuzzy_min_len: int = 4     # fuzzy matching only for words this long


# ============================================================
# BK-tree (metric tree over Levenshtein distance)
# ============================================================
class BKTree:
    __slots__ = ("_root",)

    def __init__(self) -> None:
        # node = [word, {distance: child_node}]
        self._root: Optional[list] = None

    def add(self, word: str) -> None:
        if self._root is None:
            self._root = [word, {}]
            return
        node = self._root
        while True:
            d = _levenshtein(word, node[0])
            if d == 0:
                return
            child = node[1].get(d)
            if child is None:
                node[1][d] = [word, {}]
                return
            node = child

    def contains_within(self, word: str, max_dist: int = 1) -> bool:
        """True if any stored word is within max_dist of *word*."""
        if self._root is None:
            return False
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = _levenshtein(word, node[0])
            if d <= max_dist:
                return True
            lo, hi = d - max_dist, d + max_dist
            for dist, child in node[1].items():
                if lo <= dist <= hi:
                    stack.append(child)
        return False


# ============================================================
# MinHash LSH over character trigrams (optional, ablation only)
# ============================================================
class MinHashLSH:
    """Banded minhash index: approximate Jaccard lookup over trigram sets."""

    NUM_PERMS = 16
    BANDS = 4          # 4 bands x 4 rows
    ROWS = NUM_PERMS // BANDS
    JACCARD_MIN = 0.5
    _PRIME = (1 << 61) - 1

    def __init__(self, seed: int = 42) -> None:
        rng = random.Random(seed)
        self._a = [rng.randrange(1, self._PRIME) for _ in range(self.NUM_PERMS)]
        self._b = [rng.randrange(0, self._PRIME) for _ in range(self.NUM_PERMS)]
        self._buckets: Dict[Tuple[int, int], List[int]] = {}
        self._words: List[str] = []
        self._gram_sets: List[frozenset] = []

    @staticmethod
    def _grams(word: str) -> frozenset:
        padded = "^" + word + "$"
        return frozenset(padded[i:i + 3] for i in range(len(padded) - 2))

    def _signature(self, grams: frozenset) -> List[int]:
        hashes = [zlib.crc32(g.encode("utf-8")) for g in grams]
        return [min(((a * h + b) % self._PRIME) & 0xFFFFFFFF
                    for h in hashes)
                for a, b in zip(self._a, self._b)]

    def add(self, word: str) -> None:
        grams = self._grams(word)
        if not grams:
            return
        idx = len(self._words)
        self._words.append(word)
        self._gram_sets.append(grams)
        sig = self._signature(grams)
        for band in range(self.BANDS):
            key = (band, hash(tuple(sig[band * self.ROWS:(band + 1) * self.ROWS])))
            self._buckets.setdefault(key, []).append(idx)

    def contains_similar(self, word: str) -> bool:
        grams = self._grams(word)
        if not grams:
            return False
        sig = self._signature(grams)
        seen: set[int] = set()
        for band in range(self.BANDS):
            key = (band, hash(tuple(sig[band * self.ROWS:(band + 1) * self.ROWS])))
            for idx in self._buckets.get(key, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                g2 = self._gram_sets[idx]
                jac = len(grams & g2) / len(grams | g2)
                if jac >= self.JACCARD_MIN:
                    return True
        return False


# ============================================================
# Per-variant evidence components (for debugging / ablation)
# ============================================================
@dataclass
class EvidenceComponents:
    script_lang: Optional[str] = None
    exact_frac: float = 0.0     # incl. prefix credit for the last word
    fuzzy_frac: float = 0.0     # exact + ED<=1 hits
    trigram_plaus: float = 0.0  # calibrated [0,1]
    lsh_frac: float = 0.0       # exact + LSH hits
    evidence: float = 0.0       # combined, vs the claimed language


# ============================================================
# LexicalTriage
# ============================================================
class LexicalTriage:
    """Loads the triage artifacts and scores text variants against languages."""

    def __init__(self,
                 data_dir: str = DEFAULT_DATA_DIR,
                 params: Optional[TriageParams] = None,
                 enable_fuzzy: bool = False,
                 enable_lsh: bool = False,
                 fuzzy_cap: int = 300_000,
                 lsh_cap: int = 150_000,
                 seed: int = 42) -> None:
        import json

        self.params = params or TriageParams()
        self.enable_fuzzy = enable_fuzzy
        self.enable_lsh = enable_lsh
        self._cache: Dict[Tuple[str, str], float] = {}

        self._sets: Dict[str, set] = {}
        self._sorted: Dict[str, List[str]] = {}
        self._trigrams: Dict[str, dict] = {}
        self._bktrees: Dict[str, BKTree] = {}
        self._lsh: Dict[str, MinHashLSH] = {}

        rng = random.Random(seed)
        for lang in ("en", "ru", "he"):
            lex_path = os.path.join(data_dir, f"lexicon_{lang}.txt")
            with open(lex_path, encoding="utf-8") as f:
                words = f.read().split("\n")
            if words and words[-1] == "":
                words.pop()
            self._sorted[lang] = words           # already sorted by builder
            self._sets[lang] = set(words)

            tri_path = os.path.join(data_dir, f"trigrams_{lang}.json")
            with open(tri_path, encoding="utf-8") as f:
                self._trigrams[lang] = json.load(f)

            if enable_fuzzy or enable_lsh:
                p = self.params
                pool = [w for w in words if p.fuzzy_min_len <= len(w) <= 14]
                if enable_fuzzy:
                    sub = (pool if len(pool) <= fuzzy_cap
                           else rng.sample(pool, fuzzy_cap))
                    # Shuffled insertion keeps the BK-tree balanced.
                    rng.shuffle(sub)
                    tree = BKTree()
                    for w in sub:
                        tree.add(w)
                    self._bktrees[lang] = tree
                if enable_lsh:
                    sub = (pool if len(pool) <= lsh_cap
                           else rng.sample(pool, lsh_cap))
                    index = MinHashLSH(seed=seed)
                    for w in sub:
                        index.add(w)
                    self._lsh[lang] = index

    # --------------------------------------------------------
    @staticmethod
    def script_lang(text: str) -> Optional[str]:
        """Language whose script the text is written in, or None if mixed/none."""
        lang = None
        for c in text:
            if not c.isalpha():
                continue
            if "a" <= c <= "z" or "A" <= c <= "Z":
                cur = "en"
            elif "\u0400" <= c <= "\u04FF":
                cur = "ru"
            elif "\u05D0" <= c <= "\u05EA":
                cur = "he"
            else:
                return None
            if lang is None:
                lang = cur
            elif lang != cur:
                return None
        return lang

    # --------------------------------------------------------
    def _normalize(self, text: str, lang: str) -> str:
        t = text.lower()
        if lang == "he":
            t = t.translate(HEBREW_FINALS)
        return t

    def _is_prefix(self, lang: str, prefix: str) -> bool:
        words = self._sorted[lang]
        i = bisect.bisect_left(words, prefix)
        return i < len(words) and words[i].startswith(prefix)

    def _trigram_word_avg(self, lang: str, word: str) -> float:
        table = self._trigrams[lang]
        tri = table["trigrams"]
        floor = table["floor_logprob"]
        padded = "^" + word + "$"
        n = len(padded) - 2
        if n <= 0:
            return floor
        return sum(tri.get(padded[i:i + 3], floor) for i in range(n)) / n

    def _trigram_plausibility(self, lang: str, words: List[str]) -> float:
        """Calibrated [0,1]: logistic over the z-score of the avg logprob."""
        import math
        table = self._trigrams[lang]
        mean, std = table["calib_mean"], table["calib_std"]
        if std <= 0.0 or not words:
            return 0.0
        avg = sum(self._trigram_word_avg(lang, w) for w in words) / len(words)
        z = (avg - mean) / std
        return 1.0 / (1.0 + math.exp(-z))

    # --------------------------------------------------------
    def components(self, text: str, lang: str) -> EvidenceComponents:
        """Full evidence breakdown of *text* interpreted as language *lang*."""
        p = self.params
        comp = EvidenceComponents()
        comp.script_lang = self.script_lang(text)
        if comp.script_lang != lang or lang not in self._sets:
            return comp  # script mismatch: zero evidence by construction

        norm = self._normalize(text, lang)
        words = [w for w in norm.split(" ") if w]
        if not words:
            return comp

        lex = self._sets[lang]
        tree = self._bktrees.get(lang)
        lsh = self._lsh.get(lang)
        exact = fuzzy = lshh = 0
        last = len(words) - 1
        for i, w in enumerate(words):
            is_exact = w in lex
            # The last word is still being typed: prefix of a lexicon word
            # counts as (weaker-side) exact evidence, mirroring the
            # FeedbackLogger prefix-matching approach.
            if not is_exact and i == last and len(w) >= p.min_prefix_len:
                is_exact = self._is_prefix(lang, w)
            if is_exact:
                exact += 1
                fuzzy += 1
                lshh += 1
                continue
            if (tree is not None and len(w) >= p.fuzzy_min_len
                    and tree.contains_within(w, 1)):
                fuzzy += 1
            if (lsh is not None and len(w) >= p.fuzzy_min_len
                    and lsh.contains_similar(w)):
                lshh += 1

        n = len(words)
        comp.exact_frac = exact / n
        comp.fuzzy_frac = fuzzy / n
        comp.lsh_frac = lshh / n
        comp.trigram_plaus = self._trigram_plausibility(lang, words)
        comp.evidence = max(
            p.w_exact * comp.exact_frac,
            p.w_fuzzy * comp.fuzzy_frac if tree is not None else 0.0,
            p.w_trigram * comp.trigram_plaus,
            p.w_lsh * comp.lsh_frac if lsh is not None else 0.0,
        )
        return comp

    def evidence(self, text: str, lang: str) -> float:
        """Combined lexical evidence in [0,1] that *text* is language *lang*."""
        key = (text, lang)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ev = self.components(text, lang).evidence
        self._cache[key] = ev
        return ev

    def clear_cache(self) -> None:
        self._cache.clear()


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    triage = LexicalTriage()
    print(f"loaded in {time.perf_counter() - t0:.1f}s "
          f"(rapidfuzz={_HAVE_RAPIDFUZZ})")
    for text, lang in [("hello world", "en"), ("ghbdtn", "en"),
                       ("привет", "ru"), ("שלום לכולם", "he"),
                       ("hello", "ru"), ("קןקךךם", "he")]:
        c = triage.components(text, lang)
        print(f"{text!r} as {lang}: ev={c.evidence:.3f} "
              f"(exact={c.exact_frac:.2f} fuzzy={c.fuzzy_frac:.2f} "
              f"tri={c.trigram_plaus:.2f} lsh={c.lsh_frac:.2f})")
