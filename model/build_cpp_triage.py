#!/usr/bin/env python3
"""Compile the triage lexicons into a compact DAWG binary for the C++ app.

Reads  model/triage_data/lexicon_{en,ru,he}.txt   (run build_triage_artifacts.py first)
Writes cpp/triage_lexicons.bin                    (loaded by cpp/src/Triage.cpp)

A minimal acyclic DFA (DAWG) is built per language with the incremental
algorithm of Daciuk et al. (2000) for sorted input, then serialized as
CSR-style flat arrays:

    header : magic "KSDAWG01", uint32 langCount
    per lang:
        char[4]  lang code ("en\0\0" ...)
        uint32   stateCount, transCount, wordCount
        uint32   first[stateCount + 1]      transition slice per state (CSR)
        uint8    finalBits[ceil(stateCount/8)]
        uint16   labels[transCount]         (all chars are BMP)
        uint32   targets[transCount]

State 0 is the root. Exact lookup walks labels (binary search per state)
and checks the final bit; prefix lookup only requires consuming all query
characters. The build is self-verified: a path-count DP over the DAWG must
equal the input word count, and membership is sampled against the raw set.

Usage:  python build_cpp_triage.py
"""
from __future__ import annotations

import os
import random
import struct
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "triage_data")
OUT_PATH = os.path.join(SCRIPT_DIR, "..", "cpp", "triage_lexicons.bin")

LANGS = ("en", "ru", "he")
MAGIC = b"KSDAWG01"


class _Node:
    __slots__ = ("edges", "final", "id")
    _next_id = 0

    def __init__(self) -> None:
        self.edges: dict[str, _Node] = {}   # insertion order == sorted order
        self.final = False
        self.id = _Node._next_id
        _Node._next_id += 1


def _replace_or_register(node: _Node, register: dict) -> None:
    """Minimize the right spine (chain of last children) below *node*."""
    chain = []
    n = node
    while n.edges:
        last_char = next(reversed(n.edges))
        child = n.edges[last_char]
        chain.append((n, last_char, child))
        n = child
    for parent, ch, child in reversed(chain):
        sig = (child.final,
               tuple((c, k.id) for c, k in child.edges.items()))
        existing = register.get(sig)
        if existing is not None:
            parent.edges[ch] = existing
        else:
            register[sig] = child


def build_dawg(words: list[str]) -> _Node:
    """Incremental minimal-DAWG construction; *words* must be sorted+unique."""
    root = _Node()
    register: dict = {}
    for w in words:
        node = root
        i = 0
        while i < len(w) and w[i] in node.edges:
            node = node.edges[w[i]]
            i += 1
        if node.edges:
            _replace_or_register(node, register)
        while i < len(w):
            nxt = _Node()
            node.edges[w[i]] = nxt
            node = nxt
            i += 1
        node.final = True
    _replace_or_register(root, register)
    return root


def flatten(root: _Node):
    """BFS state numbering -> (first, finals, labels, targets)."""
    index: dict[int, int] = {root.id: 0}
    order = [root]
    head = 0
    while head < len(order):
        n = order[head]
        head += 1
        for child in n.edges.values():
            if child.id not in index:
                index[child.id] = len(order)
                order.append(child)

    first = [0] * (len(order) + 1)
    finals = [False] * len(order)
    labels: list[int] = []
    targets: list[int] = []
    for si, n in enumerate(order):
        finals[si] = n.final
        first[si] = len(labels)
        for ch, child in n.edges.items():   # already sorted
            cp = ord(ch)
            if cp > 0xFFFF:
                raise ValueError(f"non-BMP char U+{cp:X} in lexicon")
            labels.append(cp)
            targets.append(index[child.id])
    first[len(order)] = len(labels)
    return first, finals, labels, targets


def count_words(first, finals, targets) -> int:
    """DP path count over the DAG == number of distinct accepted words."""
    n_states = len(first) - 1
    memo = [-1] * n_states
    # reverse-topological via iterative DFS
    stack = [(0, False)]
    while stack:
        s, done = stack.pop()
        if memo[s] >= 0:
            continue
        if done:
            total = 1 if finals[s] else 0
            for t in range(first[s], first[s + 1]):
                total += memo[targets[t]]
            memo[s] = total
        else:
            stack.append((s, True))
            for t in range(first[s], first[s + 1]):
                if memo[targets[t]] < 0:
                    stack.append((targets[t], False))
    return memo[0]


def accepts(first, finals, labels, targets, word: str, prefix=False) -> bool:
    """Reference lookup on the flat arrays (mirrors the C++ reader)."""
    import bisect
    s = 0
    for ch in word:
        cp = ord(ch)
        lo, hi = first[s], first[s + 1]
        i = bisect.bisect_left(labels, cp, lo, hi)
        if i >= hi or labels[i] != cp:
            return False
        s = targets[i]
    return True if prefix else finals[s]


def main() -> int:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(LANGS))

    rng = random.Random(42)
    for lang in LANGS:
        path = os.path.join(DATA_DIR, f"lexicon_{lang}.txt")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found — run build_triage_artifacts.py first")
            return 1
        with open(path, encoding="utf-8") as f:
            words = [w for w in f.read().split("\n") if w]
        words = sorted(set(words))

        t0 = time.perf_counter()
        root = build_dawg(words)
        first, finals, labels, targets = flatten(root)
        build_s = time.perf_counter() - t0

        # ---- verification ----
        n = count_words(first, finals, targets)
        assert n == len(words), f"{lang}: DAWG word count {n} != {len(words)}"
        wset = set(words)
        sample = words if len(words) <= 30000 else rng.sample(words, 30000)
        for w in sample:
            assert accepts(first, finals, labels, targets, w), f"{lang}: missing {w!r}"
            assert accepts(first, finals, labels, targets, w[: max(1, len(w) - 1)],
                           prefix=True), f"{lang}: prefix walk failed for {w!r}"
        neg = 0
        for w in sample[:10000]:
            m = w[:-1] + chr(ord(w[-1]) + 1)   # mutate last char
            if m not in wset:
                neg += 1
                assert not accepts(first, finals, labels, targets, m), \
                    f"{lang}: false accept {m!r}"
        print(f"{lang}: {len(words):>9} words -> {len(finals):>7} states, "
              f"{len(labels):>8} transitions  (build {build_s:.1f}s, "
              f"verified {len(sample)} pos / {neg} neg)")

        # ---- serialize ----
        code = lang.encode("ascii").ljust(4, b"\0")
        out += code
        out += struct.pack("<III", len(finals), len(labels), len(words))
        out += struct.pack(f"<{len(first)}I", *first)
        fb = bytearray((len(finals) + 7) // 8)
        for i, f_ in enumerate(finals):
            if f_:
                fb[i >> 3] |= 1 << (i & 7)
        out += fb
        out += struct.pack(f"<{len(labels)}H", *labels)
        out += struct.pack(f"<{len(targets)}I", *targets)

    out_path = os.path.normpath(OUT_PATH)
    with open(out_path, "wb") as f:
        f.write(out)
    print(f"\nwrote {out_path}  ({len(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
