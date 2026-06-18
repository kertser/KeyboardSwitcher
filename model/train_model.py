#!/usr/bin/env python3
"""train_model.py — full retraining pipeline for the v2 BiLSTM language model.

What changed vs. the original LangModel.ipynb pipeline
------------------------------------------------------
1. Dedicated PAD token at index 0 (no real character maps to 0).  In the v1
   dictionary index 0 was the real letter "м", so padding injected phantom
   "м"s.  Here real chars are indexed 1..N and PAD=0 is masked out.
2. The model (Languages_torch.LanguageClassifier) is a BiLSTM with masked
   mean+max pooling instead of reading out[:, -1, :] after the padding.
3. Data augmentation to match the INFERENCE distribution, which v1 never saw:
     • PREFIXES   — every real word also emits truncated prefixes with the same
                    label, so partial words (every keystroke) are in-distribution.
     • PHRASES    — multi-word same-language concatenations labelled by language,
                    so Hebrew/Russian phrases (the FN cases) are in-distribution.
     • NEGATIVES  — each real word's wrong-layout renderings are labelled N/A
                    (0), exactly as v1 did, plus optional prefix negatives.
4. Class balancing (sample equal words per language; cap the N/A class) and
   label smoothing for better-calibrated softmax confidences (the adaptive
   thresholds depend on confidence being meaningful).

Outputs (written into this directory, then copy lang_model.onnx + dictionary.json
into ../cpp/):
    dictionary.json   real chars -> 1..N   (PAD=0 implicit)
    dictionary.pkl    same, pickled
    lang_model.pth    best checkpoint
    lang_model.onnx   exported model (dynamic seq axis)

Usage (on the GPU box):
    pip install torch numpy
    python train_model.py --words-per-lang 300000 --epochs 8
    # then:  python convert_to_onnx.py     (or rely on --export, on by default)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Languages_torch import (
    LanguageClassifier, PAD_IDX,
    english_layout, russian_layout, hebrew_layout,
    numbers, special_characters,
    convert_text_bidirectional,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB = {
    "en": os.path.join(SCRIPT_DIR, "vocabulary", "english_vocabulary"),
    "ru": os.path.join(SCRIPT_DIR, "vocabulary", "russian_vocabulary"),
    "he": os.path.join(SCRIPT_DIR, "vocabulary", "hebrew_vocabulary"),
}
LAYOUT = {"en": english_layout, "ru": russian_layout, "he": hebrew_layout}
LANG_TO_TARGET = {"en": 1, "he": 2, "ru": 3}


# ---------------------------------------------------------------------------
# Vocabulary / dictionary
# ---------------------------------------------------------------------------
def build_char_index():
    """Deterministic char->index with PAD reserved at 0; real chars 1..N.

    Built from the union of the three layouts + digits + punctuation so it
    matches the v1 coverage exactly (same characters, different indices).
    """
    combined = english_layout + hebrew_layout + russian_layout + numbers + special_characters
    seen, ordered = set(), []
    for ch in combined:
        if ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    # Reserve 0 for PAD; start real chars at 1.
    return {ch: i + 1 for i, ch in enumerate(ordered)}


def load_words(path, min_len=1, max_len=20):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if min_len <= len(w) <= max_len and w.isalpha():
                out.append(w)
    return out


# ---------------------------------------------------------------------------
# Dataset construction with augmentation
# ---------------------------------------------------------------------------
def build_samples(args, rng):
    """Return a list of (text, target) tuples.

    target: 1=en, 2=he, 3=ru, 0=N/A (garbled / partial-garble / unknown).
    """
    samples: list[tuple[str, int]] = []
    words = {lang: load_words(VOCAB[lang]) for lang in LANG_TO_TARGET}

    # Balance: equal number of source words per language.
    n_per = min(args.words_per_lang, *(len(words[l]) for l in words))
    for lang in words:
        words[lang] = rng.sample(words[lang], n_per)
    print(f"  using {n_per} words/lang")

    def emit_prefixes(text, target, prob):
        # min prefix length 2 so we don't train on single characters.
        for L in range(2, len(text)):
            if rng.random() < prob:
                samples.append((text[:L], target))

    for lang, target in LANG_TO_TARGET.items():
        src_layout = LAYOUT[lang]
        for w in words[lang]:
            # POSITIVE: the real word on its own layout.
            samples.append((w, target))
            emit_prefixes(w, target, args.prefix_prob)

            # NEGATIVES: the same word typed on the two OTHER layouts → garbled,
            # labelled N/A (this is what teaches "garbled cross-layout = not yet
            # a language", preventing false switches).  Down-sample to balance.
            for other, ol in LAYOUT.items():
                if other == lang:
                    continue
                garbled = convert_text_bidirectional(w, src_layout, ol)
                if garbled == w:
                    continue
                if rng.random() < args.na_keep_prob:
                    samples.append((garbled, 0))
                    emit_prefixes(garbled, 0, args.prefix_prob * args.na_keep_prob)

    # PHRASES: multi-word, same language → positive; wrong-layout → N/A.
    n_phrase = int(n_per * args.phrase_ratio)
    for lang, target in LANG_TO_TARGET.items():
        pool = words[lang]
        src_layout = LAYOUT[lang]
        for _ in range(n_phrase):
            k = rng.choice([2, 3])
            phrase = " ".join(rng.sample(pool, k))
            samples.append((phrase, target))
            emit_prefixes(phrase, target, args.prefix_prob * 0.5)
            # one wrong-layout negative per phrase
            other = rng.choice([l for l in LAYOUT if l != lang])
            garbled = convert_text_bidirectional(phrase, src_layout, LAYOUT[other])
            if garbled != phrase and rng.random() < args.na_keep_prob:
                samples.append((garbled, 0))

    rng.shuffle(samples)
    return samples


class CharDataset(Dataset):
    def __init__(self, samples, char_to_index, max_length):
        self.samples = samples
        self.cti = char_to_index
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, target = self.samples[idx]
        ids = [self.cti.get(c, PAD_IDX) for c in text if c in self.cti]
        ids = ids[: self.max_length]
        if len(ids) < self.max_length:
            ids += [PAD_IDX] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long), target


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    # per-class recall (0=N/A,1=en,2=he,3=ru)
    cls_correct = [0, 0, 0, 0]
    cls_total = [0, 0, 0, 0]
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += criterion(out, yb).item()
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        for c in range(4):
            m = yb == c
            cls_total[c] += m.sum().item()
            cls_correct[c] += (pred[m] == c).sum().item()
    acc = 100.0 * correct / max(total, 1)
    recalls = [100.0 * cls_correct[c] / cls_total[c] if cls_total[c] else 0.0
               for c in range(4)]
    return loss_sum / max(len(loader), 1), acc, recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words-per-lang", type=int, default=300000)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--prefix-prob", type=float, default=0.25,
                    help="probability of emitting each prefix length")
    ap.add_argument("--na-keep-prob", type=float, default=0.5,
                    help="down-sample factor for N/A (garbled) negatives")
    ap.add_argument("--phrase-ratio", type=float, default=0.3,
                    help="phrases per language = ratio * words_per_lang")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    # 1) dictionary
    print("[1/5] Building dictionary (PAD=0 reserved) ...")
    char_to_index = build_char_index()
    with open(os.path.join(SCRIPT_DIR, "dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(char_to_index, f, ensure_ascii=False)
    with open(os.path.join(SCRIPT_DIR, "dictionary.pkl"), "wb") as f:
        pickle.dump(char_to_index, f)
    num_embeddings = max(char_to_index.values()) + 1
    print(f"      {len(char_to_index)} real chars, num_embeddings={num_embeddings}")

    # 2) samples
    print("[2/5] Building augmented samples ...")
    t0 = time.time()
    samples = build_samples(args, rng)
    print(f"      {len(samples)} samples in {time.time()-t0:.0f}s")
    dist = [0, 0, 0, 0]
    for _, t in samples:
        dist[t] += 1
    print(f"      class distribution N/A/en/he/ru = {dist}")

    # 3) split
    n_val = int(len(samples) * args.val_frac)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    train_ds = CharDataset(train_samples, char_to_index, args.max_length)
    val_ds = CharDataset(val_samples, char_to_index, args.max_length)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=pin)

    # 4) model / train
    print("[3/5] Building model ...")
    model = LanguageClassifier(num_embeddings, args.embed, args.hidden,
                               args.layers, 4, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                     factor=0.5, patience=1)

    print("[4/5] Training ...")
    best_val = float("inf")
    bad = 0
    best_path = os.path.join(SCRIPT_DIR, "lang_model.pth")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for bi, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if bi % 200 == 0:
                print(f"  epoch {epoch} batch {bi}/{len(train_loader)} "
                      f"loss {loss.item():.4f}", end="\r")
        vloss, vacc, vrec = evaluate(model, val_loader, criterion, device)
        scheduler.step(vloss)
        print(f"\n  epoch {epoch}: train_loss {running/len(train_loader):.4f} "
              f"val_loss {vloss:.4f} val_acc {vacc:.2f}%  "
              f"recall N/A {vrec[0]:.1f} en {vrec[1]:.1f} he {vrec[2]:.1f} ru {vrec[3]:.1f}")
        if vloss < best_val - 1e-4:
            best_val = vloss
            bad = 0
            torch.save(model.state_dict(), best_path)
            print(f"      ✓ saved best ({best_path})")
        else:
            bad += 1
            if bad >= args.patience:
                print("      early stopping.")
                break

    # 5) export
    if not args.no_export:
        print("[5/5] Exporting ONNX ...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        export_onnx(model, char_to_index, device)
    print("Done. Copy lang_model.onnx + dictionary.json into ../cpp/ and rebuild.")


def export_onnx(model, char_to_index, device, onnx_path=None):
    onnx_path = onnx_path or os.path.join(SCRIPT_DIR, "lang_model.onnx")
    model.eval().to(device)
    dummy = torch.zeros(1, 16, dtype=torch.long, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"      wrote {onnx_path}")


if __name__ == "__main__":
    main()

