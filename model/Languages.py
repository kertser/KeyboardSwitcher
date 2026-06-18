# This library is a language toolset for the project.

# import the necessary packages
import onnxruntime as ort
import json
import pickle
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Keyboard layout strings
# ============================================================
russian_layout = 'ёйцукенгшщзхъфывапролджэ\ячсмитьбю.ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭ/ЯЧСМИТЬБЮ,'
english_layout = '''`qwertyuiop[]asdfghjkl;'\zxcvbnm,./~QWERTYUIOP{}ASDFGHJKL:"|ZXCVBNM<>?'''
hebrew_layout = ";/'קראטוןםפ][שדגכעיחלךף,\זסבהנמצתץ."
special_characters = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
numbers = "0123456789"


def get_layout_for_language(lang: str) -> str:
    """Get the keyboard layout string for a language code."""
    if lang == 'ru':
        return russian_layout
    if lang == 'he':
        return hebrew_layout
    return english_layout


# ============================================================
# Detection result (mirrors C++ DetectionResult struct)
# ============================================================
@dataclass
class DetectionResult:
    language: str       # "en", "he", "ru"
    confidence: float   # softmax probability of the top class
    scores: list        # raw softmax probabilities [N/A, en, he, ru]


# ============================================================
# Model loading (prefers dictionary.json, falls back to .pkl)
# ============================================================
def load_model():
    """Load the ONNX model and character dictionary."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    onnx_model_path = os.path.join(current_directory, "lang_model.onnx")
    ort_session = ort.InferenceSession(onnx_model_path)

    # Prefer dictionary.json (same as C++ build), fall back to .pkl
    json_path = os.path.join(current_directory, "dictionary.json")
    pkl_path = os.path.join(current_directory, "dictionary.pkl")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            char_to_index = {k: int(v) for k, v in raw.items()}
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            char_to_index = pickle.load(f)
    else:
        raise FileNotFoundError("Neither dictionary.json nor dictionary.pkl found")

    return [ort_session, char_to_index, 45]


# ============================================================
# Class-index → language string helpers
# ============================================================
_CLASS_TO_LANG = {0: None, 1: "en", 2: "he", 3: "ru"}


def _class_to_language(cls: int) -> Optional[str]:
    return _CLASS_TO_LANG.get(cls)


# ============================================================
# Simple prediction (kept for backward compatibility)
# ============================================================
def predict_language(text, ort_session, char_to_index, max_length):
    """AI-assisted language detection – returns language string or None."""
    input_indices = [char_to_index.get(char, 0) for char in text if char in char_to_index]

    # Padding / truncation
    if len(input_indices) < max_length:
        input_indices += [0] * (max_length - len(input_indices))
    input_indices = input_indices[:max_length]

    input_tensor = np.array(input_indices, dtype=np.int64).reshape(1, -1)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)

    output_probs = ort_outputs[0]
    predicted_class = int(np.argmax(output_probs))

    return _class_to_language(predicted_class)


# ============================================================
# Prediction with confidence (mirrors C++ PredictLanguageWithConfidence)
# ============================================================
# Process-wide memoization of inference results.  The ONNX model is fixed for
# the lifetime of a process, so the softmax output for a given input string is
# constant and param-independent.  The offline tuning harnesses (especially
# recalibrate.py, which sweeps gate parameters over a large grid) call this
# function on the same variant strings thousands of times; caching turns the
# dominant cost (ONNX inference) into a one-time-per-string expense and makes
# the grid sweep effectively free after the first config.
_PREDICT_CACHE: dict = {}


def clear_predict_cache() -> None:
    """Drop the memoized inference results (call after swapping the model)."""
    _PREDICT_CACHE.clear()


def predict_language_with_confidence(
    text, ort_session, char_to_index, max_length
) -> Optional[DetectionResult]:
    """Run inference and return a DetectionResult with softmax scores,
    or None if N/A has the highest probability."""

    cached = _PREDICT_CACHE.get(text, False)
    if cached is not False:
        return cached

    result = _predict_language_with_confidence_uncached(
        text, ort_session, char_to_index, max_length)
    _PREDICT_CACHE[text] = result
    return result


def _encode(text, char_to_index, max_length):
    """Mirror the production tokenizer: drop OOV chars, pad/truncate to max_length."""
    idx = [char_to_index.get(c, 0) for c in text if c in char_to_index]
    if len(idx) < max_length:
        idx += [0] * (max_length - len(idx))
    return idx[:max_length]


def _scores_to_result(scores) -> Optional[DetectionResult]:
    best_class, best_prob = 0, 0.0
    for i in range(1, 4):
        if scores[i] > best_prob:
            best_prob = scores[i]
            best_class = i
    if scores[0] > best_prob:
        return None
    lang = _class_to_language(best_class)
    if lang is None:
        return None
    return DetectionResult(language=lang, confidence=best_prob, scores=scores)


def prewarm_cache(texts, ort_session, char_to_index, max_length, batch_size=1024) -> int:
    """Batch-infer every string in *texts* once and populate the predict cache.

    The offline tuning harnesses query the same variant strings repeatedly across
    a large parameter grid.  Running ONNX one row at a time dominates the runtime;
    batching hundreds of rows per session.run() call is an order of magnitude
    faster.  Returns the number of newly inferred (previously uncached) strings.
    """
    pending = []
    seen = set()
    for t in texts:
        if t in _PREDICT_CACHE or t in seen:
            continue
        seen.add(t)
        pending.append(t)

    if not pending:
        return 0

    input_name = ort_session.get_inputs()[0].name
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch = np.array([_encode(t, char_to_index, max_length) for t in chunk],
                         dtype=np.int64)
        logits = ort_session.run(None, {input_name: batch})[0]
        logits = logits[:, :4]
        m = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - m)
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        for t, row in zip(chunk, softmax):
            _PREDICT_CACHE[t] = _scores_to_result(row.tolist())
    return len(pending)


def _predict_language_with_confidence_uncached(
    text, ort_session, char_to_index, max_length
) -> Optional[DetectionResult]:
    input_indices = [char_to_index.get(char, 0) for char in text if char in char_to_index]

    # Padding / truncation
    if len(input_indices) < max_length:
        input_indices += [0] * (max_length - len(input_indices))
    input_indices = input_indices[:max_length]

    input_tensor = np.array(input_indices, dtype=np.int64).reshape(1, -1)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)

    logits = ort_outputs[0].flatten()
    if len(logits) < 4:
        return None

    # Softmax (numerically stable)
    max_logit = float(np.max(logits[:4]))
    exps = np.exp(logits[:4] - max_logit)
    softmax_scores = exps / np.sum(exps)
    scores = softmax_scores.tolist()

    # Argmax over classes 1-3 (skip N/A = class 0)
    best_class = 0
    best_prob = 0.0
    for i in range(1, 4):
        if scores[i] > best_prob:
            best_prob = scores[i]
            best_class = i

    # If N/A has the highest probability, return None
    if scores[0] > best_prob:
        return None

    lang = _class_to_language(best_class)
    if lang is None:
        return None

    return DetectionResult(language=lang, confidence=best_prob, scores=scores)


# ============================================================
# Layout conversion functions
# ============================================================
def create_conversion_map(source_layout, target_layout):
    """Create a character conversion map between two keyboard layouts."""
    conversion_map = {}
    for src_char, tgt_char in zip(source_layout, target_layout):
        conversion_map[ord(src_char)] = ord(tgt_char)
    return conversion_map


def convert_text(text, conversion_map):
    """Convert text using a pre-built conversion map."""
    return text.translate(conversion_map)


def convert_text_bidirectional(text, from_layout, to_layout):
    """Convert text from one keyboard layout to another."""
    if to_layout == hebrew_layout:
        text = text.lower()
    return convert_text(text, create_conversion_map(from_layout, to_layout))



if __name__ == '__main__':
    # Example
    input_text = "hello"
    args = load_model()

    predicted_language = predict_language(input_text, *args)
    print("Predicted language:", predicted_language)

    result = predict_language_with_confidence(input_text, *args)
    if result:
        print(f"Language: {result.language}, Confidence: {result.confidence:.4f}")
        print(f"Scores: {result.scores}")
