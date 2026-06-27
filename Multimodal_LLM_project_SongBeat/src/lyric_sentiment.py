"""Lyric sentiment / emotion -> valence-arousal, using a LOCAL open model.

Primary backend: a small HuggingFace emotion classifier (transformers, runs
locally, no API key). Fallback: VADER lexicon (pure Python, zero download) so
the pipeline always produces a result.

Emotion labels are mapped to valence-arousal coordinates following the standard
circumplex model of affect (Russell, 1980).
"""
from __future__ import annotations
import numpy as np

from config import SETTINGS
from schemas import LyricEmotion

# Circumplex coordinates (valence, arousal) in [-1, 1] for common emotion labels.
_EMOTION_VA: dict[str, tuple[float, float]] = {
    "joy":      (0.85, 0.55),
    "love":     (0.80, 0.20),
    "surprise": (0.30, 0.75),
    "neutral":  (0.00, 0.00),
    "sadness":  (-0.75, -0.55),
    "fear":     (-0.65, 0.70),
    "anger":    (-0.70, 0.75),
    "disgust":  (-0.70, 0.35),
}

_classifier = None  # lazy singleton


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    try:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model=SETTINGS.emotion_model,
            top_k=None,
            truncation=True,
        )
    except Exception:
        _classifier = False  # mark as unavailable; trigger fallback
    return _classifier


def _from_transformer(text: str) -> LyricEmotion | None:
    clf = _load_classifier()
    if not clf:
        return None
    try:
        raw = clf(text[:2000])
        rows = raw[0] if isinstance(raw[0], list) else raw
        scores = {r["label"].lower(): float(r["score"]) for r in rows}
    except Exception:
        return None

    # Expected-value over the circumplex, weighted by class probability.
    v = a = 0.0
    for label, p in scores.items():
        cv, ca = _EMOTION_VA.get(label, (0.0, 0.0))
        v += p * cv
        a += p * ca
    dominant = max(scores, key=scores.get)
    return LyricEmotion(valence=float(np.clip(v, -1, 1)),
                        arousal=float(np.clip(a, -1, 1)),
                        label=dominant, scores=scores, backend="transformers")


def _from_vader(text: str) -> LyricEmotion:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        s = sia.polarity_scores(text)
        valence = float(s["compound"])               # -1..1
        arousal = float(np.clip((s["pos"] + s["neg"]) * 2 - 1, -1, 1))
        scores = s
    except Exception:
        # Last-resort tiny lexicon so tests run with zero third-party deps.
        pos = sum(w in text.lower() for w in
                  ("love", "happy", "joy", "smile", "bright", "dance", "shine", "good"))
        neg = sum(w in text.lower() for w in
                  ("sad", "cry", "pain", "alone", "dark", "hate", "lost", "tears"))
        total = pos + neg or 1
        valence = (pos - neg) / total
        arousal = float(np.clip((pos + neg) / 8.0 * 2 - 1, -1, 1))
        scores = {"pos": pos, "neg": neg}
    label = "positive" if valence > 0.15 else "negative" if valence < -0.15 else "neutral"
    return LyricEmotion(valence=valence, arousal=arousal, label=label,
                        scores=scores, backend="vader")


def analyze(text: str) -> LyricEmotion:
    """Analyse lyric text; prefer the local transformer, fall back to VADER."""
    if not text or not text.strip():
        return LyricEmotion(0.0, 0.0, "neutral", {}, "empty")
    return _from_transformer(text) or _from_vader(text)
