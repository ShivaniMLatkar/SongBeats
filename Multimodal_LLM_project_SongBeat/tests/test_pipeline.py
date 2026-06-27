"""End-to-end and unit tests that run with fallbacks only (no model downloads).

Run from the project root:
    python -m pytest -q
or without pytest installed:
    python tests/test_pipeline.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import AudioEmotion
from src import fusion, lyric_sentiment
from src.rag_context import ReferenceRetriever
from src.graph import analyze_song


def test_lyric_sentiment_polarity():
    happy = lyric_sentiment.analyze("I'm so happy, joy and love, smile and shine all day")
    sad = lyric_sentiment.analyze("alone, lost, pain, crying, dark and broken")
    assert happy.valence > sad.valence


def test_fusion_detects_contrast():
    upbeat_audio = AudioEmotion(valence=0.8, arousal=0.7)
    dark_lyric = lyric_sentiment.analyze("alone, lost, pain, crying, dark and broken, gun and danger")
    result = fusion.fuse(upbeat_audio, dark_lyric)
    assert result.label in {"contrast/ironic", "partial-mismatch"}
    assert 0.0 <= result.score <= 1.0


def test_fusion_detects_alignment():
    happy_audio = AudioEmotion(valence=0.8, arousal=0.6)
    happy_lyric = lyric_sentiment.analyze("happy joy love smile bright sunshine dancing")
    result = fusion.fuse(happy_audio, happy_lyric)
    assert result.score > 0.6


def test_rag_retrieves_relevant():
    r = ReferenceRetriever()
    hits = r.retrieve("anger rage fight burn it down fury", k=2)
    assert len(hits) == 2
    assert any("angry" in h.get("mood", "") for h in hits)


def test_pipeline_no_audio():
    res = analyze_song(None, "happy joy love smile bright sunshine", title="t")
    assert res.lyric.valence > 0
    assert res.retrieved_context


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
