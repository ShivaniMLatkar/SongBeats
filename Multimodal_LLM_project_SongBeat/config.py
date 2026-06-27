"""Central configuration for the SongBeat / SonicSentic pipeline.

Everything here is plain Python so it can be edited without touching logic.
All model choices default to LOCAL / OPEN models and degrade gracefully when a
dependency or download is unavailable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_SONGS = DATA_DIR / "reference_songs.json"
SAMPLES_DIR = DATA_DIR / "samples"


@dataclass
class Settings:
    # --- Lyric sentiment ---
    # A small, local HF emotion classifier. If it can't be loaded, we fall back
    # to the VADER lexicon (no download, pure Python).
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"

    # --- Fusion explanation LLM (local / open) ---
    # A small seq2seq model used only to phrase the natural-language verdict.
    # If unavailable, a deterministic template is used instead.
    explanation_model: str = "google/flan-t5-small"
    use_llm_explanation: bool = True

    # --- RAG ---
    rag_top_k: int = 3
    # Chroma's default embedding is used when chromadb is installed; otherwise a
    # built-in TF-IDF vector store provides the same retrieve() interface.
    collection_name: str = "songbeat_reference"

    # --- Audio ---
    sample_rate: int = 22050
    max_audio_seconds: float = 60.0  # analyse at most this many seconds

    # --- Alignment thresholds (valence-arousal space, each axis in [-1, 1]) ---
    aligned_distance: float = 0.5      # <= this => "aligned"
    mismatch_valence_gap: float = 0.6  # opposite-sign valence gap => "ironic/contrast"


SETTINGS = Settings()
