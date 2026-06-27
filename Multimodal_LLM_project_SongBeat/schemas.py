"""Typed data structures shared across the pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, TypedDict, Any


@dataclass
class AudioEmotion:
    """Audio-derived emotion in valence-arousal space (each axis in [-1, 1])."""
    valence: float
    arousal: float
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LyricEmotion:
    """Lyric-derived emotion in valence-arousal space plus the dominant label."""
    valence: float
    arousal: float
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    backend: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentResult:
    distance: float            # euclidean distance in VA space
    score: float               # 0..1 alignment score (1 = perfectly aligned)
    label: str                 # aligned | contrast/ironic | partial-mismatch
    explanation: str
    audio: AudioEmotion
    lyric: LyricEmotion
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class PipelineState(TypedDict, total=False):
    """State object threaded through the LangGraph DAG."""
    audio_path: Optional[str]
    lyrics: str
    title: str
    audio_emotion: AudioEmotion
    lyric_emotion: LyricEmotion
    retrieved_context: list[dict[str, Any]]
    result: AlignmentResult
