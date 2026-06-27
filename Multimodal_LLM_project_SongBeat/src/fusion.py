"""Decision-level fusion of audio and lyric emotion + natural-language verdict.

The two modalities are compared in the shared valence-arousal plane. We report a
distance, a 0..1 alignment score, a categorical label, and an explanation. The
explanation is phrased by a small LOCAL seq2seq model (flan-t5-small) when
available, conditioned on the retrieved reference exemplars; otherwise a
deterministic template is used.
"""
from __future__ import annotations
import math

from config import SETTINGS
from schemas import AudioEmotion, LyricEmotion, AlignmentResult

_MAX_DIST = math.sqrt(8.0)  # max euclidean distance in [-1,1]^2 space
_llm = None


def _classify(audio: AudioEmotion, lyric: LyricEmotion, distance: float) -> str:
    valence_gap = abs(audio.valence - lyric.valence)
    opposite = (audio.valence * lyric.valence) < 0
    if opposite and valence_gap >= SETTINGS.mismatch_valence_gap:
        return "contrast/ironic"          # e.g. upbeat music, sorrowful words
    if distance <= SETTINGS.aligned_distance:
        return "aligned"
    return "partial-mismatch"


def _load_llm():
    global _llm
    if _llm is not None or not SETTINGS.use_llm_explanation:
        return _llm
    try:
        from transformers import pipeline
        _llm = pipeline("text2text-generation", model=SETTINGS.explanation_model)
    except Exception:
        _llm = False
    return _llm


def _template_explanation(audio, lyric, label, score) -> str:
    mood = {"aligned": "reinforce each other",
            "contrast/ironic": "pull in opposite directions",
            "partial-mismatch": "only partly agree"}[label]
    return (
        f"The music (valence {audio.valence:+.2f}, arousal {audio.arousal:+.2f}) and the "
        f"lyrics (valence {lyric.valence:+.2f}, arousal {lyric.arousal:+.2f}, dominant "
        f"emotion '{lyric.label}') {mood}. Alignment score {score:.2f} -> {label}."
    )


def _llm_explanation(audio, lyric, label, score, context) -> str | None:
    llm = _load_llm()
    if not llm:
        return None
    ctx = "; ".join(f"{c.get('title','?')} ({c.get('mood','?')})" for c in context[:3])
    prompt = (
        "Explain in 2 sentences whether a song's music and lyrics express the same "
        "emotion.\n"
        f"Music valence={audio.valence:+.2f} arousal={audio.arousal:+.2f}.\n"
        f"Lyrics valence={lyric.valence:+.2f} arousal={lyric.arousal:+.2f} "
        f"emotion={lyric.label}.\n"
        f"Verdict={label}, score={score:.2f}. Similar reference songs: {ctx}."
    )
    try:
        out = llm(prompt, max_new_tokens=80)[0]["generated_text"].strip()
        return out or None
    except Exception:
        return None


def fuse(audio: AudioEmotion, lyric: LyricEmotion,
         context: list[dict] | None = None) -> AlignmentResult:
    context = context or []
    distance = math.sqrt((audio.valence - lyric.valence) ** 2 +
                         (audio.arousal - lyric.arousal) ** 2)
    score = max(0.0, 1.0 - distance / _MAX_DIST)
    label = _classify(audio, lyric, distance)
    explanation = (_llm_explanation(audio, lyric, label, score, context)
                   or _template_explanation(audio, lyric, label, score))
    return AlignmentResult(distance=round(distance, 4), score=round(score, 4),
                           label=label, explanation=explanation,
                           audio=audio, lyric=lyric, retrieved_context=context)
