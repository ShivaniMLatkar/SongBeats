"""Audio feature extraction -> valence/arousal proxies, using librosa only.

This replaces the (now deprecated, Nov 2024) Spotify audio-features endpoint by
computing features locally:

  arousal  ~ energy (RMS), tempo, spectral centroid, zero-crossing rate
  valence  ~ mode (major/minor via chroma), harmonic ratio, brightness, tempo

Valence is the genuinely hard axis in MIR; the heuristic here is intentionally
transparent and tunable rather than a black box. Each axis is mapped to [-1, 1].
"""
from __future__ import annotations
import numpy as np

from config import SETTINGS
from schemas import AudioEmotion

# Krumhansl major/minor key profiles - used to estimate mode (major=bright/positive).
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _squash(x: float) -> float:
    """Map an arbitrary real to [-1, 1] smoothly."""
    return float(np.tanh(x))


def _norm(value: float, lo: float, hi: float) -> float:
    """Linear normalise to [-1, 1] given an expected [lo, hi] range, clipped."""
    if hi == lo:
        return 0.0
    z = (value - lo) / (hi - lo)  # 0..1
    return float(np.clip(z * 2 - 1, -1, 1))


def _estimate_mode(chroma: np.ndarray) -> float:
    """Return correlation-based major-vs-minor score in [-1, 1] (major positive)."""
    profile = chroma.mean(axis=1)
    if profile.sum() == 0:
        return 0.0
    profile = profile / profile.sum()
    best_major = max(np.corrcoef(np.roll(profile, -k), _MAJOR)[0, 1] for k in range(12))
    best_minor = max(np.corrcoef(np.roll(profile, -k), _MINOR)[0, 1] for k in range(12))
    return float(np.clip(best_major - best_minor, -1, 1))


def extract(audio_path: str) -> AudioEmotion:
    """Extract emotion proxies from an audio file. Raises if librosa unavailable."""
    import librosa  # imported lazily so the rest of the package loads without it

    y, sr = librosa.load(audio_path, sr=SETTINGS.sample_rate,
                         duration=SETTINGS.max_audio_seconds, mono=True)
    if y.size == 0:
        raise ValueError(f"No audio samples loaded from {audio_path}")

    rms = float(np.mean(librosa.feature.rms(y=y)))
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mode = _estimate_mode(chroma)

    # Harmonic-to-percussive ratio: more harmonic content tends to read as warmer/positive.
    y_h, y_p = librosa.effects.hpss(y)
    h_energy = float(np.mean(y_h ** 2)) + 1e-9
    p_energy = float(np.mean(y_p ** 2)) + 1e-9
    harmonic_ratio = h_energy / (h_energy + p_energy)  # 0..1

    features = {
        "rms_energy": rms,
        "tempo_bpm": tempo,
        "spectral_centroid": centroid,
        "zero_crossing_rate": zcr,
        "mode_score": mode,
        "harmonic_ratio": harmonic_ratio,
    }

    # --- Arousal: physical energy / activity of the signal ---
    arousal = float(np.mean([
        _norm(rms, 0.0, 0.2),
        _norm(tempo, 60.0, 180.0),
        _norm(centroid, 800.0, 4000.0),
        _norm(zcr, 0.02, 0.20),
    ]))

    # --- Valence: brightness + mode + harmonicity + mild tempo lift ---
    valence = float(np.mean([
        mode,                                   # major vs minor, already [-1,1]
        _norm(harmonic_ratio, 0.3, 0.8),
        _norm(centroid, 1000.0, 3500.0) * 0.5,  # brightness, down-weighted
        _norm(tempo, 70.0, 140.0) * 0.5,
    ]))

    return AudioEmotion(valence=_squash(valence), arousal=_squash(arousal), features=features)
