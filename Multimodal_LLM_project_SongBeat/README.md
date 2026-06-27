# SongBeat — Audio↔Lyric Emotional Alignment

A multimodal pipeline that measures whether a song's **music** and its **lyrics**
express the same emotion, or pull against each other (the classic "happy melody,
dark words" effect). Audio and lyrics are each projected into a shared
**valence–arousal** plane and compared.

This is a re-grounded, working implementation of the *SonicSentic* concept. It
keeps the parts that earn their place (LangGraph orchestration, RAG calibration)
and drops the parts that were résumé padding (autonomous agents, multi-provider
LLMs). See **Validity notes** below.

## Validity notes (read before reusing the original description)

| Original claim | Reality | What this repo does |
|---|---|---|
| Audio features (Spotify) | Spotify **deprecated** `audio-features`/`audio-analysis` on 2024-11-27; no new-app access, no official replacement. | Extracts features **locally with librosa** → valence/arousal. No external API. |
| "Autonomous agent orchestration" | Overkill — the task is a deterministic 3-step pipeline. Agents add cost/latency/nondeterminism. | **LangGraph as a defined DAG**, not an agent loop. |
| "RAG / vector DB" | Only meaningful if it retrieves real context. | RAG retrieves **emotionally-similar reference songs** to calibrate the fusion verdict (few-shot context). |
| "Multimodal" | The LLM only sees text. | Honest **late (decision-level) fusion** of two unimodal signals in V-A space. |
| "Multiple LLM providers" | Unnecessary. | **One local/open model** (no API keys); graceful fallbacks. |
| "Accuracy vs human benchmarks" | Needs labelled ground truth. | Hook provided; pair with DEAM (audio V-A) or MoodyLyrics (lyric mood). |

## Architecture

```
ingest → audio_node → lyric_node → rag_node → fusion_node     (LangGraph DAG)
            │             │            │            │
        librosa      HF emotion     Chroma      V-A distance +
        features    model / VADER  reference   local LLM verdict
        → V/A        → V/A          songs
```

Every stage degrades gracefully so the pipeline always runs:

- **Lyrics:** HF emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) → falls back to **VADER** → falls back to a tiny built-in lexicon.
- **RAG:** **Chroma** vector DB → falls back to a built-in **TF-IDF** cosine store.
- **Orchestration:** **LangGraph** `StateGraph` → falls back to a plain function pipeline (identical behaviour).
- **Verdict phrasing:** local **flan-t5-small** → falls back to a deterministic template.

> The fallbacks let you run with zero downloads; install the full
> `requirements.txt` for best accuracy (the HF emotion model is much stronger
> than VADER on figurative lyrics).

## Install & run

```bash
pip install -r requirements.txt          # full stack (recommended)
# minimal: pip install numpy librosa soundfile vaderSentiment

# analyse a song (audio + lyrics file)
python run.py --audio data/samples/demo_audio.wav --lyrics data/samples/demo_lyrics.txt --title "demo"

# lyrics only, full JSON
python run.py --lyrics-text "I'm walking on sunshine" --title "demo" --json

# tests (run with fallbacks, no downloads)
python tests/test_pipeline.py
```

## Output

For each song you get audio V/A, lyric V/A + dominant emotion, an **alignment
score (0–1)**, a label (`aligned` / `contrast/ironic` / `partial-mismatch`), a
natural-language verdict, and the retrieved reference songs used for context.

## Layout

```
config.py              tunable settings & model names
schemas.py             typed data structures
src/audio_features.py  librosa → valence/arousal
src/lyric_sentiment.py local emotion model + fallbacks
src/rag_context.py     Chroma / TF-IDF reference retriever
src/fusion.py          V-A fusion + local-LLM verdict
src/graph.py           LangGraph DAG (+ plain fallback)
run.py                 CLI
data/reference_songs.json  RAG exemplars
tests/                 end-to-end tests
```

## Honest limitations

- Audio valence is heuristic (mode/brightness/harmonicity); arousal is more
  reliable than valence — true everywhere in MIR, not just here.
- Lyric acquisition at scale is a licensing problem (Genius has no official
  full-lyric API; Musixmatch full lyrics need a paid/licensed tier). Use a
  dataset that bundles lyrics for benchmarking.
- The "accuracy vs humans" claim only holds once evaluated against a labelled
  dataset — wire `data/reference_songs.json` up to DEAM/MoodyLyrics to report
  real numbers.
```
