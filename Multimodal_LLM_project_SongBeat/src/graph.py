"""LangGraph orchestration of the SongBeat pipeline.

This is the honest use of LangChain/LangGraph: a *defined* directed graph, not an
autonomous agent loop. Nodes:

    ingest -> audio_node -> lyric_node -> rag_node -> fusion_node

If langgraph isn't installed, `build_pipeline` returns an equivalent plain-Python
callable with the same signature, so behaviour is identical either way.
"""
from __future__ import annotations
from typing import Callable

from schemas import PipelineState, AlignmentResult
from src import audio_features, lyric_sentiment, fusion
from src.rag_context import ReferenceRetriever


def _make_nodes(retriever: ReferenceRetriever):
    def audio_node(state: PipelineState) -> PipelineState:
        path = state.get("audio_path")
        if path:
            state["audio_emotion"] = audio_features.extract(path)
        else:
            # No audio supplied: neutral audio so the run still completes.
            from schemas import AudioEmotion
            state["audio_emotion"] = AudioEmotion(0.0, 0.0, {"note": "no audio provided"})
        return state

    def lyric_node(state: PipelineState) -> PipelineState:
        state["lyric_emotion"] = lyric_sentiment.analyze(state.get("lyrics", ""))
        return state

    def rag_node(state: PipelineState) -> PipelineState:
        state["retrieved_context"] = retriever.retrieve(state.get("lyrics", ""))
        return state

    def fusion_node(state: PipelineState) -> PipelineState:
        state["result"] = fusion.fuse(state["audio_emotion"], state["lyric_emotion"],
                                      state.get("retrieved_context", []))
        return state

    return audio_node, lyric_node, rag_node, fusion_node


def build_pipeline(retriever: ReferenceRetriever | None = None) -> Callable[[PipelineState], PipelineState]:
    retriever = retriever or ReferenceRetriever()
    audio_node, lyric_node, rag_node, fusion_node = _make_nodes(retriever)

    try:
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(PipelineState)
        g.add_node("audio", audio_node)
        g.add_node("lyric", lyric_node)
        g.add_node("rag", rag_node)
        g.add_node("fusion", fusion_node)
        g.add_edge(START, "audio")
        g.add_edge("audio", "lyric")
        g.add_edge("lyric", "rag")
        g.add_edge("rag", "fusion")
        g.add_edge("fusion", END)
        compiled = g.compile()

        def run(state: PipelineState) -> PipelineState:
            return compiled.invoke(state)
        run.backend = "langgraph"  # type: ignore[attr-defined]
        return run
    except Exception:
        def run(state: PipelineState) -> PipelineState:
            for node in (audio_node, lyric_node, rag_node, fusion_node):
                state = node(state)
            return state
        run.backend = "plain"  # type: ignore[attr-defined]
        return run


def analyze_song(audio_path: str | None, lyrics: str, title: str = "",
                 retriever: ReferenceRetriever | None = None) -> AlignmentResult:
    pipeline = build_pipeline(retriever)
    state: PipelineState = {"audio_path": audio_path, "lyrics": lyrics, "title": title}
    return pipeline(state)["result"]
