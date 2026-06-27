"""RAG layer with a real purpose: retrieve emotionally-similar reference songs.

Why RAG here is not decoration: the fusion step is calibrated few-shot. Given a
new song's lyrics, we retrieve the k nearest *reference exemplars* (songs with
known mood labels + audio valence-arousal). Those exemplars are injected into the
fusion prompt so the local LLM judges alignment relative to comparable songs
rather than in a vacuum, and they give the user genre/mood context.

Primary backend: Chroma (persistent vector DB, default local embeddings).
Fallback: a self-contained TF-IDF cosine store with the same interface, so the
pipeline runs even when chromadb isn't installed.
"""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config import SETTINGS, REFERENCE_SONGS

_TOKEN = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _TfidfStore:
    """Minimal dependency-free vector store (TF-IDF + cosine)."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.vectors: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}

    def build(self, exemplars: list[dict[str, Any]]) -> None:
        self.docs = exemplars
        tokenised = [_tokens(e["lyrics"]) for e in exemplars]
        df: Counter = Counter()
        for toks in tokenised:
            df.update(set(toks))
        n = len(exemplars)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vec(toks) for toks in tokenised]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: (c / len(toks)) * self.idf.get(t, 1.0) for t, c in tf.items()} if toks else {}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        qv = self._vec(_tokens(query))
        scored = sorted(((self._cos(qv, v), d) for v, d in zip(self.vectors, self.docs)),
                        key=lambda x: x[0], reverse=True)
        return [{**d, "similarity": round(s, 3)} for s, d in scored[:k]]


class _ChromaStore:
    """Chroma-backed store. Uses Chroma's default local embedding function."""

    def __init__(self) -> None:
        import chromadb
        self.client = chromadb.Client()
        self.col = self.client.get_or_create_collection(SETTINGS.collection_name)
        self.by_id: dict[str, dict[str, Any]] = {}

    def build(self, exemplars: list[dict[str, Any]]) -> None:
        ids, docs, metas = [], [], []
        for i, e in enumerate(exemplars):
            sid = str(i)
            self.by_id[sid] = e
            ids.append(sid)
            docs.append(e["lyrics"])
            metas.append({"title": e.get("title", ""), "mood": e.get("mood", ""),
                          "valence": e.get("valence", 0.0), "arousal": e.get("arousal", 0.0)})
        self.col.add(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        res = self.col.query(query_texts=[query], n_results=k)
        out = []
        for sid, dist in zip(res["ids"][0], res["distances"][0]):
            doc = dict(self.by_id[sid])
            doc["similarity"] = round(1.0 - float(dist), 3)
            out.append(doc)
        return out


class ReferenceRetriever:
    """Public RAG interface; picks Chroma if available, else TF-IDF."""

    def __init__(self, exemplars: list[dict[str, Any]] | None = None) -> None:
        if exemplars is None:
            exemplars = self.load_reference_songs()
        try:
            self.store: Any = _ChromaStore()
            self.backend = "chroma"
        except Exception:
            self.store = _TfidfStore()
            self.backend = "tfidf"
        self.store.build(exemplars)

    @staticmethod
    def load_reference_songs() -> list[dict[str, Any]]:
        if Path(REFERENCE_SONGS).exists():
            return json.loads(Path(REFERENCE_SONGS).read_text())
        return []

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        return self.store.retrieve(query, k or SETTINGS.rag_top_k)
