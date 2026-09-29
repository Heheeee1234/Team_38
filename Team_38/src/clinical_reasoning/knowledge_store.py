from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .schema import Citation

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    source_class: str
    version: str
    source_url: str
    locator: str
    applicability: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    document: KnowledgeDocument
    chunk_text: str
    chunk_index: int
    score: float


class EmbeddingBackend(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """CPU-capable local embeddings. Model files download once, then run offline."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local dense retrieval needs sentence-transformers. Install the P4 "
                "extra with `pip install -e \".[p4-local]\"`."
            ) from exc
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            ), dtype=np.float32,
        )


def clean_text(text: str) -> str:
    """Normalize whitespace and common HTML entities without changing meaning."""
    text = text.replace("\u00a0", " ").replace("&nbsp;", " ")
    return _SPACE.sub(" ", text).strip()


def load_corpus(path: str | Path) -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []
    seen: set[str] = set()
    required = (
        "document_id", "title", "source_class", "version", "source_url",
        "locator", "applicability", "text",
    )
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in required if not row.get(key)]
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {missing}")
            if row["document_id"] in seen:
                raise ValueError(f"{path}:{line_no} duplicate document_id {row['document_id']!r}")
            seen.add(row["document_id"])
            docs.append(KnowledgeDocument(**{
                key: clean_text(str(row[key])) for key in required
            }))
    if not docs:
        raise ValueError(f"Knowledge corpus is empty: {path}")
    return docs


def chunk_document(doc: KnowledgeDocument, max_chars: int = 700,
                   overlap_sentences: int = 1) -> list[str]:
    if max_chars < 1 or overlap_sentences < 0:
        raise ValueError("max_chars must be positive and overlap_sentences nonnegative")
    sentences = [clean_text(s) for s in _SENTENCE_SPLIT.split(doc.text) if clean_text(s)]
    if not sentences:
        return [clean_text(doc.text)]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(item) + 1 for item in current)
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def corpus_fingerprint(documents: Sequence[KnowledgeDocument]) -> str:
    payload = json.dumps([doc.__dict__ for doc in documents], sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PersistentVectorRetriever:
    """Local dense retrieval backed by a small portable SQLite vector index."""

    def __init__(self, documents: list[KnowledgeDocument], index_path: str | Path,
                 backend: EmbeddingBackend | None = None, max_chunk_chars: int = 700,
                 model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.index_path = Path(index_path)
        self.backend = backend or SentenceTransformerBackend(model_id)
        self.model_id = self.backend.model_id
        self.chunks = [
            RetrievedChunk(doc, text, index, 0.0)
            for doc in documents
            for index, text in enumerate(chunk_document(doc, max_chunk_chars))
        ]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

    @classmethod
    def from_jsonl(cls, corpus_path: str | Path, index_path: str | Path,
                   **kwargs) -> "PersistentVectorRetriever":
        return cls(load_corpus(corpus_path), index_path, **kwargs)

    def _connect(self):
        return sqlite3.connect(self.index_path)

    @contextmanager
    def _connection(self):
        db = self._connect()
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _ensure_index(self) -> None:
        fingerprint = corpus_fingerprint(self.documents)
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL, embedding BLOB NOT NULL,
                PRIMARY KEY(document_id, chunk_index))""")
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            expected = {"corpus_sha256": fingerprint, "model_id": self.model_id}
            if all(metadata.get(key) == value for key, value in expected.items()):
                count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
                if count == len(self.chunks):
                    self._vectors = [np.frombuffer(row[0], dtype=np.float32).copy()
                                     for row in db.execute("SELECT embedding FROM chunks ORDER BY rowid")]
                    return
            vectors = self.backend.encode([chunk.chunk_text for chunk in self.chunks])
            if vectors.ndim != 2 or vectors.shape[0] != len(self.chunks):
                raise ValueError("Embedding backend returned an invalid matrix")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM metadata")
            db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?)",
                [(chunk.document.document_id, chunk.chunk_index, chunk.chunk_text,
                  vector.astype(np.float32).tobytes())
                 for chunk, vector in zip(self.chunks, vectors)],
            )
            db.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", expected.items())
            self._vectors = [vector.copy() for vector in vectors]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if top_k <= 0 or not query.strip():
            return []
        query_vector = self.backend.encode([clean_text(query)])[0]
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = np.asarray(self._vectors) @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [RetrievedChunk(self.chunks[int(i)].document,
                               self.chunks[int(i)].chunk_text,
                               self.chunks[int(i)].chunk_index,
                               float(scores[int(i)])) for i in order]


class TfidfRetriever:
    """Dependency-light lexical fallback for fully offline smoke runs."""

    def __init__(self, documents: list[KnowledgeDocument], max_chunk_chars: int = 700):
        self.chunks = [RetrievedChunk(d, text, i, 0.0)
                       for d in documents
                       for i, text in enumerate(chunk_document(d, max_chunk_chars))]
        df: Counter[str] = Counter()
        for chunk in self.chunks:
            df.update(set(self._tokens(chunk.chunk_text)))
        self._n = len(self.chunks)
        self._idf = {term: math.log((1 + self._n) / (1 + freq)) + 1
                     for term, freq in df.items()}
        self._vectors = [self._vector(chunk.chunk_text) for chunk in self.chunks]

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs):
        return cls(load_corpus(path), **kwargs)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN.findall(text)]

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(self._tokens(text))
        total = sum(counts.values()) or 1
        return {term: count / total * self._idf.get(term, 1.0)
                for term, count in counts.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(value * b.get(term, 0.0) for term, value in a.items())
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        return 0.0 if not norm_a or not norm_b else dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vector = self._vector(query)
        ranked = [(self._cos(query_vector, vector), chunk)
                  for chunk, vector in zip(self.chunks, self._vectors)]
        ranked.sort(key=lambda pair: (pair[0], pair[1].document.document_id), reverse=True)
        return [RetrievedChunk(chunk.document, chunk.chunk_text, chunk.chunk_index, score)
                for score, chunk in ranked[:max(0, top_k)] if score > 0]


RAGRetriever = TfidfRetriever


def citation_from_chunk(chunk: RetrievedChunk) -> Citation:
    doc = chunk.document
    return Citation(doc.document_id, doc.version, doc.locator, doc.source_url,
                    chunk.score, chunk.chunk_text)
