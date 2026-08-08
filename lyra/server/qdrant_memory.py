"""Qdrant-backed episodic memory with EmbeddingGemma + BM25 hybrid search."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Protocol

from qdrant_client import QdrantClient, models


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DIM = 768
DEFAULT_DENSE_MODEL = "google/embeddinggemma-300m"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


def document_prefix(text: str, title: str = "none") -> str:
    """EmbeddingGemma document template."""
    return f"title: {title} | text: {text}"


def query_prefix(text: str) -> str:
    """EmbeddingGemma query template for retrieval."""
    return f"task: search result | query: {text}"


def entry_point_id(entry_id: str) -> str:
    """Stable UUID string derived from an episodic entry id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lyra:episodic:{entry_id}"))


class Embedder(Protocol):
    """Produces dense + sparse vectors for documents and queries."""

    dense_dim: int

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        ...

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        ...


class FastEmbedHybridEmbedder:
    """
    FastEmbed wrapper for EmbeddingGemma (dense) + BM25 (sparse).

    Models are loaded lazily on first embed so importing the module is cheap.
    """

    def __init__(
        self,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        dense_dim: int = EMBEDDING_DIM,
    ):
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.dense_dim = dense_dim
        self._dense = None
        self._sparse = None

    def _ensure_models(self) -> None:
        if self._dense is not None and self._sparse is not None:
            return
        from fastembed import SparseTextEmbedding, TextEmbedding

        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        if self.dense_model_name not in supported:
            raise RuntimeError(
                f"FastEmbed does not list {self.dense_model_name}. "
                "Install FastEmbed from git (see requirements.txt) so EmbeddingGemma is available."
            )
        self._dense = TextEmbedding(model_name=self.dense_model_name)
        self._sparse = SparseTextEmbedding(model_name=self.sparse_model_name)

    @staticmethod
    def _to_sparse(embedding: Any) -> models.SparseVector:
        indices = embedding.indices.tolist() if hasattr(embedding.indices, "tolist") else list(embedding.indices)
        values = embedding.values.tolist() if hasattr(embedding.values, "tolist") else list(embedding.values)
        return models.SparseVector(indices=[int(i) for i in indices], values=[float(v) for v in values])

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        self._ensure_models()
        assert self._dense is not None and self._sparse is not None
        prefixed = [document_prefix(t) for t in texts]
        dense_vecs = list(self._dense.embed(prefixed))
        sparse_vecs = list(self._sparse.embed(texts))
        results: list[tuple[list[float], models.SparseVector]] = []
        for dense, sparse in zip(dense_vecs, sparse_vecs):
            results.append((dense.tolist(), self._to_sparse(sparse)))
        return results

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        self._ensure_models()
        assert self._dense is not None and self._sparse is not None
        # Apply EmbeddingGemma query prefix explicitly; BM25 uses raw query text.
        dense = next(iter(self._dense.embed([query_prefix(text)])))
        sparse = next(iter(self._sparse.embed([text])))
        return dense.tolist(), self._to_sparse(sparse)


class FakeHybridEmbedder:
    """Deterministic embedder for unit tests (no model download)."""

    def __init__(self, dense_dim: int = 32):
        self.dense_dim = dense_dim

    def _dense_from_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self.dense_dim)]
        # Boost dimensions based on token hashes so similar words score higher.
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dense_dim
            values[idx] += 1.0
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def _sparse_from_text(self, text: str) -> models.SparseVector:
        indices: list[int] = []
        values: list[float] = []
        for token in sorted(set(text.lower().split())):
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % 10_000
            indices.append(idx)
            values.append(1.0)
        return models.SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        return [(self._dense_from_text(t), self._sparse_from_text(t)) for t in texts]

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        return self._dense_from_text(text), self._sparse_from_text(text)


class QdrantEpisodicStore:
    """Persistent episodic memory in Qdrant with hybrid dense+sparse retrieval."""

    def __init__(
        self,
        *,
        url: str | None = "http://localhost:6333",
        collection: str = "lyra_episodic",
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        vector_size: int | None = None,
        location: str | None = None,
    ):
        if client is not None:
            self.client = client
            self._local_mode = False
        elif location == ":memory:" or url == ":memory:":
            self.client = QdrantClient(location=":memory:")
            self._local_mode = True
        else:
            self.client = QdrantClient(url=url or "http://localhost:6333")
            self._local_mode = False

        self.collection = collection
        self.embedder: Embedder = embedder or FastEmbedHybridEmbedder(
            dense_dim=vector_size or EMBEDDING_DIM
        )
        self.vector_size = vector_size or getattr(self.embedder, "dense_dim", EMBEDDING_DIM)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        if not self._local_mode:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="timestamp",
                field_schema=models.PayloadSchemaType.FLOAT,
            )

    def upsert_entry(self, entry: dict[str, Any]) -> None:
        text = entry.get("text", "")
        dense, sparse = self.embedder.embed_documents([text])[0]
        point_id = entry_point_id(str(entry["id"]))
        payload = {
            "id": entry["id"],
            "timestamp": float(entry.get("timestamp", 0.0)),
            "readable_time": entry.get("readable_time", ""),
            "speaker": entry.get("speaker", ""),
            "is_user": bool(entry.get("is_user", True)),
            "text": text,
            "confidence": float(entry.get("confidence", 1.0)),
        }
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: sparse,
                    },
                    payload=payload,
                )
            ],
        )

    def search_hybrid(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        dense, sparse = self.embedder.embed_query(query)
        prefetch_limit = max(top_k * 5, 20)
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse,
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            score = float(point.score) if point.score is not None else 0.0
            payload["relevance_score"] = round(score, 3)
            results.append(payload)
        return results

    def scroll_all(self, limit: int = 10_000) -> list[dict[str, Any]]:
        points, _next = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        entries = [dict(p.payload or {}) for p in points]
        entries.sort(key=lambda e: float(e.get("timestamp", 0.0)))
        return entries

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def clear(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            self.client.delete_collection(self.collection)
        self._ensure_collection()

    def prune_to_max(self, max_entries: int) -> None:
        if max_entries <= 0:
            return
        total = self.count()
        if total <= max_entries:
            return
        overflow = total - max_entries
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=overflow,
            with_payload=True,
            with_vectors=False,
            order_by=models.OrderBy(key="timestamp", direction=models.Direction.ASC),
        )
        if not points:
            # Fallback without order_by for older / in-memory quirks
            all_entries = self.scroll_all()
            to_delete = all_entries[:overflow]
            ids = [entry_point_id(str(e["id"])) for e in to_delete if "id" in e]
        else:
            ids = [p.id for p in points]
        if ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=ids),
            )

    def health(self) -> dict[str, Any]:
        try:
            count = self.count()
            return {"ok": True, "collection": self.collection, "points": count}
        except Exception as exc:
            return {"ok": False, "collection": self.collection, "error": str(exc)}
