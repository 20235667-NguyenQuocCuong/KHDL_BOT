from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

import config
from models import Article

logger = logging.getLogger("embedder")

# In-memory storage
_embeddings: Optional[np.ndarray] = None
_documents: list[str] = []
_metadatas: list[dict] = []
_index: Optional[faiss.IndexFlatIP] = None


def _get_data_path() -> Path:
    """Get the path for FAISS index and metadata files."""
    base_path = Path(config.FAISS_PATH)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _save_state():
    """Save FAISS index and metadata to disk."""
    if _index is None or len(_documents) == 0:
        return

    data_path = _get_data_path()

    # Ensure directory exists
    data_path.mkdir(parents=True, exist_ok=True)

    index_file = data_path / "index.faiss"
    metadata_file = data_path / "metadata.pkl"

    # Save FAISS index
    faiss.write_index(_index, str(index_file))

    # Save documents and metadatas
    with open(str(metadata_file), "wb") as f:
        pickle.dump({"documents": _documents, "metadatas": _metadatas}, f)

    logger.debug("Saved FAISS index with %d documents to %s", len(_documents), data_path)


def _load_state() -> bool:
    """Load FAISS index and metadata from disk."""
    global _embeddings, _documents, _metadatas, _index

    data_path = _get_data_path()
    index_file = data_path / "index.faiss"
    metadata_file = data_path / "metadata.pkl"

    logger.info("Checking FAISS files at: %s", data_path)
    logger.info("index.faiss exists: %s", index_file.exists())
    logger.info("metadata.pkl exists: %s", metadata_file.exists())

    if not index_file.exists() or not metadata_file.exists():
        logger.info("No existing FAISS index found, starting fresh")
        return False

    try:
        # Load FAISS index
        logger.info("Loading FAISS index from %s...", index_file)
        _index = faiss.read_index(str(index_file))
        logger.info("FAISS index loaded successfully, ntotal=%d", _index.ntotal)

        # Load documents and metadatas
        logger.info("Loading metadata from %s...", metadata_file)
        with open(str(metadata_file), "rb") as f:
            data = pickle.load(f)
            _documents = data["documents"]
            _metadatas = data["metadatas"]

        logger.info("Loaded FAISS index with %d documents from %s", len(_documents), data_path)
        logger.info("Metadata keys: %s", list(data.keys()) if isinstance(data, dict) else "N/A")
        return True
    except Exception as exc:
        logger.warning("Failed to load FAISS index: %s. Starting fresh.", exc)
        import traceback
        logger.warning("Traceback: %s", traceback.format_exc())
        return False


def init_index(dim: Optional[int] = None):
    """Initialize FAISS index and load existing data from disk."""
    global _index, _embeddings, _documents, _metadatas

    if _index is None:
        dim = dim or get_embedding_dim()
        # Use Inner Product for cosine similarity (after normalization)
        _index = faiss.IndexFlatIP(dim)
        _embeddings = np.zeros((0, dim), dtype=np.float32)

        # Try to load existing data
        if not _load_state():
            logger.info("Initialized new FAISS index with dimension %d", dim)
        elif _index.d != dim:
            logger.warning(
                "Existing FAISS index has dimension %d but embedding model '%s' "
                "produces %d – ignoring old index (it will be overwritten on next crawl).",
                _index.d, config.EMBEDDING_MODEL, dim,
            )
            _index = faiss.IndexFlatIP(dim)
            _documents = []
            _metadatas = []


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


_model = None


def _get_model():
    """Load the local sentence-transformers model once (downloaded on first use)."""
    global _model
    if _model is None:
        # Clear SSL_CERT_FILE if it points to non-existent file (breaks HF download)
        ssl_cert = os.environ.get("SSL_CERT_FILE")
        if ssl_cert and not os.path.exists(ssl_cert):
            del os.environ["SSL_CERT_FILE"]
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model '%s'...", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(
            "Embedding model loaded (dim=%d, device=%s)",
            _dim_of(_model), _model.device,
        )
    return _model


def _dim_of(model) -> int:
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return fn()


def get_embedding_dim() -> int:
    """Vector dimension of the configured embedding model."""
    return _dim_of(_get_model())


def _embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts locally. Returns float32 array of shape (n, dim)."""
    prefix = config.EMBEDDING_QUERY_PREFIX if is_query else config.EMBEDDING_PASSAGE_PREFIX
    inputs = [prefix + t for t in texts] if prefix else texts
    embeddings = _get_model().encode(
        inputs,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(inputs) > config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_articles(articles: list[Article]) -> int:
    """Embed and store articles in FAISS index."""
    global _documents, _metadatas, _index, _embeddings

    if not articles:
        return 0

    # Initialize index if needed
    init_index()

    # Deduplication: skip articles that already exist in the index
    existing_urls = {meta.get("url") for meta in _metadatas if meta.get("url")}
    new_articles = [a for a in articles if a.url not in existing_urls]

    skipped = len(articles) - len(new_articles)
    if skipped > 0:
        logger.info("Skipped %d articles that already exist in FAISS index", skipped)

    if not new_articles:
        logger.info("No new articles to embed")
        return 0

    start = time.time()
    total_chunks = 0

    all_chunks: list[str] = []
    all_metadatas: list[dict] = []

    for article in new_articles:
        # Split article into chunks
        chunks = _chunk_text(article.content)
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{article.url}::chunk::{idx}".encode()
            ).hexdigest()

            all_chunks.append(chunk)
            all_metadatas.append({
                "title": article.title,
                "url": article.url,
                "source": article.source,
                "category": article.category,
                "published_at": (
                    article.published_at.isoformat()
                    if article.published_at else ""
                ),
                "crawled_at": article.crawled_at.isoformat(),
                "chunk_index": idx,
                "chunk_id": chunk_id,
            })
            total_chunks += 1

    if not all_chunks:
        return 0

    logger.info(
        "Embedding and storing %d chunks from %d articles...",
        len(all_chunks), len(new_articles)
    )

    # Get embeddings and normalize
    embeddings_np = _embed_texts(all_chunks)
    embeddings_normalized = _normalize_embeddings(embeddings_np)

    # Add to FAISS index
    _index.add(embeddings_normalized)

    # Update in-memory storage
    start_idx = len(_documents)
    _documents.extend(all_chunks)
    _metadatas.extend(all_metadatas)

    # Save to disk
    _save_state()

    elapsed = time.time() - start
    logger.info(
        "Successfully stored %d chunks in FAISS (%.1fs, total: %d documents)",
        total_chunks, elapsed, len(_documents)
    )

    return total_chunks


def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    words = text.split()
    words_per_chunk = max(1, int(chunk_size / 1.3))
    words_overlap = max(0, int(overlap / 1.3))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - words_overlap

    return chunks


def get_stats() -> dict:
    """Get statistics about the FAISS index."""
    count = len(_documents) if _documents else 0

    stats: dict = {
        "total_chunks": count,
        "sources_breakdown": {},
        "categories_breakdown": {},
        "last_crawled_at": None,
    }

    if count == 0:
        return stats

    latest_crawl = None
    for meta in _metadatas:
        src = meta.get("source", "Unknown")
        cat = meta.get("category", "Unknown")
        stats["sources_breakdown"][src] = stats["sources_breakdown"].get(src, 0) + 1
        stats["categories_breakdown"][cat] = stats["categories_breakdown"].get(cat, 0) + 1

        crawled = meta.get("crawled_at", "")
        if crawled and (latest_crawl is None or crawled > latest_crawl):
            latest_crawl = crawled

    stats["last_crawled_at"] = latest_crawl
    return stats


def clear_index():
    """Clear the FAISS index and delete stored data."""
    global _embeddings, _documents, _metadatas, _index

    _documents = []
    _metadatas = []
    _embeddings = None
    _index = None

    # Delete files
    data_path = _get_data_path()
    for file in ["index.faiss", "metadata.pkl"]:
        file_path = data_path / file
        if file_path.exists():
            os.remove(str(file_path))

    logger.info("Cleared FAISS index and deleted stored data")

