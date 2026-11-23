"""
API kiểm thử phần Crawler + Embedding (dùng với Postman).

Chạy:
    uvicorn server:app --reload --port 8000

Các endpoint:
    GET    /health                  kiểm tra server
    GET    /feeds                   danh sách RSS feed đang cấu hình
    POST   /crawl?limit=2           chỉ crawl, giữ kết quả trong bộ nhớ
    GET    /articles                xem các bài vừa crawl (chưa embed)
    POST   /embed                   embed các bài vừa crawl vào FAISS
    POST   /crawl-and-embed?limit=2 chạy cả pipeline
    GET    /stats                   thống kê FAISS index
    GET    /chunks?skip=0&limit=5   xem chunk đã lưu (kèm vector rút gọn)
    POST   /chunk-preview           thử chia chunk một đoạn văn bản bất kỳ
    DELETE /index                   xoá toàn bộ index
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import config
import crawler
import faiss_embedder as fe
from models import Article

config.setup_logging()

app = FastAPI(
    title="BC_GK - Crawler & Embedding API",
    description="API kiểm thử phần crawl tin tức và embedding vào FAISS",
    version="1.0.0",
)

# Bài đã crawl nhưng chưa embed
_pending: list[Article] = []


def _crawl(limit: Optional[int]) -> tuple[list[Article], float]:
    config.MAX_ARTICLES_PER_FEED = limit or 0
    t0 = time.time()
    articles = crawler.run_crawl()
    return articles, time.time() - t0


def _article_summary(a: Article) -> dict:
    return {
        "title": a.title,
        "url": a.url,
        "source": a.source,
        "category": a.category,
        "published_at": a.published_at.isoformat() if a.published_at else None,
        "content_length": len(a.content),
        "content_preview": a.content[:200],
    }


class ChunkPreviewRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = None
    overlap: Optional[int] = None


@app.on_event("startup")
def _startup() -> None:
    fe._load_state()


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": config.EMBEDDING_MODEL}


@app.get("/feeds")
def feeds():
    return {"total": len(config.RSS_FEEDS), "feeds": config.RSS_FEEDS}


@app.post("/crawl")
def crawl(
    limit: Optional[int] = Query(None, ge=1, description="Số bài tối đa mỗi feed"),
    clear_visited: bool = Query(False, description="Cho phép crawl lại các URL đã thấy"),
):
    global _pending
    if clear_visited:
        crawler._visited_urls.clear()
    articles, elapsed = _crawl(limit)
    _pending.extend(articles)
    return {
        "crawled": len(articles),
        "pending_total": len(_pending),
        "elapsed_seconds": round(elapsed, 2),
        "articles": [_article_summary(a) for a in articles],
    }


@app.get("/articles")
def articles(full: bool = Query(False, description="Trả về toàn bộ nội dung bài")):
    data = []
    for a in _pending:
        item = _article_summary(a)
        if full:
            item["content"] = a.content
        data.append(item)
    return {"pending_total": len(_pending), "articles": data}


@app.post("/embed")
def embed():
    global _pending
    if not _pending:
        raise HTTPException(400, "Chưa có bài nào. Gọi POST /crawl trước.")
    n_articles = len(_pending)
    t0 = time.time()
    fe.init_index()
    n_chunks = fe.embed_articles(_pending)
    elapsed = time.time() - t0
    _pending = []
    return {
        "articles_input": n_articles,
        "new_chunks": n_chunks,
        "elapsed_seconds": round(elapsed, 2),
        "stats": fe.get_stats(),
    }


@app.post("/crawl-and-embed")
def crawl_and_embed(
    limit: Optional[int] = Query(None, ge=1, description="Số bài tối đa mỗi feed"),
    reset: bool = Query(False, description="Xoá index cũ trước khi chạy"),
):
    if reset:
        fe.clear_index()
        crawler._visited_urls.clear()
    articles, t_crawl = _crawl(limit)
    t0 = time.time()
    fe.init_index()
    n_chunks = fe.embed_articles(articles) if articles else 0
    t_embed = time.time() - t0
    return {
        "crawled": len(articles),
        "new_chunks": n_chunks,
        "crawl_seconds": round(t_crawl, 2),
        "embed_seconds": round(t_embed, 2),
        "stats": fe.get_stats(),
    }


@app.get("/stats")
def stats():
    s = fe.get_stats()
    s["total_articles"] = len({m["url"] for m in fe._metadatas})
    s["vector_dim"] = fe._index.d if fe._index is not None else None
    s["pending_articles"] = len(_pending)
    return s


@app.get("/chunks")
def chunks(
    skip: int = Query(0, ge=0),
    limit: int = Query(5, ge=1, le=100),
    source: Optional[str] = Query(None, description="Lọc theo báo, vd: VnExpress"),
    vector_preview: int = Query(8, ge=0, le=1024, description="Số phần tử vector hiển thị"),
):
    if fe._index is None or not fe._documents:
        return {"total": 0, "items": []}
    idx = [i for i, m in enumerate(fe._metadatas) if not source or m.get("source") == source]
    items = []
    for i in idx[skip:skip + limit]:
        vec = fe._index.reconstruct(i)
        items.append({
            "id": i,
            **fe._metadatas[i],
            "text": fe._documents[i],
            "word_count": len(fe._documents[i].split()),
            "vector_preview": [round(float(x), 5) for x in vec[:vector_preview]],
            "vector_norm": round(float((vec ** 2).sum() ** 0.5), 5),
        })
    return {"total": len(idx), "skip": skip, "limit": limit, "items": items}


@app.post("/chunk-preview")
def chunk_preview(req: ChunkPreviewRequest):
    parts = fe._chunk_text(req.text, req.chunk_size, req.overlap)
    return {
        "input_words": len(req.text.split()),
        "num_chunks": len(parts),
        "chunks": [{"index": i, "word_count": len(c.split()), "text": c} for i, c in enumerate(parts)],
    }


@app.delete("/index")
def delete_index():
    global _pending
    fe.clear_index()
    crawler._visited_urls.clear()
    _pending = []
    return {"status": "cleared"}
