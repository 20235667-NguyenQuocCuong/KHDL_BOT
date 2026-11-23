#!/usr/bin/env python3
"""
Pipeline giữa kỳ: Crawl RSS  ->  Trích nội dung  ->  Chunking  ->  Embedding  ->  Lưu FAISS.

Cách chạy:
    python crawl_and_embed.py              # crawl toàn bộ 12 feed
    python crawl_and_embed.py --limit 5    # mỗi feed chỉ lấy 5 bài (demo nhanh)
    python crawl_and_embed.py --reset      # xoá index cũ trước khi chạy

Không cần API key: model embedding chạy local (tự tải ở lần đầu).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Tránh lỗi OpenMP trùng lặp trên Windows (faiss + torch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl tin tức và embed vào FAISS")
    parser.add_argument("--limit", type=int, default=None,
                        help="Số bài tối đa mỗi feed (mặc định: không giới hạn)")
    parser.add_argument("--reset", action="store_true",
                        help="Xoá FAISS index cũ trước khi chạy")
    args = parser.parse_args()

    if args.limit is not None:
        config.MAX_ARTICLES_PER_FEED = args.limit

    config.setup_logging()

    from crawler import run_crawl
    from faiss_embedder import clear_index, embed_articles, get_stats, init_index

    if args.reset:
        clear_index()

    print("=" * 60)
    print("BƯỚC 1: CRAWL RSS + TRÍCH NỘI DUNG BÀI BÁO")
    print("=" * 60)
    t0 = time.time()
    articles = run_crawl()
    t_crawl = time.time() - t0
    print(f"-> Thu được {len(articles)} bài trong {t_crawl:.1f}s")

    if not articles:
        print("Không có bài mới. Kết thúc.")
        return 0

    print("\n" + "=" * 60)
    print("BƯỚC 2: CHUNKING + EMBEDDING + LƯU FAISS")
    print("=" * 60)
    t1 = time.time()
    init_index()
    n_chunks = embed_articles(articles)
    t_embed = time.time() - t1
    print(f"-> Tạo {n_chunks} chunk mới trong {t_embed:.1f}s")

    print("\n" + "=" * 60)
    print("THỐNG KÊ INDEX")
    print("=" * 60)
    print(json.dumps(get_stats(), indent=2, ensure_ascii=False))
    print(f"\nDữ liệu lưu tại: {config.FAISS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
