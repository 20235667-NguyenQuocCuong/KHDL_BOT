"""
Cấu hình cho phần Crawler + Embedding (báo cáo giữa kỳ).
Chỉ giữ các thiết lập liên quan đến thu thập dữ liệu và tạo vector,
không bao gồm phần tìm kiếm / LLM.
"""

import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# Nơi lưu FAISS index + metadata
FAISS_PATH = os.getenv("FAISS_PATH", str(BASE_DIR / "faiss_data"))

# ---------------- Embedding ----------------
# Model chạy local (sentence-transformers), tự tải về ở lần chạy đầu.
# Đổi model => đổi số chiều vector => phải xoá faiss_data/ và crawl lại.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
# Model E5 cần tiền tố "query: " / "passage: "; với BAAI/bge-m3 thì để rỗng.
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ---------------- Chunking ----------------
CHUNK_SIZE = 300         # số token xấp xỉ mỗi chunk
CHUNK_OVERLAP = 50       # số token chồng lấn giữa 2 chunk liên tiếp

# ---------------- Nguồn RSS ----------------
RSS_FEEDS = [
    # VnExpress
    {"url": "https://vnexpress.net/rss/khoa-hoc.rss",   "source": "VnExpress",  "category": "Cong nghe"},
    {"url": "https://vnexpress.net/rss/kinh-doanh.rss",  "source": "VnExpress",  "category": "Kinh te"},
    {"url": "https://vnexpress.net/rss/the-thao.rss",    "source": "VnExpress",  "category": "The thao"},
    {"url": "https://vnexpress.net/rss/the-gioi.rss",    "source": "VnExpress",  "category": "The gioi"},
    # Tuoi Tre
    {"url": "https://tuoitre.vn/rss/nhip-song-so.rss",  "source": "Tuoi Tre",   "category": "Cong nghe"},
    {"url": "https://tuoitre.vn/rss/kinh-doanh.rss",    "source": "Tuoi Tre",   "category": "Kinh te"},
    {"url": "https://tuoitre.vn/rss/the-thao.rss",      "source": "Tuoi Tre",   "category": "The thao"},
    {"url": "https://tuoitre.vn/rss/the-gioi.rss",      "source": "Tuoi Tre",   "category": "The gioi"},
    # Thanh Nien
    {"url": "https://thanhnien.vn/rss/cong-nghe.rss",    "source": "Thanh Nien", "category": "Cong nghe"},
    {"url": "https://thanhnien.vn/rss/kinh-te.rss",      "source": "Thanh Nien", "category": "Kinh te"},
    {"url": "https://thanhnien.vn/rss/the-thao.rss",     "source": "Thanh Nien", "category": "The thao"},
    {"url": "https://thanhnien.vn/rss/the-gioi.rss",     "source": "Thanh Nien", "category": "The gioi"},
]

# Giới hạn số bài mỗi feed (0 = không giới hạn). Dùng để demo nhanh.
MAX_ARTICLES_PER_FEED = int(os.getenv("MAX_ARTICLES_PER_FEED", "0"))

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
