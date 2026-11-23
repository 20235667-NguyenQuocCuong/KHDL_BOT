#!/usr/bin/env python3
"""
Xem nội dung FAISS index đã tạo (để minh hoạ trong báo cáo).

    python inspect_index.py            # thống kê + 3 chunk mẫu
    python inspect_index.py --n 10     # in 10 chunk mẫu
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

import config
import faiss_embedder as fe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Số chunk mẫu cần in")
    args = parser.parse_args()

    if not fe._load_state():
        print(f"Chưa có index tại {config.FAISS_PATH}. Hãy chạy crawl_and_embed.py trước.")
        return 1

    docs, metas, index = fe._documents, fe._metadatas, fe._index
    n_articles = len({m["url"] for m in metas})
    lengths = [len(d.split()) for d in docs]

    print("=" * 60)
    print("TỔNG QUAN")
    print("=" * 60)
    print(f"Số vector trong FAISS : {index.ntotal}")
    print(f"Số chiều vector       : {index.d}")
    print(f"Loại index            : {type(index).__name__} (Inner Product trên vector đã chuẩn hoá = cosine)")
    print(f"Số bài báo            : {n_articles}")
    print(f"Số chunk              : {len(docs)}")
    print(f"Chunk / bài (TB)      : {len(docs) / max(n_articles, 1):.2f}")
    print(f"Số từ / chunk (TB)    : {sum(lengths) / max(len(lengths), 1):.1f} "
          f"(min {min(lengths)}, max {max(lengths)})")

    print("\nTheo nguồn x chuyên mục (số chunk):")
    combo = Counter((m["source"], m["category"]) for m in metas)
    for (src, cat), c in sorted(combo.items()):
        print(f"  {src:<12} | {cat:<10} | {c}")

    print("\nThống kê đầy đủ:")
    print(json.dumps(fe.get_stats(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"{args.n} CHUNK MẪU")
    print("=" * 60)
    for i in range(min(args.n, len(docs))):
        m = metas[i]
        vec = index.reconstruct(i)
        print(f"\n[{i}] {m['source']} / {m['category']} - chunk #{m['chunk_index']}")
        print(f"    Tiêu đề : {m['title']}")
        print(f"    URL     : {m['url']}")
        print(f"    Văn bản : {docs[i][:200]}...")
        print(f"    Vector  : [{', '.join(f'{x:.4f}' for x in vec[:6])}, ...] "
              f"(||v|| = {float((vec ** 2).sum() ** 0.5):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
