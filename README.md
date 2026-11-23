# KHDL_BOT – Báo cáo giữa kỳ: Crawler + Embedding

Phần đầu của hệ thống chatbot hỏi đáp tin tức tiếng Việt theo hướng RAG. Báo cáo giữa kỳ gồm hai khâu: **thu thập dữ liệu (crawler)** và **tạo vector (embedding)**. Phần tìm kiếm và sinh câu trả lời sẽ làm ở giai đoạn sau.

## 1. Pipeline

```
RSS feed (12 feed: VnExpress, Tuổi Trẻ, Thanh Niên × 4 chuyên mục)
   │  feedparser
   ▼
Lấy link bài ──► requests + BeautifulSoup (CSS selector riêng cho từng báo)
   │               └─ nếu không trích được nội dung thì dùng summary trong RSS
   ▼
Article (title, content, url, source, category, published_at, crawled_at)
   │  bỏ URL trùng (trong phiên crawl và so với index đã có)
   ▼
Chunking: ~230 từ/chunk, chồng lấn ~38 từ (CHUNK_SIZE=300, CHUNK_OVERLAP=50 token)
   │
   ▼
Embedding: intfloat/multilingual-e5-base (768 chiều, chạy local, tiền tố "passage: ")
   │  chuẩn hoá L2
   ▼
FAISS IndexFlatIP (inner product trên vector đơn vị = cosine)
   └─► faiss_data/index.faiss + faiss_data/metadata.pkl
```

## 2. Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `config.py` | Danh sách RSS feed, model embedding, kích thước chunk, đường dẫn lưu |
| `models.py` | Lớp `Article` (pydantic) |
| `crawler.py` | Đọc RSS, tải bài, trích nội dung theo từng báo, bỏ URL trùng |
| `faiss_embedder.py` | Chunking, embedding, lưu và nạp FAISS, bỏ bài đã có trong index, thống kê |
| `crawl_and_embed.py` | Chạy toàn bộ pipeline từ dòng lệnh |
| `inspect_index.py` | In thống kê index và vài chunk/vector mẫu |
| `server.py` | API FastAPI để kiểm thử bằng Postman |
| `postman_collection.json` | Bộ request Postman dựng sẵn |

## 3. Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
git clone https://github.com/20235667-NguyenQuocCuong/KHDL_BOT.git
cd KHDL_BOT

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

- Không cần API key.
- Lần chạy đầu sẽ tự tải model embedding (khoảng 1 GB) từ HuggingFace.
- Trên Windows, nếu terminal bị lỗi font tiếng Việt thì chạy `set PYTHONIOENCODING=utf-8` (cmd) hoặc `$env:PYTHONIOENCODING="utf-8"` (PowerShell).

## 4. Chạy bằng dòng lệnh

```bash
# Demo nhanh: mỗi feed lấy 2 bài, xoá index cũ trước khi chạy
python crawl_and_embed.py --limit 2 --reset

# Crawl đầy đủ (chạy lại thì chỉ thêm bài mới)
python crawl_and_embed.py

# Xem kết quả: thống kê và 5 chunk mẫu
python inspect_index.py --n 5
```

| Tham số | Ý nghĩa |
|---|---|
| `--limit N` | Mỗi feed chỉ lấy tối đa N bài |
| `--reset` | Xoá `faiss_data/` trước khi chạy |
| `--n N` (inspect) | Số chunk mẫu in ra |

## 5. Kiểm thử bằng Postman

### 5.1. Khởi động server

```bash
uvicorn server:app --reload --port 8000
```

Có thể kiểm tra nhanh trên trình duyệt:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`: Swagger UI, gọi thử API ngay trên trình duyệt

### 5.2. Import collection

1. Mở Postman, chọn **Import**, rồi chọn file `postman_collection.json`.
2. Collection **BC_GK - Crawler & Embedding** sẽ xuất hiện, có sẵn biến `base_url = http://127.0.0.1:8000`. Đổi cổng thì sửa biến này ở tab *Variables* của collection.

### 5.3. Danh sách endpoint

| # | Method | Endpoint | Tham số | Mô tả |
|---|---|---|---|---|
| 1 | GET | `/health` | | Kiểm tra server |
| 2 | GET | `/feeds` | | Danh sách 12 RSS feed |
| 3 | POST | `/crawl` | `limit`, `clear_visited` | Chỉ crawl, giữ bài trong bộ nhớ |
| 4 | GET | `/articles` | `full` | Xem bài đã crawl nhưng chưa embed |
| 5 | POST | `/embed` | | Embed các bài đang chờ vào FAISS |
| 6 | POST | `/crawl-and-embed` | `limit`, `reset` | Chạy cả pipeline |
| 7 | GET | `/stats` | | Thống kê index theo nguồn và chuyên mục |
| 8 | GET | `/chunks` | `skip`, `limit`, `source`, `vector_preview` | Xem chunk đã lưu, kèm vector rút gọn và độ dài vector |
| 9 | POST | `/chunk-preview` | body JSON | Thử chia chunk một đoạn văn bản |
| 10 | DELETE | `/index` | | Xoá toàn bộ index |

### 5.4. Kịch bản kiểm thử gợi ý

Chạy lần lượt theo thứ tự sau.

**Bước 1. `DELETE /index`**: bắt đầu từ index trống.
```json
{ "status": "cleared" }
```

**Bước 2. `POST /crawl?limit=2`**: crawl 2 bài mỗi feed. Kết quả mong đợi: `crawled` khoảng 24 bài, mỗi bài có `content_length > 0`.
```json
{
  "crawled": 24,
  "pending_total": 24,
  "elapsed_seconds": 12.8,
  "articles": [
    {
      "title": "Nữ kỹ sư đứng sau mô hình AI làm phim gây sốt ở Trung Quốc",
      "source": "VnExpress",
      "category": "Cong nghe",
      "content_length": 9305,
      "content_preview": "..."
    }
  ]
}
```

**Bước 3. `GET /articles`**: thấy 24 bài đang chờ embed.

**Bước 4. `POST /embed`**: embed các bài đó. Lần đầu sẽ lâu hơn vì phải nạp model.
```json
{
  "articles_input": 24,
  "new_chunks": 98,
  "elapsed_seconds": 27.2,
  "stats": { "total_chunks": 98, "sources_breakdown": { "VnExpress": 42, "Tuoi Tre": 32, "Thanh Nien": 24 } }
}
```

**Bước 5. `GET /stats`**: kiểm tra `vector_dim = 768`, `total_articles = 24`, `pending_articles = 0`.

**Bước 6. `GET /chunks?limit=2&vector_preview=6`**: mỗi chunk có metadata, văn bản, `word_count` không vượt quá 230 và `vector_norm` xấp xỉ 1.0 (đã chuẩn hoá).

**Bước 7. Kiểm tra chống trùng**: gọi `POST /crawl?limit=2&clear_visited=true` rồi `POST /embed`. Các bài đã có trong index bị bỏ qua, nên `new_chunks` chỉ tính cho bài mới (có thể bằng 0).

**Bước 8. `POST /chunk-preview`** với body:
```json
{
  "text": "Đội tuyển Việt Nam giành chiến thắng 2-0 trước Thái Lan trong trận chung kết lượt đi. Hai bàn thắng được ghi ở hiệp hai.",
  "chunk_size": 13,
  "overlap": 4
}
```
Văn bản 25 từ được chia thành 4 chunk (10, 10, 10 và 4 từ), hai chunk liền kề trùng nhau 3 từ. Số từ mỗi chunk tính bằng `int(chunk_size / 1.3)`, số từ chồng lấn tính bằng `int(overlap / 1.3)`.

**Bước 9. `POST /embed` khi không còn bài chờ**: trả lỗi `400`.
```json
{ "detail": "Chưa có bài nào. Gọi POST /crawl trước." }
```

**Bước 10. `POST /crawl-and-embed?limit=1&reset=true`**: chạy lại cả pipeline trong một request.

## 6. Kết quả chạy thử

Chạy với `--limit 2` trên CPU:

| Chỉ số | Giá trị |
|---|---|
| Số bài crawl | 24 bài / 12 feed |
| Thời gian crawl | khoảng 13 giây |
| Số chunk | 98 (trung bình 4,1 chunk/bài) |
| Số từ / chunk | trung bình 210 (min 57, max 230) |
| Thời gian embedding | khoảng 27 giây |
| Số chiều vector | 768, đã chuẩn hoá (‖v‖ = 1) |
| Theo nguồn | VnExpress 42, Tuổi Trẻ 32, Thanh Niên 24 chunk |

## 7. Cấu hình qua biến môi trường

Có thể đặt các biến sau trong file `.env`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `FAISS_PATH` | `./faiss_data` | Thư mục lưu index |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Model embedding |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | Tiền tố cho văn bản (để rỗng khi dùng `BAAI/bge-m3`) |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size khi encode |
| `MAX_ARTICLES_PER_FEED` | `0` | Giới hạn số bài mỗi feed, 0 là không giới hạn |
| `LOG_LEVEL` | `INFO` | Mức log |

> Đổi `EMBEDDING_MODEL` thì số chiều vector cũng đổi. Khi đó cần xoá `faiss_data/` (hoặc dùng `--reset`) rồi crawl lại.
