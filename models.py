from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    """Một bài báo sau khi crawl."""
    title: str
    content: str
    url: str
    source: str
    category: str
    published_at: Optional[datetime] = None
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
