from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("blog_archive.sqlite3")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_blog_store() -> None:
    """Create the persistent blog archive if it does not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blogs (
                blog_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                title TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT '',
                markdown TEXT NOT NULL,
                image_specs TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blogs_thread ON blogs(thread_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blogs_created ON blogs(created_at)")


def save_blog(*, thread_id: str, title: str, topic: str, markdown: str,
              image_specs: Optional[list[dict[str, Any]]] = None) -> str:
    """Save a new blog archive entry and return its unique Blog ID."""
    init_blog_store()
    blog_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            INSERT INTO blogs
                (blog_id, thread_id, title, topic, markdown, image_specs, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (blog_id, thread_id, title.strip() or "Untitled blog", topic.strip(),
              markdown, json.dumps(image_specs or [], ensure_ascii=False, default=str), created_at))
    return blog_id


def list_blogs(search: str = "", limit: int = 500) -> list[dict[str, Any]]:
    """List newest blogs first, optionally searching title, topic, and body."""
    init_blog_store()
    term = f"%{search.strip()}%"
    with _connect() as conn:
        rows = conn.execute("""
            SELECT blog_id, thread_id, title, topic, created_at
            FROM blogs
            WHERE (? = '%%' OR title LIKE ? OR topic LIKE ?)
            ORDER BY created_at DESC LIMIT ?
        """, (term, term, term, limit)).fetchall()
    return [dict(row) for row in rows]


def get_blog(blog_id: str) -> Optional[dict[str, Any]]:
    init_blog_store()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM blogs WHERE blog_id = ?", (blog_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["image_specs"] = json.loads(result.get("image_specs") or "[]")
    return result
