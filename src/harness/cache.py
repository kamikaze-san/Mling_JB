import sqlite3
import hashlib
import json
import os
import time
from typing import Optional, Dict, Any

class GenerationCache:
    """
    Persistent SQLite + JSONL generation cache.
    Ensures zero duplicate inference compute across laptop, Mac M4, and cloud runs.
    """
    def __init__(self, db_path: str = "./cache/generations.sqlite", jsonl_backup: Optional[str] = "./cache/generations.jsonl"):
        self.db_path = db_path
        self.jsonl_backup = jsonl_backup
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if self.jsonl_backup:
            os.makedirs(os.path.dirname(self.jsonl_backup), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT,
                    prompt TEXT,
                    response TEXT,
                    system_prompt TEXT,
                    temperature REAL,
                    max_new_tokens INTEGER,
                    tags TEXT,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON generations(model_id)")

    @staticmethod
    def compute_key(model_id: str, prompt: str, system_prompt: str = "", temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        payload = f"{model_id}::{system_prompt}::{prompt}::{temperature:.4f}::{max_new_tokens}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model_id: str, prompt: str, system_prompt: str = "", temperature: float = 0.0, max_new_tokens: int = 512) -> Optional[str]:
        cache_key = self.compute_key(model_id, prompt, system_prompt, temperature, max_new_tokens)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM generations WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def set(self, model_id: str, prompt: str, response: str, system_prompt: str = "", temperature: float = 0.0, max_new_tokens: int = 512, tags: Optional[Dict[str, Any]] = None):
        cache_key = self.compute_key(model_id, prompt, system_prompt, temperature, max_new_tokens)
        tags_json = json.dumps(tags or {})
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO generations 
                (cache_key, model_id, prompt, response, system_prompt, temperature, max_new_tokens, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cache_key, model_id, prompt, response, system_prompt, temperature, max_new_tokens, tags_json, now))

        # Append to JSONL for easy inspection and remote rsync
        if self.jsonl_backup:
            record = {
                "cache_key": cache_key,
                "model_id": model_id,
                "prompt": prompt,
                "response": response,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "tags": tags or {},
                "created_at": now
            }
            with open(self.jsonl_backup, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

