# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import sqlite3
import threading
import time
import json
import hashlib

from typing import Optional, Dict
from pathlib import Path

class Cache:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        # Ensure that the connection is set up for serialized mode, which is the default.
        self.conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0
        )  # Increased timeout
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()
        self.cache_hits = 0
        self.cache_misses = 0

    def set_item(self, key: str, value: Dict):
        hashed_key = self.hash_key(key)
        json_value = json.dumps(value)
        with self.lock:
            retry_count = 5
            while retry_count > 0:
                try:
                    self.conn.execute(
                        "REPLACE INTO cache (key, value) VALUES (?, ?)",
                        (hashed_key, json_value),
                    )
                    self.conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e):
                        retry_count -= 1
                        time.sleep(0.1)  # Wait a bit for the lock to be released
                    else:
                        raise

    def hash_key(self, key: str) -> str:
        """Generate a SHA-256 hash of the key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def get_item(self, key: str) -> Optional[Dict]:
        hashed_key = self.hash_key(key)
        with self.lock:
            cursor = self.conn.execute("SELECT value FROM cache WHERE key = ?", (hashed_key,))
            item = cursor.fetchone()
            if item:
                self.cache_hits += 1
                return json.loads(item[0])
            else:
                self.cache_misses += 1
                return None

    def __del__(self):
        self.conn.close()

    def cache_stats(self):
        cache_file = self.db_path
        print(f"Cache hits: {self.cache_hits}")
        print(f"Cache misses: {self.cache_misses}")
        print(
            f'Cache file size: {f"{os.path.getsize(cache_file) / (1024 * 1024 * 1024):.2f} GB"}'
        )
        total, used, free = shutil.disk_usage(os.getcwd())
        print(f"Free: {free / (1024 * 1024 * 1024):.2f} GB")
