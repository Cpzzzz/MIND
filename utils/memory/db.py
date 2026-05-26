import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List

import requests

from utils.memory.maintenance import maintain_memory as algorithm_mm


class MemorySystem:
    """
    JSONL-backed memory system for MIND.

    It stores all successful optimizations in full history and maintains a
    bounded active memory through the MM algorithm.
    """

    def __init__(
        self,
        persist_directory: str = "artifacts/memory/shared",
        active_cap: int = 300,
        update_batch_size: int = 100,
        embedding_model: str = None,
        mm_alpha: float = 1.0,
        mm_beta: float = 1.0,
        lambda_val: float = 0.5,
    ):
        self.root_dir = Path(persist_directory)
        self.active_cap = active_cap
        self.update_batch_size = update_batch_size
        self.embedding_model = embedding_model
        self.mm_alpha = mm_alpha
        self.mm_beta = mm_beta
        self.lambda_val = lambda_val
        self.lock = threading.Lock()

        self.jsonl_dir = self.root_dir / "jsonl"
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.jsonl_dir / "active_memory.jsonl"
        self.history_file = self.jsonl_dir / "full_history.jsonl"
        self.pending_file = self.jsonl_dir / "pending_new_examples.jsonl"

        self.active_file.touch(exist_ok=True)
        self.history_file.touch(exist_ok=True)
        self.pending_file.touch(exist_ok=True)

    def _load_jsonl(self, path: Path) -> List[Dict]:
        data = []
        if not path.exists():
            return data
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as error:
            logging.error("Failed to load JSONL from %s: %s", path, error)
        return data

    def _save_jsonl(self, path: Path, data: List[Dict], mode: str = "w"):
        try:
            with open(path, mode, encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as error:
            logging.error("Failed to save JSONL to %s: %s", path, error)

    def add_to_history(self, examples: List[Dict]):
        if not examples:
            return

        cleaned_examples = []
        for example in examples:
            cleaned = example.copy()
            cleaned.pop("ori_embedding", None)
            cleaned.pop("_tmp_idx", None)
            cleaned.pop("quality", None)
            cleaned_examples.append(cleaned)

        with self.lock:
            self._save_jsonl(self.history_file, cleaned_examples, mode="a")
        logging.info("Added %s examples to full memory history.", len(cleaned_examples))

    def update_active_memory(self, new_examples: List[Dict]):
        if not new_examples:
            return

        with self.lock:
            pending = self._load_jsonl(self.pending_file)
            for example in new_examples:
                cleaned = example.copy()
                cleaned.pop("ori_embedding", None)
                cleaned.pop("_tmp_idx", None)
                cleaned.pop("quality", None)
                pending.append(cleaned)

            if len(pending) < self.update_batch_size:
                self._save_jsonl(self.pending_file, pending, mode="w")
                logging.info(
                    "Buffered %s/%s new examples before the next MM update.",
                    len(pending),
                    self.update_batch_size,
                )
                return

            current_active = self._load_jsonl(self.active_file)
            updated = algorithm_mm(
                new_examples=pending,
                memory_limit=self.active_cap,
                current_memory=current_active,
                embedding_model=self.embedding_model,
                alpha=self.mm_alpha,
                beta=self.mm_beta,
                lambda_val=self.lambda_val,
            )
            if pending and not updated:
                logging.error("MM update produced empty memory; keeping active memory and pending examples unchanged.")
                self._save_jsonl(self.pending_file, pending, mode="w")
                return

            cleaned = []
            for example in updated:
                item = example.copy()
                item.pop("ori_embedding", None)
                item.pop("_tmp_idx", None)
                item.pop("quality", None)
                cleaned.append(item)
            self._save_jsonl(self.active_file, cleaned, mode="w")
            self._save_jsonl(self.pending_file, [], mode="w")

        logging.info("Active memory updated. Size: %s. Consumed %s pending examples.", len(cleaned), len(pending))

    def get_all_active(self) -> List[Dict]:
        with self.lock:
            return self._load_jsonl(self.active_file)


class RemoteMemorySystem:
    """Client-side adapter for the shared MIND memory server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def add_to_history(self, examples: List[Dict]):
        response = requests.post(f"{self.server_url}/history/add", json={"records": examples}, timeout=30)
        response.raise_for_status()

    def update_active_memory(self, new_examples: List[Dict]):
        timeout = float(os.getenv("MIND_MEMORY_UPDATE_TIMEOUT", "300"))
        response = requests.post(f"{self.server_url}/active/update", json={"records": new_examples}, timeout=timeout)
        response.raise_for_status()

    def get_all_active(self) -> List[Dict]:
        try:
            response = requests.get(f"{self.server_url}/active/all", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            logging.error("[Remote] Failed to get active memory: %s", error)
            return []
