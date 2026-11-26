"""
Step 1 跨次游標與耗盡狀態持久化模組（供 OpenAlex 與 Europe PMC 共用）
- 儲存每組查詢的 next_cursor 與 exhausted 狀態，讓 --append 直接接續深挖並跳過已見底查詢
- 使用臨時檔 + os.replace 原子存檔，避免中途強制中斷造成 JSON 損毀
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def make_cursor_key(
    source: str,
    query: str,
    from_year: int,
    sort: str = "",
) -> str:
    """建立跨次游標狀態的唯一鍵。"""
    return f"{source}||y>={from_year}||sort={sort}||{query}"


class Step1CursorStore:
    """管理單一金屬 Step 1 跨次查詢游標與耗盡標記的 JSON 狀態檔。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = {"version": 1, "entries": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
                self._data = {"version": int(raw.get("version", 1)), "entries": dict(raw["entries"])}
        except (OSError, ValueError, TypeError):
            self._data = {"version": 1, "entries": {}}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
        except OSError:
            pass

    def get(self, key: str) -> tuple[str | None, bool]:
        """回傳 (next_cursor, exhausted)。若無紀錄則回傳 (None, False)。"""
        entry = self._data["entries"].get(key)
        if not isinstance(entry, dict):
            return None, False
        next_cursor = entry.get("next_cursor")
        if next_cursor is not None and not str(next_cursor).strip():
            next_cursor = None
        exhausted = bool(entry.get("exhausted", False))
        return (str(next_cursor) if next_cursor is not None else None), exhausted

    def update(
        self,
        key: str,
        next_cursor: str | None,
        exhausted: bool,
        *,
        persist: bool = True,
    ) -> None:
        clean_cursor = str(next_cursor).strip() if next_cursor else None
        self._data["entries"][key] = {
            "next_cursor": None if exhausted else clean_cursor,
            "exhausted": bool(exhausted),
            "last_run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if persist:
            self.save()

    def reset(self, key: str, *, persist: bool = True) -> None:
        if key in self._data["entries"]:
            self._data["entries"].pop(key, None)
            if persist:
                self.save()
