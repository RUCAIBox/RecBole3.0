from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recbole3.dataset.parser import ParsedData

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class DatasetCache:
    """Small JSONL-backed cache helper for dataset parsers."""

    root: PathLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def path(self, *parts: PathLike) -> Path:
        path = Path(self.root)
        for part in parts:
            path /= Path(part)
        return path

    def exists(self, *relative_paths: PathLike) -> bool:
        if not relative_paths:
            return self.path().exists()
        return all(self.path(relative_path).exists() for relative_path in relative_paths)

    def read_frame(
        self,
        relative_path: PathLike,
        *,
        required: bool = False,
        description: str | None = None,
    ) -> pd.DataFrame:
        path = self.path(relative_path)
        if not path.exists():
            if required:
                raise FileNotFoundError(f"{description or relative_path} not found at {path}.")
            return pd.DataFrame()
        if path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_json(path, lines=True, convert_dates=False)

    def write_frame(self, relative_path: PathLike, frame: pd.DataFrame | None) -> None:
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frame must be a pandas DataFrame or None, got {type(frame).__name__}.")
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if frame is None or frame.empty:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8") as handle:
            frame.to_json(handle, orient="records", lines=True, force_ascii=False)

    def get_or_create_frame(
        self,
        relative_path: PathLike,
        builder: Callable[[], pd.DataFrame],
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        if not force and self.path(relative_path).exists():
            return self.read_frame(relative_path, required=True)
        frame = builder()
        self.write_frame(relative_path, frame)
        return frame

    def parsed_exists(self) -> bool:
        return self.exists("interactions.jsonl", "users.jsonl", "items.jsonl")

    def read_parsed(self) -> ParsedData:
        return ParsedData(
            interactions=self.read_frame(
                "interactions.jsonl",
                required=True,
                description="Parsed interactions cache",
            ),
            user_table=self.read_frame(
                "users.jsonl",
                required=True,
                description="Parsed users cache",
            ),
            item_table=self.read_frame(
                "items.jsonl",
                required=True,
                description="Parsed items cache",
            ),
        )

    def write_parsed(self, parsed: ParsedData) -> None:
        self.write_frame("interactions.jsonl", parsed.interactions)
        self.write_frame("users.jsonl", parsed.user_table)
        self.write_frame("items.jsonl", parsed.item_table)


__all__ = [
    "DatasetCache",
]
