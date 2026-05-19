from __future__ import annotations

import hashlib


AVAZU_NUM_FEATURES = 22
AVAZU_FEATURE_COLUMNS = tuple(f"feature_{index}" for index in range(AVAZU_NUM_FEATURES))
AVAZU_SPLIT_COLUMN = "_source_split"


def hash_feature(*, field_id: int, value: str, hash_size: int) -> int:
    feature_id = f"{field_id}_{value}" if value else f"{field_id}_"
    digest = hashlib.md5(feature_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % int(hash_size)


__all__ = [
    "AVAZU_FEATURE_COLUMNS",
    "AVAZU_NUM_FEATURES",
    "AVAZU_SPLIT_COLUMN",
    "hash_feature",
]