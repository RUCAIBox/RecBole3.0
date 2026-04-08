from __future__ import annotations

from typing import Any


def load_huggingface_dataset(
    dataset_id: str,
    subset_name: str,
    *,
    split: str,
    cache_dir: str,
    trust_remote_code: bool,
) -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`datasets` is required when download_source='huggingface'. Install recbole3[huggingface]."
        ) from exc
    return load_dataset(
        dataset_id,
        subset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )


def load_modelscope_dataset(
    dataset_id: str,
    subset_name: str,
    *,
    split: str,
    trust_remote_code: bool,
) -> Any:
    try:
        from modelscope.msdatasets import MsDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`modelscope` is required when download_source='modelscope'. Install recbole3[modelscope]."
        ) from exc
    return MsDataset.load(
        dataset_id,
        subset_name=subset_name,
        split=split,
        trust_remote_code=trust_remote_code,
    )


__all__ = [
    "load_huggingface_dataset",
    "load_modelscope_dataset",
]
