from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from recbole3.dataset.base import BaseTaskDataset, FrameDataset
from recbole3.dataset.config import DatasetConfig, SplitConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, LABEL, TIMESTAMP, USER_ID


# Extra item-table columns carried through prepare() for the AgentCF++ model.
DOMAIN = "domain"
MAIN_CATEGORY = "main_category"


def _default_domain_main_category() -> dict[str, str]:
    """Domain -> Amazon `main_category` label (ported from AgentCF-plus/config.py)."""
    return {
        "Books": "Books",
        "Movies_and_TV": "Movies & TV",
        "Beauty_and_Personal_Care": "All Beauty",
        "Electronics": "All Electronics",
        "Sports_and_Outdoors": "Sports & Outdoors",
        "CDs_and_Vinyl": "Digital Music",
        "Video_Games": "Video Games",
    }


@dataclass(slots=True)
class AgentCFPPCrossConfig(DatasetConfig):
    """Configuration for the AgentCF++ cross-domain dataset.

    Two data sources are supported via `source`:

    - `amazon2023` (default): reuse the framework's Amazon Reviews 2023 downloader.
      For each domain in `domain_list` (each a valid Amazon 2023 category), the
      reviews + metadata are downloaded automatically (HuggingFace / ModelScope),
      merged into a cross-domain dataset, filtered to cross-domain users, split
      leave-one-out by time, and per-domain candidate pools are sampled. Users
      only specify the domains; no files need to be prepared by hand.

    - `local_csv`: read pre-built cross-domain CSV files (the original AgentCF++
      layout). Use this when you already have your own cross-domain data.
    """

    name: str = field(default="agentcfpp_cross", metadata={"help": "Dataset name."})
    source: Literal["amazon2023", "local_csv"] = field(
        default="amazon2023",
        metadata={"help": "Where cross-domain data comes from: auto-download or local CSV files."},
    )
    domain_list: tuple[str, ...] = field(
        default_factory=lambda: ("Books", "Video_Games", "Movies_and_TV"),
        metadata={"help": "Ordered list of domains (each a valid Amazon 2023 category for source=amazon2023)."},
    )
    domain_main_category_dict: dict[str, str] = field(
        default_factory=_default_domain_main_category,
        metadata={"help": "Mapping from domain name to Amazon main_category label."},
    )

    # --- source=amazon2023 ---
    download_source: Literal["huggingface", "modelscope"] = field(
        default="modelscope",
        metadata={"help": "Remote source used to snapshot Amazon 2023 raw data."},
    )
    kcore: Literal["full", "5core"] = field(
        default="5core",
        metadata={"help": "Amazon 2023 review subset to download per domain."},
    )
    download_dir: str = field(default="data/raw", metadata={"help": "Raw Amazon 2023 cache root."})
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed cache root."})
    max_users: int = field(default=100, metadata={"help": "Cap on cross-domain users kept (0 = no cap)."})
    min_domains_per_user: int = field(
        default=2,
        metadata={"help": "Keep only users who interacted in at least this many domains."},
    )
    min_inter_per_user: int = field(default=2, metadata={"help": "Drop users with fewer interactions than this."})
    n_random_item: int = field(default=100, metadata={"help": "Candidate pool size sampled per user per domain."})
    sample_seed: int = field(default=42, metadata={"help": "Seed for user selection and candidate sampling."})
    dump_dir: str = field(
        default="",
        metadata={
            "help": "If set, write the built cross-domain data here as local_csv-format files "
            "(meta + train/test inter + per-domain random pools). Re-readable via source=local_csv."
        },
    )

    # --- source=local_csv ---
    data_dir: str = field(default="", metadata={"help": "Root directory containing the CSV files (local_csv)."})
    inter_train_file: str = field(
        default="inter_crossdomain_timesequence_train.csv",
        metadata={"help": "Training interaction CSV filename (local_csv)."},
    )
    inter_test_file: str = field(
        default="inter_crossdomain_timesequence_test.csv",
        metadata={"help": "Test interaction CSV filename (local_csv)."},
    )
    meta_file: str = field(default="meta_crossdomain.csv", metadata={"help": "Item metadata CSV filename (local_csv)."})
    random_files: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Per-domain candidate CSV filenames, aligned with domain_list order (local_csv)."},
    )
    user_col: str = field(default="user_id", metadata={"help": "User id column in interaction/meta CSVs."})
    item_col: str = field(default="parent_asin", metadata={"help": "Item id column in interaction/meta CSVs."})
    random_user_col: str = field(
        default="Unnamed: 0",
        metadata={"help": "User id column in the per-domain random candidate CSVs."},
    )

    split: SplitConfig = field(
        default_factory=lambda: SplitConfig(strategy="leave_one_out", order="chronological", per_user=True),
        metadata={"help": "Split config (data is pre-split by file/time)."},
    )


class AgentCFPPCrossParser(BaseDatasetParser):
    """Parser producing a cross-domain dataset, either auto-downloaded or from local CSVs."""

    config_cls = AgentCFPPCrossConfig
    config: AgentCFPPCrossConfig

    def __init__(self, config: AgentCFPPCrossConfig):
        super().__init__(config)
        # Raw (un-remapped) per-domain candidate pools: {domain: {raw_user_id: [raw_item_id, ...]}}.
        self._raw_candidate_pools: dict[str, dict[str, list[str]]] = {}

    @property
    def data_dir(self) -> Path:
        return Path(self.config.data_dir)

    def parse(self) -> ParsedData:
        if self.config.source == "amazon2023":
            return self._parse_amazon2023()
        return self._parse_local_csv()

    # ==================== source = amazon2023 ====================

    def _parse_amazon2023(self) -> ParsedData:
        from recbole3.dataset.amazon2023.utils import AMAZON2023_AVAILABLE_CATEGORIES

        domain_list = list(self.config.domain_list)
        inter_frames = []

        for domain in domain_list:
            if domain not in AMAZON2023_AVAILABLE_CATEGORIES:
                raise ValueError(
                    f"Domain '{domain}' is not a valid Amazon 2023 category. "
                    f"Available: {', '.join(AMAZON2023_AVAILABLE_CATEGORIES)}"
                )
            print(f"[agentcfpp_cross] downloading Amazon 2023 ratings for domain '{domain}' (kcore={self.config.kcore})")
            inter = self._download_ratings(domain)
            inter = inter.loc[:, [USER_ID, ITEM_ID, TIMESTAMP]].copy()
            inter[DOMAIN] = domain
            inter_frames.append(inter)

        interactions = pd.concat(inter_frames, ignore_index=True)

        # Filter to cross-domain users FIRST so we only need titles for kept items.
        interactions = self._filter_cross_domain_users(interactions)
        interactions = self._split_leave_one_out(interactions)

        kept_items_by_domain: dict[str, set[str]] = {}
        for domain in domain_list:
            kept_items_by_domain[domain] = set(
                interactions.loc[interactions[DOMAIN] == domain, ITEM_ID].astype(str)
            )

        item_table = self._build_item_table(kept_items_by_domain)
        self._raw_candidate_pools = self._sample_candidate_pools(interactions, item_table)

        interactions[LABEL] = None

        if self.config.dump_dir:
            self._dump_to_csv(interactions, item_table, self._raw_candidate_pools)

        return ParsedData(interactions=interactions, item_table=item_table)

    def _dump_to_csv(
        self,
        interactions: pd.DataFrame,
        item_table: pd.DataFrame,
        candidate_pools: dict[str, dict[str, list[str]]],
    ) -> None:
        """Persist the built cross-domain data as local_csv-format files.

        The layout written here is exactly what `source=local_csv` reads back, so
        you can inspect the split or re-run without re-downloading:
            <dump_dir>/meta_crossdomain.csv
            <dump_dir>/inter_crossdomain_timesequence_train.csv
            <dump_dir>/inter_crossdomain_timesequence_test.csv
            <dump_dir>/random/random_<domain>.csv
        """
        dump_dir = Path(self.config.dump_dir)
        (dump_dir / "random").mkdir(parents=True, exist_ok=True)

        user_col = self.config.user_col
        item_col = self.config.item_col

        # Item metadata: parent_asin,title,main_category,categories,price,subtitle.
        meta_out = pd.DataFrame({item_col: item_table[ITEM_ID].astype(str)})
        meta_out["title"] = item_table["title"].astype(str) if "title" in item_table.columns else ""
        meta_out[MAIN_CATEGORY] = item_table[MAIN_CATEGORY].astype(str) if MAIN_CATEGORY in item_table.columns else ""
        for col in ("categories", "price", "subtitle"):
            meta_out[col] = item_table[col].astype(str) if col in item_table.columns else ""
        meta_out.to_csv(dump_dir / self.config.meta_file, index=False)

        # Interactions split into train/test by the _split column.
        for split, filename in (("train", self.config.inter_train_file), ("test", self.config.inter_test_file)):
            subset = interactions[interactions["_split"] == split]
            inter_out = pd.DataFrame(
                {user_col: subset[USER_ID].astype(str), item_col: subset[ITEM_ID].astype(str)}
            )
            inter_out.to_csv(dump_dir / filename, index=False)

        # Per-domain candidate pools: first column matches random_user_col, then item columns.
        for domain in self.config.domain_list:
            domain_pool = candidate_pools.get(domain, {})
            rows = []
            for raw_user, items in domain_pool.items():
                row = {self.config.random_user_col: raw_user}
                for idx, item in enumerate(items):
                    row[f"item_{idx}"] = item
                rows.append(row)
            pool_frame = pd.DataFrame(rows)
            pool_frame.to_csv(dump_dir / "random" / f"random_{domain}.csv", index=False)

        print(f"[agentcfpp_cross] dumped local_csv-format data to {dump_dir}")

    def _hf_repo_id(self) -> str:
        return "McAuley-Lab/Amazon-Reviews-2023"

    def _download_ratings(self, domain: str) -> pd.DataFrame:
        """Download the rating-only interaction CSV for one domain via the HF Hub.

        Bypasses the (now-removed) dataset-script loader, so it works with
        modern `datasets`/`huggingface_hub` without downgrading anything.
        """
        from huggingface_hub import hf_hub_download

        rel_path = f"benchmark/{self.config.kcore}/rating_only/{domain}.csv"
        local_path = hf_hub_download(
            self._hf_repo_id(),
            rel_path,
            repo_type="dataset",
            cache_dir=str(Path(self.config.download_dir) / "amazon2023_hf"),
        )
        frame = pd.read_csv(local_path)
        frame = frame.rename(columns={"user_id": USER_ID, "parent_asin": ITEM_ID, "timestamp": TIMESTAMP})
        return frame

    def _build_item_table(self, kept_items_by_domain: dict[str, set[str]]) -> pd.DataFrame:
        """Stream each domain's metadata jsonl, keeping titles only for kept items."""
        rows: list[dict[str, str]] = []
        for domain, kept_items in kept_items_by_domain.items():
            main_category = self.config.domain_main_category_dict.get(domain, domain)
            if not kept_items:
                continue
            titles = self._download_item_titles(domain, kept_items)
            for item_id in kept_items:
                rows.append(
                    {
                        ITEM_ID: item_id,
                        "title": titles.get(item_id, ""),
                        MAIN_CATEGORY: main_category,
                        DOMAIN: domain,
                    }
                )
        item_table = pd.DataFrame(rows, columns=[ITEM_ID, "title", MAIN_CATEGORY, DOMAIN])
        return item_table.drop_duplicates(subset=[ITEM_ID], keep="first").reset_index(drop=True)

    def _download_item_titles(self, domain: str, kept_items: set[str]) -> dict[str, str]:
        """Download a domain's metadata jsonl and extract titles for kept items only."""
        import json

        from huggingface_hub import hf_hub_download

        rel_path = f"raw/meta_categories/meta_{domain}.jsonl"
        print(f"[agentcfpp_cross] fetching item titles for domain '{domain}' ({len(kept_items)} items)")
        local_path = hf_hub_download(
            self._hf_repo_id(),
            rel_path,
            repo_type="dataset",
            cache_dir=str(Path(self.config.download_dir) / "amazon2023_hf"),
        )
        titles: dict[str, str] = {}
        with open(local_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = str(record.get("parent_asin", ""))
                if asin in kept_items:
                    titles[asin] = str(record.get("title", "")).strip()
                    if len(titles) == len(kept_items):
                        break
        return titles

    def _filter_cross_domain_users(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Keep users active across >= min_domains_per_user domains, capped at max_users."""
        per_user_domains = interactions.groupby(USER_ID)[DOMAIN].nunique()
        per_user_count = interactions.groupby(USER_ID).size()
        eligible = per_user_domains[
            (per_user_domains >= self.config.min_domains_per_user)
            & (per_user_count >= self.config.min_inter_per_user)
        ].index

        eligible = sorted(eligible, key=lambda u: (-int(per_user_count[u]), str(u)))
        if self.config.max_users and len(eligible) > self.config.max_users:
            eligible = eligible[: self.config.max_users]

        kept = set(eligible)
        if not kept:
            raise ValueError(
                "No cross-domain users found after filtering. Try lowering min_domains_per_user / "
                "min_inter_per_user, using kcore=full, or choosing overlapping domains."
            )
        print(f"[agentcfpp_cross] kept {len(kept)} cross-domain users")
        return interactions.loc[interactions[USER_ID].isin(kept)].reset_index(drop=True)

    def _split_leave_one_out(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Per user, hold out the latest interaction as test; the rest is train."""
        frame = interactions.copy()
        frame[TIMESTAMP] = pd.to_numeric(frame[TIMESTAMP], errors="coerce").fillna(0)
        frame = frame.sort_values([USER_ID, TIMESTAMP]).reset_index(drop=True)

        split = np.array(["train"] * len(frame), dtype=object)
        last_positions = frame.groupby(USER_ID, sort=False).tail(1).index
        for pos in last_positions:
            split[pos] = "test"
        frame["_split"] = split
        # Re-stamp synthetic increasing timestamps to keep ordering deterministic downstream.
        frame[TIMESTAMP] = range(len(frame))
        return frame

    def _sample_candidate_pools(
        self,
        interactions: pd.DataFrame,
        item_table: pd.DataFrame,
    ) -> dict[str, dict[str, list[str]]]:
        """For each domain and user, sample n_random_item items from that domain."""
        rng = np.random.default_rng(self.config.sample_seed)
        pools: dict[str, dict[str, list[str]]] = {}
        users = [str(u) for u in interactions[USER_ID].unique()]

        for domain in self.config.domain_list:
            domain_items = item_table.loc[item_table[DOMAIN] == domain, ITEM_ID].astype(str).tolist()
            if not domain_items:
                pools[domain] = {}
                continue
            n = min(self.config.n_random_item, len(domain_items))
            domain_pool: dict[str, list[str]] = {}
            for user in users:
                idx = rng.choice(len(domain_items), size=n, replace=False)
                domain_pool[user] = [domain_items[i] for i in idx]
            pools[domain] = domain_pool
        return pools

    # ==================== source = local_csv ====================

    def _parse_local_csv(self) -> ParsedData:
        data_dir = self.data_dir
        item_table = self._read_meta_file(data_dir / self.config.meta_file)

        train_inter = self._read_inter_file(data_dir / self.config.inter_train_file)
        test_inter = self._read_inter_file(data_dir / self.config.inter_test_file)

        train_inter[TIMESTAMP] = range(len(train_inter))
        offset = len(train_inter)
        test_inter[TIMESTAMP] = range(offset, offset + len(test_inter))
        train_inter["_split"] = "train"
        test_inter["_split"] = "test"

        interactions = pd.concat([train_inter, test_inter], ignore_index=True)
        self._raw_candidate_pools = self._read_candidate_pools(data_dir)
        return ParsedData(interactions=interactions, item_table=item_table)

    def _read_inter_file(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype=str)
        return pd.DataFrame(
            {
                USER_ID: frame[self.config.user_col].astype(str),
                ITEM_ID: frame[self.config.item_col].astype(str),
                LABEL: None,
            }
        )

    def _read_meta_file(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype=str)
        reverse_map = {v: k for k, v in self.config.domain_main_category_dict.items()}

        out = pd.DataFrame({ITEM_ID: frame[self.config.item_col].astype(str)})
        for col in ("title", MAIN_CATEGORY, "categories", "price", "subtitle"):
            out[col] = frame[col].astype(str) if col in frame.columns else ""
        out[DOMAIN] = out[MAIN_CATEGORY].map(lambda mc: reverse_map.get(str(mc).strip(), ""))
        out = out.drop_duplicates(subset=[ITEM_ID], keep="first").reset_index(drop=True)
        return out

    def _read_candidate_pools(self, data_dir: Path) -> dict[str, dict[str, list[str]]]:
        pools: dict[str, dict[str, list[str]]] = {}
        domain_list = list(self.config.domain_list)
        random_files = list(self.config.random_files)

        for idx, domain in enumerate(domain_list):
            if idx >= len(random_files) or not random_files[idx]:
                continue
            path = data_dir / random_files[idx]
            if not path.exists():
                print(f"[agentcfpp_cross] candidate file missing for domain '{domain}': {path}")
                continue
            frame = pd.read_csv(path, dtype=str)
            user_col = self.config.random_user_col
            item_cols = [c for c in frame.columns if c != user_col]
            domain_pool: dict[str, list[str]] = {}
            for _, row in frame.iterrows():
                raw_user = str(row[user_col])
                items = [str(row[c]) for c in item_cols if pd.notna(row[c]) and str(row[c]) != "0"]
                domain_pool[raw_user] = items
            pools[domain] = domain_pool
        return pools


class AgentCFPPCrossDataset(BaseTaskDataset):
    """AgentCF++ cross-domain dataset with pre-split train/test (auto-built or from CSV).

    The data has no separate validation split, so test is reused as validation.
    """

    config_cls = AgentCFPPCrossConfig
    parser_cls = AgentCFPPCrossParser
    config: AgentCFPPCrossConfig

    def _build_prepared_datasets(self) -> None:
        interactions = self._interactions

        if "_split" in interactions.columns:
            train_frame = (
                interactions[interactions["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
            )
            test_interactions = (
                interactions[interactions["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
            )

            protocol = self._require_eval_config().protocol
            if protocol in {"full", "sampled"}:
                test_frame = self._build_eval_frame(
                    test_interactions,
                    seen_history_interactions=train_frame,
                    split="test",
                )
                valid_frame = test_frame.copy()
            else:
                valid_frame = test_interactions
                test_frame = test_interactions

            self._train_dataset = FrameDataset(train_frame)
            self._valid_dataset = FrameDataset(valid_frame)
            self._test_dataset = FrameDataset(test_frame)
        else:
            super()._build_prepared_datasets()

    def get_item_domains(self) -> dict[int, str]:
        """Return framework item_id -> domain name."""
        self._require_prepared()
        if DOMAIN not in self._item_table.columns:
            return {}
        return {int(row[ITEM_ID]): str(row[DOMAIN]) for _, row in self._item_table.iterrows()}

    def get_domain_candidate_pools(self) -> dict[str, dict[int, list[int]]]:
        """Return per-domain candidate pools in framework ids.

        Shape: {domain: {framework_user_id: [framework_item_id, ...]}}. Raw ids
        absent from the id maps are dropped.
        """
        self._require_prepared()
        raw_pools = getattr(self._parser, "_raw_candidate_pools", {})
        user_id_map = getattr(self, "_user_id_map", {})
        item_id_map = getattr(self, "_item_id_map", {})

        mapped: dict[str, dict[int, list[int]]] = {}
        for domain, domain_pool in raw_pools.items():
            mapped_domain: dict[int, list[int]] = {}
            for raw_user, raw_items in domain_pool.items():
                if raw_user not in user_id_map:
                    continue
                fw_user = int(user_id_map[raw_user])
                fw_items = [int(item_id_map[i]) for i in raw_items if i in item_id_map]
                if fw_items:
                    mapped_domain[fw_user] = fw_items
            mapped[domain] = mapped_domain
        return mapped


__all__ = [
    "AgentCFPPCrossConfig",
    "AgentCFPPCrossDataset",
    "AgentCFPPCrossParser",
    "DOMAIN",
    "MAIN_CATEGORY",
]
