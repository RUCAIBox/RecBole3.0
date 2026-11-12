"""RecBole histories and train-only Swing neighborhoods for ManCAR."""

# Upstream: https://github.com/FuCongResearchSquad/ManCAR/tree/0a259aa1988892b3edd938437671e6b90190a423
# Copyright (c) 2026 ManCAR Authors. See the MIT notice in this package's __init__.py.

from collections import defaultdict
import heapq
import itertools
import logging
import math
import random

import pandas as pd
import torch

from recbole3.dataset import ITEM_ID, LABEL, USER_ID
from recbole3.model.base import BaseCollator
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS

from .layers import MAX_ITEM_SEQ_LEN


class ManCARModelDataset(BaseSequentialModelDataset):
    @classmethod
    def from_task_dataset(cls, dataset, *, model_config):
        prepared = super().from_task_dataset(dataset, model_config=model_config)
        if raw_ids := getattr(dataset, "_user_id_map", None):
            raw_users = {mapped: raw for raw, mapped in raw_ids.items()}
            train = prepared.get_train_dataset().frame
            train["source_user_id"] = train[USER_ID].map(raw_users)
        # Match the reference CSV export's first-occurrence numbering.
        # Keep the task IDs intact for the shared evaluator.
        num_items = prepared.get_num_items()
        order = dict.fromkeys(
            item
            for split in (
                prepared.get_train_dataset(),
                prepared.get_eval_dataset("valid"),
                prepared.get_eval_dataset("test"),
            )
            for item in split.frame[ITEM_ID]
        )
        order.update(dict.fromkeys(range(num_items)))
        item_ids = [0] * num_items
        for source_id, task_id in enumerate(order, start=1):
            item_ids[task_id] = source_id
        prepared.mancar_item_ids = tuple(item_ids)
        return prepared


def build_swing(train, config, *, item_ids):
    """Build Swing from complete train prefixes, before the model's history cap."""
    if train.empty:
        return {}
    user_column = "source_user_id" if "source_user_id" in train else USER_ID
    prefixes = train.loc[train[user_column].duplicated(keep="last")]
    if LABEL in prefixes:
        labels = prefixes[LABEL]
        prefixes = prefixes.loc[labels.isna() | pd.to_numeric(labels).gt(0)]
    histories = (
        prefixes.groupby(user_column, sort=True)[ITEM_ID]
        .agg(lambda items: [item_ids[int(item)] for item in items])
        .rename(HISTORY_ITEM_IDS)
        .rename_axis(USER_ID)
        .reset_index()
    )
    swing = _Swing(config)
    item2user = swing._gen_item2user(histories)
    user2item = {user: set(items) for user, items in histories.itertuples(index=False, name=None)}
    return swing._build_swing_ali(histories[HISTORY_ITEM_IDS], config.swing_topk, item2user, user2item)


class _Swing:
    def __init__(self, config):
        self.swing_alpha1 = config.swing_alpha1
        self.swing_alpha2 = config.swing_alpha2
        self.swing_beta = config.swing_beta
        self.swing_m = config.swing_m
        self.swing_min_u = config.swing_min_u
        self.rng = random.Random(config.graph_seed)

    def topk_per_item(self, counter: dict, k: int):
        """
        Convert undirected pair scores into directed top‑k neighbor lists.

        Parameters
        ----------
        counter : dict[(i,j)] -> float
            Accumulated symmetrical scores where i<j.
        k : int
            Keep only top-k neighbors per item.

        Returns
        -------
        dict[item] -> List[(nbr, score)]
        """
        neigh = defaultdict(list)
        for (i, j), w in counter.items():
            neigh[i].append((w, j))
            neigh[j].append((w, i))

        return {
            itm: [(j, w) for w, j in heapq.nlargest(k, lst)]
            for itm, lst in neigh.items()
        }

    def _gen_item2user(self, df):
        """
        Build item->set(users) from a DataFrame containing [USER_ID, ITEM_SEQ_ID].

        Returns
        -------
        dict[item] -> set(users)
        """
        item2user = defaultdict(set)
        for u, items in df.itertuples(index=False, name=None):
            for i in items:
                item2user[i].add(u)
        return item2user

    def _build_swing_ali(self, data, topk, item2user, user2item):
        """
        Alibaba Swing variant (paper/formula with α1, α2, β, M sampling).

        data      : iterable of lists (user baskets)
        item2user : dict[item] -> set(users)
        user2item : dict[user] -> set(items)
        """
        logging.info(f"swing ali parameters: a1={self.swing_alpha1}, a2={self.swing_alpha2}, b={self.swing_beta}, M={self.swing_m}, min_common_user_num={self.swing_min_u}")
        counter = defaultdict(float)

        w3_dict = {item: 1.0 / math.sqrt(len(item2user[item])) for item in item2user.keys()}
        _uv_cache = {}

        def _sim(i, j, alpha1=5.0, alpha2=1.0, beta=0.3, M=10, min_common_user_num=2):
            """
            Compute sim(i,j) with truncated user pairs and cached per-(u,v) weights.
            """
            users = list(item2user[i] & item2user[j])
            if len(users) > M:
                users = self.rng.sample(users, M)

            w3 = w3_dict.get(j)
            if w3 is None or len(users) < min_common_user_num:
                return 0.0

            total = 0.0
            for u, v in itertools.combinations(users, 2):
                key = (u, v) if u < v else (v, u)
                w_uv = _uv_cache.get(key)
                if w_uv is None:
                    Iu = user2item[u]; Iv = user2item[v]
                    inter_uv = len(Iu & Iv)
                    if inter_uv == 0:
                        w_uv = 0.0
                    else:
                        w1 = 1.0 / ((len(Iu)+alpha1)**beta * (len(Iv)+alpha1)**beta)
                        w2 = 1.0 / (inter_uv + alpha2)
                        w_uv = w1 * w2
                    _uv_cache[key] = w_uv
                total += w_uv * 2
            return total * w3

        for seq in data:
            for i, j in itertools.combinations(seq, 2):
                key_ij = (i, j) if i < j else (j, i)
                w = _sim(i, j,
                         alpha1=self.swing_alpha1, alpha2=self.swing_alpha2,
                         beta=self.swing_beta, M=self.swing_m,
                         min_common_user_num=self.swing_min_u)
                counter[key_ij] += w

        return self.topk_per_item(counter, topk)


class ManCARCollator(BaseCollator):
    def __init__(self, config, prepared_data, *, neighbors, training):
        super().__init__(config, prepared_data)
        self.neighbors = neighbors
        self.training = training
        self.num_items = prepared_data.get_num_items()
        self.item_ids = torch.tensor(prepared_data.mancar_item_ids, dtype=torch.long)

    def __call__(self, records):
        if isinstance(records, list):
            records = pd.DataFrame(records)
        b = len(records)
        c = self.config
        history = torch.zeros((b, MAX_ITEM_SEQ_LEN), dtype=torch.long)
        lengths = []
        prompts = []
        triggers = []
        priors = []
        for row, h in enumerate(records[HISTORY_ITEM_IDS]):
            ids = [int(i) for i in h[-MAX_ITEM_SEQ_LEN:]]
            if any(i < 0 or i >= self.num_items for i in ids):
                raise ValueError("History ID outside item vocabulary.")
            # SwingReader keeps the original length even after cropping IDs to 50.
            lengths.append(len(h))
            if ids:
                history[row, -len(ids) :] = self.item_ids[ids]
            trigger = history[row, -c.trigger_num :].tolist()
            triggers.append(trigger)
            prompt = []
            for item in trigger:
                n = [int(i) for i, _ in self.neighbors.get(item, [])][
                    : c.items_per_trigger
                ]
                prompt.extend(n + [0] * (c.items_per_trigger - len(n)))
            prompts.append(prompt)
            n = [int(i) for i, _ in self.neighbors.get(int(history[row, -1]), [])][:10]
            priors.append(n + [0] * (10 - len(n)))
        batch = {
            "history_id": history,
            "item_seq_len": torch.tensor(lengths),
            "prompt_ids": torch.tensor(prompts, dtype=torch.long),
            "trigger_ids": torch.tensor(triggers, dtype=torch.long),
            "neighbor_item_i2i": torch.tensor(priors, dtype=torch.long),
        }
        # No evaluation target is passed to the encoder.
        if self.training:
            targets = torch.tensor(records[ITEM_ID].to_numpy(), dtype=torch.long)
            if ((targets < 0) | (targets >= self.num_items)).any():
                raise ValueError("Invalid target item.")
            batch[ITEM_ID] = self.item_ids[targets]
        return batch
