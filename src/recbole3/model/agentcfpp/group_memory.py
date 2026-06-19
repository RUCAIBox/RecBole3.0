from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.utils import ITEM_ID, USER_ID
from recbole3.model.agentcfpp.agents import GroupState
from recbole3.model.agentcfpp.config import AgentCFPPConfig
from recbole3.model.agentcfpp.prompts import GROUP_SUMMARY_PROMPT, USER_TAG_PROMPT

if TYPE_CHECKING:
    from recbole3.model.agentcfpp.model import AgentCFPPModel


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
    return np.where(norm == 0, x, x / norm)


def _extract_user_tags(model: "AgentCFPPModel", config: AgentCFPPConfig) -> dict[int, list[str]]:
    """For each user, ask the LLM to extract interest tags from their cross-domain memory."""
    client = model._get_llm_client()
    user_ids = sorted(model._user_agents.keys())

    messages = []
    for uid in user_ids:
        desc = model._user_agents[uid].all_private_description()
        messages.append([{"role": "user", "content": USER_TAG_PROMPT.substitute(user_description=desc)}])

    responses = client.chat_completion_batch(messages)
    user_tags: dict[int, list[str]] = {}
    for uid, resp in zip(user_ids, responses):
        tags: list[str] = []
        try:
            data = json.loads(resp)
            tags = [str(t) for t in data.get("interest_tags", [])]
        except (json.JSONDecodeError, AttributeError, TypeError):
            tags = []
        user_tags[uid] = tags
    return user_tags


def _cluster_tags(
    user_tags: dict[int, list[str]],
    model: "AgentCFPPModel",
    config: AgentCFPPConfig,
) -> dict[str, int]:
    """Embed unique tags and KMeans-cluster them. Returns tag -> cluster id."""
    from sklearn.cluster import KMeans

    unique_tags = sorted({tag for tags in user_tags.values() for tag in tags if tag})
    if not unique_tags:
        return {}

    client = model._get_llm_client()
    raw_embeddings = client.embedding_batch(unique_tags)

    dim = config.embedding_dim
    vectors = []
    kept_tags = []
    for tag, emb in zip(unique_tags, raw_embeddings):
        if not emb:
            continue
        vectors.append(np.asarray(emb[:dim], dtype=np.float64))
        kept_tags.append(tag)
    if not vectors:
        return {}

    matrix = _normalize_l2(np.vstack(vectors))
    n_clusters = max(1, min(config.group_n_cluster, len(kept_tags)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)
    return {tag: int(label) for tag, label in zip(kept_tags, labels)}


def _build_groups(
    user_tags: dict[int, list[str]],
    tag_to_cluster: dict[str, int],
    model: "AgentCFPPModel",
    config: AgentCFPPConfig,
) -> tuple[dict[int, list[int]], dict[int, list[str]]]:
    """Assign users to the top-N clusters. Returns (cluster -> users, cluster -> tags)."""
    cluster_tags: dict[int, list[str]] = {}
    for tag, cluster in tag_to_cluster.items():
        cluster_tags.setdefault(cluster, []).append(tag)

    # tag -> users
    tag_users: dict[str, list[int]] = {}
    for uid, tags in user_tags.items():
        for tag in tags:
            tag_users.setdefault(tag, []).append(uid)

    cluster_users: dict[int, list[int]] = {}
    for cluster, tags in cluster_tags.items():
        users: set[int] = set()
        for tag in tags:
            users.update(tag_users.get(tag, []))
        cluster_users[cluster] = sorted(users)

    top_clusters = sorted(cluster_users, key=lambda c: len(cluster_users[c]), reverse=True)[
        : config.group_num_groups
    ]
    return {c: cluster_users[c] for c in top_clusters}, {c: cluster_tags[c] for c in top_clusters}


def _summarize_group_names(
    cluster_tags: dict[int, list[str]],
    model: "AgentCFPPModel",
) -> dict[int, str]:
    """Ask the LLM to summarize each cluster's tags into a short group name phrase."""
    client = model._get_llm_client()
    clusters = sorted(cluster_tags.keys())
    messages = [
        [{"role": "user", "content": GROUP_SUMMARY_PROMPT.substitute(tag_list=", ".join(cluster_tags[c]))}]
        for c in clusters
    ]
    responses = client.chat_completion_batch(messages)
    names: dict[int, str] = {}
    for cluster, resp in zip(clusters, responses):
        name = str(resp).strip().strip('"').replace("\n", " ") or f"group_{cluster}"
        names[cluster] = name
    return names


def _build_group_memory_text(
    cluster_users: dict[int, list[int]],
    cluster_names: dict[int, str],
    model: "AgentCFPPModel",
    prepared_data: BaseTaskDataset,
    config: AgentCFPPConfig,
) -> GroupState:
    """Build per-group, per-domain recent-interaction text and user membership."""
    interactions = prepared_data.get_train_dataset()
    from recbole3.dataset.base import FrameDataset

    frame = interactions.frame if isinstance(interactions, FrameDataset) else None

    state = GroupState()
    user_to_groups: dict[int, list[str]] = {}

    for cluster, users in cluster_users.items():
        group_name = cluster_names.get(cluster, f"group_{cluster}")
        user_set = set(users)
        for uid in users:
            user_to_groups.setdefault(int(uid), []).append(group_name)

        domain_titles: dict[str, list[str]] = {}
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                uid = int(row[USER_ID])
                if uid not in user_set:
                    continue
                item_id = int(row[ITEM_ID])
                domain = model._item_domains.get(item_id, "")
                if not domain:
                    continue
                title = model._item_text_lookup[item_id] if item_id < len(model._item_text_lookup) else ""
                if title and title != "[PAD]":
                    domain_titles.setdefault(domain, []).append(title)

        group_domain_text: dict[str, str] = {}
        for domain, titles in domain_titles.items():
            recent = titles[-config.group_mem_length :]
            group_domain_text[domain] = f"{domain}: " + "; ".join(recent)
        state.groups[group_name] = group_domain_text

    state.user_to_groups = user_to_groups
    return state


def build_group_state(
    model: "AgentCFPPModel",
    prepared_data: BaseTaskDataset,
    config: AgentCFPPConfig,
) -> GroupState:
    """Run the full offline group-memory pipeline after training.

    user memory -> LLM tags -> embeddings -> KMeans -> top-N groups -> LLM names ->
    per-group recent-interaction memory.
    """
    print("[agentcfpp:group] extracting user interest tags")
    user_tags = _extract_user_tags(model, config)

    print("[agentcfpp:group] embedding + clustering tags")
    tag_to_cluster = _cluster_tags(user_tags, model, config)
    if not tag_to_cluster:
        print("[agentcfpp:group] no tags/embeddings produced; group memory disabled for this run")
        return GroupState()

    cluster_users, cluster_tags = _build_groups(user_tags, tag_to_cluster, model, config)
    if not cluster_users:
        return GroupState()

    print(f"[agentcfpp:group] summarizing {len(cluster_users)} group names")
    cluster_names = _summarize_group_names(cluster_tags, model)

    print("[agentcfpp:group] building per-group memory text")
    state = _build_group_memory_text(cluster_users, cluster_names, model, prepared_data, config)
    print(f"[agentcfpp:group] built {len(state.groups)} groups covering {len(state.user_to_groups)} users")
    return state


__all__ = ["build_group_state"]
