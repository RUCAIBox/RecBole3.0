from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.utils import ITEM_ID
from recbole3.model.agentcfpp.agents import GroupState, ItemAgentState, UserAgentState
from recbole3.model.agentcfpp.config import AgentCFPPConfig
from recbole3.model.agentcfpp.data import AgentCFPPEvalCollator, AgentCFPPTrainCollator
from recbole3.model.agentcfpp.llm_client import LLMClient
from recbole3.model.agentcfpp.output_parser import (
    fuzzy_match_title,
    parse_crossdomain_update,
    parse_evaluation_ranking,
    parse_item_update,
    parse_recommendation,
    parse_user_update,
)
from recbole3.model.agentcfpp.prompts import (
    CROSSDOMAIN_PROMPT,
    EVAL_BASIC,
    EVAL_RETRIEVAL,
    EVAL_SEQUENTIAL,
    FORWARD_PROMPT,
    ITEM_PROMPT,
    ITEM_PROMPT_TRUE,
    USER_PROMPT,
    USER_PROMPT_TRUE,
    USER_SYSTEM_ROLE,
)
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS


class AgentCFPPModel(BaseRetrievalModel):
    """AgentCF++: cross-domain LLM-based collaborative filtering with dual-layer user
    memory and optional shared group memory."""

    def __init__(self, config: AgentCFPPConfig):
        super().__init__(config)
        self.config: AgentCFPPConfig = config
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._user_agents: dict[int, UserAgentState] = {}
        self._item_agents: dict[int, ItemAgentState] = {}
        self._group_state: GroupState = GroupState()
        self._item_text_lookup: tuple[str, ...] = ()
        self._item_domains: dict[int, str] = {}
        self._domain_candidate_pools: dict[str, dict[int, list[int]]] = {}
        self._num_items: int = 0
        self._num_users: int = 0
        self._llm_client: LLMClient | None = None
        self._initialized: bool = False

    # ==================== Setup ====================

    def set_cross_domain_context(
        self,
        *,
        item_domains: dict[int, str],
        domain_candidate_pools: dict[str, dict[int, list[int]]],
    ) -> None:
        """Injected by the pipeline before model-data cloning."""
        self._item_domains = dict(item_domains)
        self._domain_candidate_pools = {d: dict(p) for d, p in domain_candidate_pools.items()}

    def set_group_state(self, group_state: GroupState) -> None:
        self._group_state = group_state

    def _get_llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(
                api_base_url=self.config.api_base_url,
                api_key_env=self.config.api_key_env,
                model_name=self.config.api_model_name,
                embedding_model=self.config.embedding_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_chat,
                request_retries=self.config.request_retries,
                retry_backoff_sec=self.config.retry_backoff_sec,
                request_timeout_sec=self.config.request_timeout_sec,
                concurrency=self.config.chat_api_batch,
            )
        return self._llm_client

    def set_llm_client(self, client: LLMClient) -> None:
        """Override the LLM client (used by tests to inject a fake)."""
        self._llm_client = client

    def ensure_initialized(self, prepared_data: BaseTaskDataset) -> None:
        if self._initialized:
            return
        self._num_items = prepared_data.get_num_items()
        self._num_users = prepared_data.get_num_users()

        item_table = prepared_data.get_item_table()
        item_texts = ["[PAD]"] * self._num_items

        # Pull domains from the item_table if not already injected by the pipeline.
        from recbole3.dataset.agentcfpp_cross import DOMAIN as DOMAIN_COL

        has_domain_col = DOMAIN_COL in item_table.columns
        for _, row in item_table.iterrows():
            item_id = int(row[ITEM_ID])
            if not (0 <= item_id < self._num_items):
                continue
            title = str(row.get("title", "")) if "title" in item_table.columns else ""
            item_texts[item_id] = title
            domain = str(row.get(DOMAIN_COL, "")) if has_domain_col else self._item_domains.get(item_id, "")
            if item_id not in self._item_domains and domain:
                self._item_domains[item_id] = domain
            init_desc = self._build_item_init_description(row, item_table.columns)
            self._item_agents[item_id] = ItemAgentState(
                item_id=item_id,
                title=title,
                domain=self._item_domains.get(item_id, ""),
                initial_description=init_desc,
            )

        self._item_text_lookup = tuple(item_texts)

        # Initialize user agents with one init memory per domain they touch.
        for user_id in range(self._num_users):
            self._user_agents[user_id] = UserAgentState(user_id=user_id)
        self._seed_user_domains(prepared_data)

        if self.config.load_agent_state_path:
            self._load_agent_states(Path(self.config.load_agent_state_path))

        self._initialized = True

    def _build_item_init_description(self, row: Any, columns: Any) -> str:
        title = str(row.get("title", "")) if "title" in columns else ""
        if not title or title == "[PAD]":
            return "[PAD]"
        fields = []
        for col in ("main_category", "subtitle", "categories", "price"):
            if col in columns:
                value = str(row.get(col, "")).strip()
                if value and value.lower() != "nan":
                    fields.append(f"'{col}': '{value}'")
        meta = ", ".join(fields)
        return f"'item_title': '{title}'" + (f", {meta}" if meta else "")

    def _seed_user_domains(self, prepared_data: BaseTaskDataset) -> None:
        """Give each user an initial private/cross memory for every domain they interacted with."""
        interactions = prepared_data.get_interactions()
        if interactions.empty:
            return
        from recbole3.dataset.utils import USER_ID

        for _, row in interactions.iterrows():
            user_id = int(row[USER_ID])
            item_id = int(row[ITEM_ID])
            domain = self._item_domains.get(item_id, "")
            if not domain or user_id not in self._user_agents:
                continue
            init_text = f"I am an Amazon buyer, and I enjoy {domain} very much."
            self._user_agents[user_id].init_domain(domain, init_text)

    # ==================== Collators / base hooks ====================

    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return AgentCFPPTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return AgentCFPPEvalCollator(self.config, prepared_data)

    def forward(self, batch: Any) -> dict[str, Any]:
        return {}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        raise RuntimeError("AgentCF++ does not use gradient-based training.")

    # ==================== Training ====================

    def _domain_of(self, item_id: int) -> str:
        return self._item_domains.get(int(item_id), "")

    def _sample_negative(self, user_id: int, pos_item_id: int, domain: str) -> int:
        """Sample a negative item from the same domain's per-user candidate pool."""
        pool = self._domain_candidate_pools.get(domain, {}).get(int(user_id), [])
        candidates = [i for i in pool if int(i) != int(pos_item_id)]
        if candidates:
            return int(random.choice(candidates))
        # Fallback: any other item in the same domain, else any other item.
        same_domain = [i for i, d in self._item_domains.items() if d == domain and i != pos_item_id]
        if same_domain:
            return int(random.choice(same_domain))
        others = [i for i in range(self._num_items) if i != pos_item_id]
        return int(random.choice(others)) if others else int(pos_item_id)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """One AgentCF++ training step over a batch of positive interactions."""
        user_ids = batch["user_ids"].tolist()
        pos_item_ids = batch["pos_item_ids"].tolist()
        batch_size = len(user_ids)
        if batch_size == 0:
            return {"accuracy": 0.0}

        client = self._get_llm_client()

        # Resolve domains and sample domain-aware negatives.
        domains = [self._domain_of(p) for p in pos_item_ids]
        neg_item_ids = [
            self._sample_negative(user_ids[i], pos_item_ids[i], domains[i]) for i in range(batch_size)
        ]

        # --- Stage 1: forward pairwise choice (batched) ---
        forward_messages = []
        for i in range(batch_size):
            user_desc = self._user_agents[user_ids[i]].cross_description(domains[i])
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description
            item_desc_str = f"title:{neg_title}. description:{neg_desc}\ntitle:{pos_title}. description:{pos_desc}"
            prompt = FORWARD_PROMPT.substitute(user_description=user_desc, list_of_item_description=item_desc_str)
            forward_messages.append([{"role": "user", "content": prompt}])

        forward_responses = client.chat_completion_batch(forward_messages)
        selections, reasons, accuracy = [], [], []
        for i, resp in enumerate(forward_responses):
            choice, explanation = parse_recommendation(resp)
            selections.append(choice)
            reasons.append(explanation)
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            _, matched_idx = fuzzy_match_title(choice, [pos_title, neg_title])
            accuracy.append(1 if matched_idx == 0 else 0)
        acc_rate = sum(accuracy) / max(batch_size, 1)

        # --- Stage 2: update private user memory (batched) ---
        user_messages = []
        for i in range(batch_size):
            user_desc = self._user_agents[user_ids[i]].private_description(domains[i])
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description
            item_desc_str = f"1. {pos_title}: {pos_desc}\n\n2. {neg_title}: {neg_desc}"
            template = USER_PROMPT_TRUE if accuracy[i] == 1 else USER_PROMPT
            system_msg = USER_SYSTEM_ROLE.substitute(user_description=user_desc)
            user_msg = template.substitute(
                list_of_item_description=item_desc_str,
                pos_item_title=pos_title,
                neg_item_title=neg_title,
                system_reason=reasons[i],
            )
            user_messages.append(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
            )
        user_responses = client.chat_completion_batch(user_messages)
        for i in range(batch_size):
            new_private = parse_user_update(user_responses[i])
            self._user_agents[user_ids[i]].update_private(domains[i], new_private)

        # --- Stage 3: cross-domain deduction (batched) ---
        cross_messages = []
        for i in range(batch_size):
            agent = self._user_agents[user_ids[i]]
            cross_msg = CROSSDOMAIN_PROMPT.substitute(
                cross_domain_preference=agent.cross_description(domains[i]),
                private_domain_description=agent.all_private_description(),
                main_kind=domains[i],
            )
            cross_messages.append([{"role": "user", "content": cross_msg}])
        cross_responses = client.chat_completion_batch(cross_messages)
        for i in range(batch_size):
            new_cross = parse_crossdomain_update(cross_responses[i])
            self._user_agents[user_ids[i]].update_cross(domains[i], new_cross)
            self._user_agents[user_ids[i]].commit_memory(domains[i])

        # --- Stage 4: update item memory (batched) ---
        item_messages = []
        for i in range(batch_size):
            user_desc = self._user_agents[user_ids[i]].cross_description(domains[i])
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description
            # Ordered (first = neg, second = pos) to match the parser.
            item_desc_str = f"title:{neg_title}. description:{neg_desc}\ntitle:{pos_title}. description:{pos_desc}"
            template = ITEM_PROMPT_TRUE if accuracy[i] == 1 else ITEM_PROMPT
            item_msg = template.substitute(
                user_description=user_desc,
                list_of_item_description=item_desc_str,
                pos_item_title=pos_title,
                neg_item_title=neg_title,
                system_reason=reasons[i],
            )
            item_messages.append([{"role": "user", "content": item_msg}])
        item_responses = client.chat_completion_batch(item_messages)
        for i in range(batch_size):
            neg_desc_new, pos_desc_new = parse_item_update(item_responses[i])
            if pos_desc_new:
                self._item_agents[pos_item_ids[i]].update(pos_desc_new)
            if neg_desc_new and self.config.update_neg_item:
                self._item_agents[neg_item_ids[i]].update(neg_desc_new)

        return {"accuracy": acc_rate}

    # ==================== Evaluation ====================

    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_item_ids is None:
            raise NotImplementedError("AgentCF++ requires candidate_item_ids for prediction.")

        user_ids = model_inputs["user_ids"]
        batch_size = user_ids.shape[0]
        candidates_np = candidate_item_ids.detach().cpu().numpy()
        user_ids_np = user_ids.detach().cpu().numpy()
        history_item_ids = model_inputs.get("history_item_ids")

        result = torch.zeros((batch_size, k), dtype=torch.long)

        chat_batch = self.config.chat_api_batch
        for start in range(0, batch_size, chat_batch):
            end = min(start + chat_batch, batch_size)
            sub_history = history_item_ids[start:end] if history_item_ids else None
            ranked = self._evaluate_batch(
                user_ids_np[start:end], candidates_np[start:end], sub_history, k=k
            )
            result[start:end] = torch.tensor(ranked, dtype=torch.long)
        return result

    def _build_eval_prompt(
        self,
        user_id: int,
        domain: str,
        candidate_num: int,
        item_desc_str: str,
        history_item_ids: tuple[int, ...] | None,
    ) -> str:
        agent = self._user_agents[user_id]
        if self.config.use_intermediate_node:
            user_desc = (
                f"===My preferences in {domain}:===\n{agent.private_description(domain)}\n"
                f"===Moreover:===\n{agent.cross_description(domain)}"
            )
        else:
            user_desc = agent.cross_description(domain)

        group_mem = ""
        if self.config.use_group_memory:
            group_mem = self._group_state.render_group_mem(user_id, domain)

        strategy = self.config.prompt_strategy
        if strategy == "B+H":
            max_his = self.config.history_max_length or 50
            his = (history_item_ids or ())[-max_his:]
            his_text = "\n".join(
                f"{idx + 1}. {self._item_text_lookup[int(h)]}"
                for idx, h in enumerate(his)
                if int(h) < len(self._item_text_lookup)
            )
            return EVAL_SEQUENTIAL.substitute(
                user_description=user_desc,
                historical_interactions=his_text,
                group_mem=group_mem,
                candidate_num=candidate_num,
                example_list_of_item_description=item_desc_str,
            )
        if strategy == "B+R":
            past_desc = agent.long_memory[-2] if len(agent.long_memory) > 1 else agent.cross_description(domain)
            return EVAL_RETRIEVAL.substitute(
                user_past_description=past_desc,
                user_description=user_desc,
                group_mem=group_mem,
                candidate_num=candidate_num,
                example_list_of_item_description=item_desc_str,
            )
        return EVAL_BASIC.substitute(
            user_description=user_desc,
            group_mem=group_mem,
            candidate_num=candidate_num,
            example_list_of_item_description=item_desc_str,
        )

    def _evaluate_batch(
        self,
        user_ids: np.ndarray,
        candidate_ids: np.ndarray,
        history_item_ids: list[tuple[int, ...]] | None,
        *,
        k: int,
    ) -> list[list[int]]:
        client = self._get_llm_client()
        batch_size = len(user_ids)

        messages_list = []
        candidate_texts_batch = []
        for i in range(batch_size):
            user_id = int(user_ids[i])
            candidates = candidate_ids[i]
            domain = self._domain_of(int(candidates[0])) if len(candidates) else ""

            candidate_text_list = []
            candidate_desc_list = []
            for j, cid in enumerate(candidates):
                cid = int(cid)
                title = self._item_text_lookup[cid] if cid < len(self._item_text_lookup) else "[UNK]"
                desc = self._item_agents[cid].current_description if cid in self._item_agents else title
                candidate_text_list.append(title)
                candidate_desc_list.append(f"title:{title}. description:{desc}")
            candidate_texts_batch.append(candidate_text_list)
            item_desc_str = "\n".join(candidate_desc_list)

            his = history_item_ids[i] if history_item_ids else None
            prompt_text = self._build_eval_prompt(user_id, domain, len(candidates), item_desc_str, his)
            messages_list.append([{"role": "user", "content": prompt_text}])

        responses = client.chat_completion_batch(messages_list, temperature=self.config.temperature_eval)

        results = []
        for i in range(batch_size):
            ranking_titles = parse_evaluation_ranking(responses[i])
            candidate_text = candidate_texts_batch[i]
            candidates = candidate_ids[i]

            scores = np.full(len(candidates), -10000.0)
            matched_indices: set[int] = set()
            for rank, title in enumerate(ranking_titles):
                if self.config.match_rule == "exact":
                    for idx, ct in enumerate(candidate_text):
                        if ct in title and idx not in matched_indices:
                            scores[idx] = len(candidates) - rank
                            matched_indices.add(idx)
                            break
                else:
                    _, matched_idx = fuzzy_match_title(title, candidate_text)
                    if matched_idx >= 0 and matched_idx not in matched_indices:
                        scores[matched_idx] = len(candidates) - rank
                        matched_indices.add(matched_idx)

            top_k_indices = np.argsort(-scores)[:k]
            results.append([int(candidates[idx]) for idx in top_k_indices])
        return results

    # ==================== State persistence ====================

    def save_agent_states(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "user", "w", encoding="utf-8") as f:
            f.write("user_id\tdomain\tprivate\tcross\n")
            for uid, agent in self._user_agents.items():
                for domain in sorted(agent.active_domains):
                    private = agent.private.get(domain, "").replace("\n", " ").replace("\t", " ")
                    cross = agent.cross.get(domain, "").replace("\n", " ").replace("\t", " ")
                    f.write(f"{uid}\t{domain}\t{private}\t{cross}\n")
        with open(path / "item", "w", encoding="utf-8") as f:
            f.write("item_id\tdomain\tdescription\n")
            for iid, agent in self._item_agents.items():
                desc = agent.current_description.replace("\n", " ").replace("\t", " ")
                f.write(f"{iid}\t{agent.domain}\t{desc}\n")

    def _load_agent_states(self, path: Path) -> None:
        user_file = path / "user"
        if user_file.exists():
            with open(user_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 4:
                        uid, domain, private, cross = parts
                        uid = int(uid)
                        if uid in self._user_agents:
                            self._user_agents[uid].update_private(domain, private)
                            self._user_agents[uid].update_cross(domain, cross)
        item_file = path / "item"
        if item_file.exists():
            with open(item_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        iid, _domain, desc = parts
                        iid = int(iid)
                        if iid in self._item_agents:
                            self._item_agents[iid].update(desc)


__all__ = ["AgentCFPPModel"]
