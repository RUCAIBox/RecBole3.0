from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from recbole3.dataset.base import BaseTaskDataset
from recbole3.dataset.utils import CANDIDATE_ITEM_IDS, ITEM_ID, USER_ID
from recbole3.model.agentcf.agents import ItemAgentState, RecAgentState, UserAgentState
from recbole3.model.agentcf.config import AgentCFConfig
from recbole3.model.agentcf.data import AgentCFEvalCollator, AgentCFTrainCollator
from recbole3.model.agentcf.llm_client import LLMClient
from recbole3.model.agentcf.output_parser import (
    fuzzy_match_title,
    parse_evaluation_ranking,
    parse_item_update,
    parse_recommendation,
    parse_user_update,
)
from recbole3.model.agentcf.prompts import (
    BACKWARD_ITEM_PROMPT,
    BACKWARD_ITEM_PROMPT_TRUE,
    BACKWARD_USER_PROMPT,
    BACKWARD_USER_PROMPT_TRUE,
    BACKWARD_USER_SYSTEM_ROLE,
    EVAL_BASIC_PROMPT,
    EVAL_RAG_PROMPT,
    EVAL_SEQUENTIAL_PROMPT,
    FORWARD_PROMPT,
)
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.sequential import HISTORY_ITEM_IDS


class AgentCFModel(BaseRetrievalModel):
    """AgentCF: LLM-based collaborative filtering with multi-agent interactions."""

    def __init__(self, config: AgentCFConfig):
        super().__init__(config)
        self.config: AgentCFConfig = config
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._user_agents: dict[int, UserAgentState] = {}
        self._item_agents: dict[int, ItemAgentState] = {}
        self._rec_agent: RecAgentState = RecAgentState()
        self._item_text_lookup: tuple[str, ...] = ()
        self._num_items: int = 0
        self._num_users: int = 0
        self._llm_client: LLMClient | None = None
        self._initialized: bool = False

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

    def ensure_initialized(self, prepared_data: BaseTaskDataset) -> None:
        if self._initialized:
            return
        self._num_items = prepared_data.get_num_items()
        self._num_users = prepared_data.get_num_users()

        item_table = prepared_data.get_item_table()
        item_texts = ["[PAD]"] * self._num_items
        item_categories: dict[int, str] = {}

        if "title" in item_table.columns:
            for _, row in item_table.iterrows():
                item_id = int(row[ITEM_ID])
                if 0 <= item_id < self._num_items:
                    title = str(row.get("title", ""))
                    item_texts[item_id] = title
                    item_categories[item_id] = str(row.get("category", "CDs"))

        self._item_text_lookup = tuple(item_texts)

        # Initialize user agents
        for user_id in range(self._num_users):
            init_desc = "I enjoy listening to CDs very much."
            self._user_agents[user_id] = UserAgentState(
                user_id=user_id,
                description=init_desc,
                memory_history=[init_desc],
                update_history=[init_desc],
            )

        # Initialize item agents
        for item_id in range(self._num_items):
            title = self._item_text_lookup[item_id]
            category = item_categories.get(item_id, "CDs")
            if title and title != "[PAD]":
                init_desc = f"The CD is called '{title}'. The category of this CD is: '{category}'."
            else:
                init_desc = "[PAD]"
            self._item_agents[item_id] = ItemAgentState(
                item_id=item_id,
                title=title,
                category=category,
                initial_description=init_desc,
                update_history=[init_desc],
                description_embeddings={init_desc: None},
            )

        # Load pre-trained agent states if configured
        if self.config.load_agent_state_path:
            self._load_agent_states(Path(self.config.load_agent_state_path))

        self._initialized = True

    def build_train_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return AgentCFTrainCollator(self.config, prepared_data, num_items=self._num_items)

    def build_eval_collator(self, prepared_data: BaseTaskDataset) -> BaseCollator:
        self.ensure_initialized(prepared_data)
        return AgentCFEvalCollator(self.config, prepared_data)

    def forward(self, batch: Any) -> dict[str, Any]:
        return {}

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        raise RuntimeError("AgentCF does not use gradient-based training.")

    def predict(
        self,
        model_inputs: Any,
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LLM-based ranking of candidate items."""
        if candidate_item_ids is None:
            raise NotImplementedError("AgentCF requires candidate_item_ids for prediction.")

        user_ids = model_inputs["user_ids"]
        batch_size = user_ids.shape[0]
        candidates_np = candidate_item_ids.detach().cpu().numpy()
        user_ids_np = user_ids.detach().cpu().numpy()

        history_item_ids = model_inputs.get("history_item_ids")

        result = torch.zeros((batch_size, k), dtype=torch.long)

        # Process in sub-batches for API efficiency
        chat_batch = self.config.chat_api_batch
        for start in range(0, batch_size, chat_batch):
            end = min(start + chat_batch, batch_size)
            sub_user_ids = user_ids_np[start:end]
            sub_candidates = candidates_np[start:end]
            sub_history = history_item_ids[start:end] if history_item_ids else None

            ranked_ids = self._evaluate_batch(
                sub_user_ids, sub_candidates, sub_history, k=k
            )
            result[start:end] = torch.tensor(ranked_ids, dtype=torch.long)

        return result

    def _evaluate_batch(
        self,
        user_ids: np.ndarray,
        candidate_ids: np.ndarray,
        history_item_ids: list[tuple[int, ...]] | None,
        *,
        k: int,
    ) -> list[list[int]]:
        """Evaluate a batch of users using LLM ranking."""
        client = self._get_llm_client()
        batch_size = len(user_ids)

        # Build evaluation prompts
        messages_list = []
        candidate_texts_batch = []

        for i in range(batch_size):
            user_id = int(user_ids[i])
            user_desc = self._user_agents[user_id].current_description
            candidates = candidate_ids[i]

            # Build candidate descriptions
            candidate_text_list = []
            candidate_desc_list = []
            for j, cid in enumerate(candidates):
                cid = int(cid)
                title = self._item_text_lookup[cid] if cid < len(self._item_text_lookup) else "[UNK]"
                desc = self._item_agents[cid].current_description if cid in self._item_agents else title
                candidate_text_list.append(title)
                candidate_desc_list.append(f"{j + 1}. {title}: {desc}")

            candidate_texts_batch.append(candidate_text_list)
            item_desc_str = "\n\n".join(candidate_desc_list)

            # Select prompt based on evaluation mode
            if self.config.evaluation_mode == "sequential" and history_item_ids:
                his = history_item_ids[i] if history_item_ids else ()
                max_his = self.config.history_max_length or 50
                real_his = his[-max_his:] if his else ()
                his_text = "\n".join(
                    f"{idx + 1}. {self._item_text_lookup[int(h)]}"
                    for idx, h in enumerate(real_his)
                    if int(h) < len(self._item_text_lookup)
                )
                prompt_text = EVAL_SEQUENTIAL_PROMPT.substitute(
                    user_description=user_desc,
                    historical_interactions=his_text,
                    candidate_num=len(candidates),
                    example_list_of_item_description=item_desc_str,
                )
            elif self.config.evaluation_mode == "rag":
                past_desc = self._user_agents[user_id].memory_history[-2] if len(self._user_agents[user_id].memory_history) > 1 else user_desc
                prompt_text = EVAL_RAG_PROMPT.substitute(
                    user_past_description=past_desc,
                    user_description=user_desc,
                    candidate_num=len(candidates),
                    example_list_of_item_description=item_desc_str,
                )
            else:
                prompt_text = EVAL_BASIC_PROMPT.substitute(
                    user_description=user_desc,
                    candidate_num=len(candidates),
                    example_list_of_item_description=item_desc_str,
                )

            messages_list.append([{"role": "user", "content": prompt_text}])

        # Call LLM
        responses = client.chat_completion_batch(messages_list, temperature=self.config.temperature_eval)

        # Parse responses into ranked item IDs
        results = []
        for i in range(batch_size):
            ranking_titles = parse_evaluation_ranking(responses[i])
            candidate_text = candidate_texts_batch[i]
            candidates = candidate_ids[i]

            scores = np.full(len(candidates), -10000.0)
            matched_indices = set()

            for rank, title in enumerate(ranking_titles):
                if self.config.match_rule == "exact":
                    for idx, ct in enumerate(candidate_text):
                        if ct in title and idx not in matched_indices:
                            scores[idx] = self.config.recall_budget - rank
                            matched_indices.add(idx)
                            break
                else:
                    _, matched_idx = fuzzy_match_title(title, candidate_text)
                    if matched_idx >= 0 and matched_idx not in matched_indices:
                        scores[matched_idx] = self.config.recall_budget - rank
                        matched_indices.add(matched_idx)

            top_k_indices = np.argsort(-scores)[:k]
            top_k_item_ids = [int(candidates[idx]) for idx in top_k_indices]
            results.append(top_k_item_ids)

        return results

    # ==================== Training Methods ====================

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute one AgentCF training step: forward → accuracy check → backward."""
        user_ids = batch["user_ids"].tolist()
        pos_item_ids = batch["pos_item_ids"].tolist()
        neg_item_ids = batch["neg_item_ids"].tolist()
        batch_size = len(user_ids)

        client = self._get_llm_client()

        for round_idx in range(self.config.all_update_rounds):
            print(f"{'~' * 20} {round_idx}-th round update! {'~' * 20}")

            # Forward: RecAgent makes pairwise choice
            selections, reasons = self._agent_forward(client, user_ids, pos_item_ids, neg_item_ids)

            # Check accuracy
            accuracy = self._compute_accuracy(selections, pos_item_ids, neg_item_ids)
            acc_rate = sum(accuracy) / max(len(accuracy), 1)
            print(f"Current accuracy is {acc_rate:.4f}")

            # Split into correct and incorrect predictions
            backward_users, backward_pos, backward_neg, backward_reasons = [], [], [], []
            backward_true_users, backward_true_pos, backward_true_neg, backward_true_reasons = [], [], [], []
            first_time_correct = set()

            for j, acc in enumerate(accuracy):
                if acc == 0:
                    backward_users.append(user_ids[j])
                    backward_pos.append(pos_item_ids[j])
                    backward_neg.append(neg_item_ids[j])
                    backward_reasons.append(reasons[j])
                else:
                    if round_idx == 0:
                        first_time_correct.add(user_ids[j])
                        backward_true_users.append(user_ids[j])
                        backward_true_pos.append(pos_item_ids[j])
                        backward_true_neg.append(neg_item_ids[j])
                        backward_true_reasons.append(reasons[j])
                    elif user_ids[j] not in first_time_correct:
                        backward_true_users.append(user_ids[j])
                        backward_true_pos.append(pos_item_ids[j])
                        backward_true_neg.append(neg_item_ids[j])
                        backward_true_reasons.append(reasons[j])

            # Backward for incorrect predictions
            if backward_users:
                self._agent_backward(
                    client, backward_reasons, backward_users, backward_pos, backward_neg
                )

            # Backward_true for correct predictions (first round updates user too)
            if backward_true_users and round_idx == 0:
                self._agent_backward_true(
                    client, backward_true_reasons, backward_true_users,
                    backward_true_pos, backward_true_neg, update_user=True
                )

        # Final backward_true without user update
        if backward_true_users:
            self._agent_backward_true(
                client, backward_true_reasons, backward_true_users,
                backward_true_pos, backward_true_neg, update_user=False
            )

        # Store examples for RAG and commit memory
        for i in range(batch_size):
            uid = user_ids[i]
            self._user_agents[uid].commit_memory()
            self._rec_agent.add_example(
                user_id=uid,
                user_desc=self._user_agents[uid].current_description,
                pos_title=self._item_text_lookup[pos_item_ids[i]],
                neg_title=self._item_text_lookup[neg_item_ids[i]],
                pos_desc=self._item_agents[pos_item_ids[i]].current_description,
                neg_desc=self._item_agents[neg_item_ids[i]].current_description,
                accuracy=accuracy[i],
                reason=reasons[i],
            )

        return {"accuracy": acc_rate}

    def _agent_forward(
        self,
        client: LLMClient,
        user_ids: list[int],
        pos_item_ids: list[int],
        neg_item_ids: list[int],
    ) -> tuple[list[str], list[str]]:
        """RecAgent makes pairwise choice between pos and neg items."""
        batch_size = len(user_ids)
        messages_list = []

        for i in range(batch_size):
            user_desc = self._user_agents[user_ids[i]].current_description
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]

            item_desc_str = (
                f"1. {pos_title}: {pos_desc}\n\n"
                f"2. {neg_title}: {neg_desc}"
            )

            prompt = FORWARD_PROMPT.substitute(
                user_description=user_desc,
                list_of_item_description=item_desc_str,
            )
            messages_list.append([{"role": "user", "content": prompt}])

        # Batch API calls
        responses = client.chat_completion_batch(messages_list)

        selections, reasons = [], []
        for resp in responses:
            choice, explanation = parse_recommendation(resp)
            selections.append(choice)
            reasons.append(explanation)

        return selections, reasons

    def _compute_accuracy(
        self,
        selections: list[str],
        pos_item_ids: list[int],
        neg_item_ids: list[int],
    ) -> list[int]:
        """Check if the LLM selected the positive item."""
        accuracy = []
        for i, selection in enumerate(selections):
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            _, matched_idx = fuzzy_match_title(selection, [pos_title, neg_title])
            accuracy.append(1 if matched_idx == 0 else 0)
        return accuracy

    def _agent_backward(
        self,
        client: LLMClient,
        reasons: list[str],
        user_ids: list[int],
        pos_item_ids: list[int],
        neg_item_ids: list[int],
    ) -> None:
        """Update user and item descriptions when recommendation was wrong."""
        batch_size = len(user_ids)

        # Update user descriptions
        user_messages = []
        for i in range(batch_size):
            user_desc = self._user_agents[user_ids[i]].current_description
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description

            item_desc_str = (
                f"1. {pos_title}: {pos_desc}\n\n"
                f"2. {neg_title}: {neg_desc}"
            )

            system_msg = BACKWARD_USER_SYSTEM_ROLE.substitute(user_description=user_desc)
            user_msg = BACKWARD_USER_PROMPT.substitute(
                list_of_item_description=item_desc_str,
                neg_item_title=neg_title,
                pos_item_title=pos_title,
                system_reason=reasons[i],
            )
            user_messages.append([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

        user_responses = client.chat_completion_batch(user_messages)
        user_updates = [parse_user_update(resp) for resp in user_responses]

        for i in range(batch_size):
            self._user_agents[user_ids[i]].update(user_updates[i])

        print("*" * 10 + "User Update Is Over!" + "*" * 10)

        # Update item descriptions
        item_messages = []
        for i in range(batch_size):
            user_desc = user_updates[i]
            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description

            item_desc_str = (
                f"1. {neg_title}: {neg_desc}\n\n"
                f"2. {pos_title}: {pos_desc}"
            )

            item_msg = BACKWARD_ITEM_PROMPT.substitute(
                user_description=user_desc,
                list_of_item_description=item_desc_str,
                neg_item_title=neg_title,
                pos_item_title=pos_title,
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

        print("*" * 10 + "Item Update Is Over!" + "*" * 10)

    def _agent_backward_true(
        self,
        client: LLMClient,
        reasons: list[str],
        user_ids: list[int],
        pos_item_ids: list[int],
        neg_item_ids: list[int],
        *,
        update_user: bool = True,
    ) -> None:
        """Update user and item descriptions when recommendation was correct."""
        batch_size = len(user_ids)

        if update_user:
            # Update user descriptions
            user_messages = []
            for i in range(batch_size):
                user_desc = self._user_agents[user_ids[i]].current_description
                pos_title = self._item_text_lookup[pos_item_ids[i]]
                neg_title = self._item_text_lookup[neg_item_ids[i]]
                pos_desc = self._item_agents[pos_item_ids[i]].current_description
                neg_desc = self._item_agents[neg_item_ids[i]].current_description

                item_desc_str = (
                    f"1. {pos_title}: {pos_desc}\n\n"
                    f"2. {neg_title}: {neg_desc}"
                )

                system_msg = BACKWARD_USER_SYSTEM_ROLE.substitute(user_description=user_desc)
                user_msg = BACKWARD_USER_PROMPT_TRUE.substitute(
                    list_of_item_description=item_desc_str,
                    pos_item_title=pos_title,
                    neg_item_title=neg_title,
                    system_reason=reasons[i],
                )
                user_messages.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])

            user_responses = client.chat_completion_batch(user_messages)
            user_updates = [parse_user_update(resp) for resp in user_responses]

            for i in range(batch_size):
                self._user_agents[user_ids[i]].update(user_updates[i])

        # Update item descriptions
        item_messages = []
        for i in range(batch_size):
            if update_user:
                user_desc = user_updates[i]
            else:
                user_desc = self._user_agents[user_ids[i]].current_description

            pos_title = self._item_text_lookup[pos_item_ids[i]]
            neg_title = self._item_text_lookup[neg_item_ids[i]]
            pos_desc = self._item_agents[pos_item_ids[i]].current_description
            neg_desc = self._item_agents[neg_item_ids[i]].current_description

            item_desc_str = (
                f"1. {neg_title}: {neg_desc}\n\n"
                f"2. {pos_title}: {pos_desc}"
            )

            item_msg = BACKWARD_ITEM_PROMPT_TRUE.substitute(
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

    # ==================== State Persistence ====================

    def save_agent_states(self, path: Path) -> None:
        """Save agent states to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save user descriptions
        with open(path / "user", "w", encoding="utf-8") as f:
            f.write("user_id:token\tuser_description:token_seq\n")
            for uid, agent in self._user_agents.items():
                desc = agent.current_description.replace("\n", " ")
                f.write(f"{uid}\t{desc}\n")

        # Save item descriptions
        with open(path / "item", "w", encoding="utf-8") as f:
            f.write("item_id:token\titem_description:token_seq\n")
            for iid, agent in self._item_agents.items():
                desc = agent.current_description.replace("\n", " ")
                f.write(f"{iid}\t{desc}\n")

    def _load_agent_states(self, path: Path) -> None:
        """Load pre-trained agent states from disk."""
        user_file = path / "user"
        if user_file.exists():
            with open(user_file, "r", encoding="utf-8") as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        uid = int(parts[0])
                        desc = parts[1]
                        if uid in self._user_agents:
                            self._user_agents[uid].update(desc)
                            self._user_agents[uid].commit_memory()

        item_file = path / "item"
        if item_file.exists():
            with open(item_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        iid = int(parts[0])
                        desc = parts[1]
                        if iid in self._item_agents:
                            self._item_agents[iid].update(desc)
