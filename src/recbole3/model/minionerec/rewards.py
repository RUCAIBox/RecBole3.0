from __future__ import annotations

import math
from collections.abc import Callable

from recbole3.model.minionerec.config import MiniOneRecConfig


RewardFunc = Callable[[list[str], list[str]], list[float]]


def build_minionerec_reward_functions(
    config: MiniOneRecConfig,
    *,
    prompt2history: dict[str, str],
    history2target: dict[str, str],
) -> tuple[RewardFunc, ...]:
    """Build the original MiniOneRec rule/ranking reward functions."""

    ndcg_rewards = [-1.0 / math.log2(index + 2) for index in range(int(config.rl_num_generations))]
    normalizer = sum(ndcg_rewards)
    ndcg_rewards = [-value / normalizer for value in ndcg_rewards]

    def rule_reward(prompts: list[str], completions: list[str]) -> list[float]:
        histories = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[history] for history in histories]
        return [
            1.0 if normalize_minionerec_rule_text(completion) == normalize_minionerec_rule_text(target) else 0.0
            for completion, target in zip(completions, targets, strict=True)
        ]

    def ndcg_rule_reward(prompts: list[str], completions: list[str]) -> list[float]:
        histories = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[history] for history in histories]
        rewards: list[float] = []
        group_rewards: list[float] = []
        group_has_hit = False
        for index, (completion, target) in enumerate(zip(completions, targets, strict=True)):
            if normalize_minionerec_ranking_text(completion) == normalize_minionerec_ranking_text(target):
                group_has_hit = True
                group_rewards.append(0.0)
            else:
                group_rewards.append(ndcg_rewards[index % int(config.rl_num_generations)])
            if (index + 1) % int(config.rl_num_generations) == 0:
                rewards.extend(group_rewards if group_has_hit else [0.0] * int(config.rl_num_generations))
                group_rewards = []
                group_has_hit = False
        return rewards

    reward_type = str(config.rl_reward_type)
    if reward_type == "rule":
        return (rule_reward,)
    if reward_type == "ranking":
        return (rule_reward, ndcg_rule_reward)
    if reward_type == "ranking_only":
        return (ndcg_rule_reward,)
    raise ValueError(f"Unsupported MiniOneRec rl_reward_type '{config.rl_reward_type}'.")


def normalize_minionerec_rule_text(text: str) -> str:
    """Original rule_reward normalization: strip newline, quote, and space."""

    return str(text).strip('\n" ')


def normalize_minionerec_ranking_text(text: str) -> str:
    """Original ndcg_rule_reward normalization: strip newline and quote only."""

    return str(text).strip('\n"')


__all__ = [
    "RewardFunc",
    "build_minionerec_reward_functions",
    "normalize_minionerec_ranking_text",
    "normalize_minionerec_rule_text",
]
