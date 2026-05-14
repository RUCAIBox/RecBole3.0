from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from recbole3.dataset import ITEM_ID
from recbole3.model.sequential import HISTORY_ITEM_IDS
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig


def _interleave_and_pad(tensors_list: list[torch.Tensor]) -> torch.Tensor:
    """Pad tensors [T_g, B, N] to same T and reshape to [B*G, T_max, N]."""
    padded = pad_sequence(tensors_list).permute(2, 1, 0, 3)  # [B, G, T_max, N]
    T, N = padded.shape[2], padded.shape[3]
    return padded.reshape(-1, T, N)


def _compute_advantages(
    last_step_logps: torch.Tensor,
    pos_items: torch.Tensor,
    k: int,
    group_num: int,
    reward_metric: str,
) -> tuple[torch.Tensor, float]:
    """Compute group-relative advantages and total reward.

    Args:
        last_step_logps: [B*G, N] log-probabilities at last recurrence step.
        pos_items: [B] target item IDs (with offset), repeated to [B*G].
        k: Top-k for reward.
        group_num: G, number of rollouts per sample.
        reward_metric: "recall" or "ndcg".

    Returns:
        advantages: [B*G] group-normalized advantages.
        total_reward: float, sum of rewards.
    """
    BG = last_step_logps.shape[0]
    k = min(k, last_step_logps.shape[-1])
    topk_idx = torch.topk(last_step_logps, k, dim=-1).indices

    if reward_metric.lower() == "recall":
        rewards = torch.any(topk_idx == pos_items.view(BG, 1), dim=1).float()
    elif reward_metric.lower() == "ndcg":
        mask = topk_idx == pos_items.unsqueeze(1)
        ranks = mask.int().argmax(dim=1)
        in_top_k = mask.sum(dim=1) > 0
        rewards = torch.where(
            in_top_k,
            1.0 / torch.log2(ranks.float() + 2.0),
            torch.tensor(0.0, device=last_step_logps.device),
        )
    else:
        raise ValueError(f"Unsupported reward_metric: {reward_metric}")

    mean_r = rewards.view(-1, group_num).mean(dim=1).repeat_interleave(group_num, dim=0)
    std_r = rewards.view(-1, group_num).std(dim=1).repeat_interleave(group_num, dim=0)
    advantages = (rewards - mean_r) / (std_r + 1e-4)
    return advantages, float(rewards.sum().item())


class LARESTrainer(Trainer):
    """Trainer with SL (supervised) and RL (GRPO) stages for LARES."""

    config_cls = TrainerConfig

    def run(
        self,
        model: Any,
        prepared_data: Any,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        self._setup_logger(model, prepared_data, output_dir)
        total_start = time.perf_counter()
        try:
            stage = str(getattr(model.config, "stage", "SL") or "SL").upper()
            if stage == "SL":
                result = self._run_sl(model, prepared_data, output_dir)
            elif stage == "RL":
                result = self._run_rl(model, prepared_data, output_dir)
            else:
                raise ValueError(f"Unknown training stage: {stage}")

            if (logger := getattr(self, "_logger", None)) is not None:
                total_elapsed = time.perf_counter() - total_start
                logger.log_summary(
                    stopped_early=False,
                    total_epochs=0,
                    best_epoch=None,
                    total_time=total_elapsed,
                )
            return result
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    # ── SL stage ───────────────────────────────────────────────────────

    def _run_sl(
        self,
        model: Any,
        prepared_data: Any,
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        fit_result = self.fit(model, prepared_data, output_dir=output_dir)

        if (logger := getattr(self, "_logger", None)) is not None:
            best_metric = fit_result.get("best_metric")
            if best_metric:
                logger.log_best(
                    epoch=fit_result["best_epoch"],
                    monitor_name=best_metric["name"],
                    best_value=best_metric["value"],
                )

        best_checkpoint = fit_result["checkpoint_paths"].get("best")
        if best_checkpoint:
            state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        print("[trainer] starting test evaluation")
        scaling_results = self._eval_recurrence_scaling(model, prepared_data)
        print("[trainer] finished test evaluation")

        self._log_scaling_results(scaling_results)

        mean_T = model.config.mean_recurrence
        return {
            "fit": fit_result,
            "test": scaling_results.get(f"{int(mean_T)}", {}),
            "test_scaling": scaling_results,
        }

    # ── RL stage ───────────────────────────────────────────────────────

    def _run_rl(
        self,
        model: Any,
        prepared_data: Any,
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        # Load pretrained SL weights
        pretrained_path = model.config.pretrain_model_path
        if not pretrained_path:
            raise ValueError("config.pretrain_model_path is required for RL stage.")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[trainer] loaded pretrained weights from {pretrained_path}")

        # Reference model (frozen)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        group_num = int(model.config.group_num)
        k = int(model.config.k)
        beta = float(model.config.beta)
        reward_metric = str(model.config.reward_metric)

        accelerator = self.create_accelerator()
        collator = model.build_train_collator(prepared_data)
        train_dataset = prepared_data.get_train_dataset()
        train_dataloader = self.build_dataloader(train_dataset, collator, shuffle=self.config.shuffle)
        optimizer = self.build_optimizer(model)

        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

        for epoch in range(1, int(self.config.max_epochs) + 1):
            model.train()
            total_loss = 0.0
            total_reward = 0.0
            n_samples = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                item_ids = batch[HISTORY_ITEM_IDS]
                lengths = batch["history_lengths"]
                pos_items = batch[ITEM_ID]

                loss, reward = self._rl_compute_loss(
                    model, ref_model, item_ids, lengths, pos_items,
                    group_num=group_num, k=k, beta=beta, reward_metric=reward_metric,
                )

                accelerator.backward(loss)
                optimizer.step()

                total_loss += float(loss.detach().item())
                total_reward += reward
                n_samples += int(item_ids.shape[0]) * group_num

            avg_reward = total_reward / max(n_samples, 1)
            print(f"[train:RL] epoch={epoch}/{int(self.config.max_epochs)} "
                  f"avg_loss={total_loss / max(len(train_dataloader), 1):.6f} "
                  f"avg_reward={avg_reward:.4f}")

        print("[trainer] starting test evaluation")
        scaling_results = self._eval_recurrence_scaling(model, prepared_data)
        print("[trainer] finished test evaluation")

        self._log_scaling_results(scaling_results)

        mean_T = model.config.mean_recurrence
        return {
            "test": scaling_results.get(f"{int(mean_T)}", {}),
            "test_scaling": scaling_results,
        }

    # ── RL loss ────────────────────────────────────────────────────────

    def _rl_compute_loss(
        self,
        model: Any,
        ref_model: Any,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        pos_items: torch.Tensor,
        *,
        group_num: int,
        k: int,
        beta: float,
        reward_metric: str,
    ) -> tuple[torch.Tensor, float]:
        per_step_logps_list: list[torch.Tensor] = []
        ref_per_step_logps_list: list[torch.Tensor] = []

        for _ in range(group_num):
            _, _, step_logps = model._encode(item_ids, lengths, return_all_states=True)
            # step_logps: [B, T, N] -> permute to [T, B, N]
            per_step_logps_list.append(step_logps.permute(1, 0, 2))

            num_steps = step_logps.shape[1]
            with torch.no_grad():
                _, _, ref_step_logps = ref_model._encode(
                    item_ids, lengths, return_all_states=True, num_steps=num_steps,
                )
                ref_per_step_logps_list.append(ref_step_logps.permute(1, 0, 2))

        # Interleave groups: [B*G, T_max, N]
        per_step_logps = _interleave_and_pad(per_step_logps_list)
        ref_per_step_logps = _interleave_and_pad(ref_per_step_logps_list)

        BG, T, N = per_step_logps.shape
        action_mask = (per_step_logps[:, :, 0] != 0).long()
        action_len = action_mask.sum(dim=1)

        # Last-step logits for reward (gather last valid position)
        idx = (action_len - 1).view(-1, 1, 1).expand(-1, 1, N)
        last_step_logps = per_step_logps.gather(1, idx).squeeze(1)

        # Repeat pos_items to match group-interleaved layout [B*G]
        pos_items_rep = pos_items.repeat_interleave(group_num, dim=0)

        advantages, total_reward = _compute_advantages(
            last_step_logps, pos_items_rep, k, group_num, reward_metric,
        )

        # Per-step log-probability of the target item
        x_idx = torch.arange(BG, device=per_step_logps.device).repeat_interleave(T)
        y_idx = torch.arange(T, device=per_step_logps.device).repeat(BG)
        z_idx = pos_items_rep.repeat_interleave(T)

        per_step_logps_target = per_step_logps[x_idx, y_idx, z_idx].view(BG, T)
        ref_per_step_logps_target = ref_per_step_logps[x_idx, y_idx, z_idx].view(BG, T)

        # KL divergence: exp(ref - pi) - (ref - pi) - 1
        per_token_kl = (
            torch.exp(ref_per_step_logps_target - per_step_logps_target)
            - (ref_per_step_logps_target - per_step_logps_target)
            - 1
        )

        # GRPO policy gradient with group-relative advantage
        per_token_loss = (
            torch.exp(per_step_logps_target - per_step_logps_target.detach())
            * advantages.unsqueeze(1)
        )
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        loss = ((per_token_loss * action_mask).sum(dim=1) / action_len.clamp(min=1)).mean()

        return loss, total_reward

    # ── recurrence-scaling evaluation ──────────────────────────────────

    def _eval_recurrence_scaling(
        self,
        model: Any,
        prepared_data: Any,
    ) -> dict[str, dict[str, float]]:
        accelerator = self.create_accelerator()

        test_recurrence_ratios = getattr(model.config, "test_recurrence_ratios", None)
        mean_T = model.config.mean_recurrence

        recurrence_list: list[int] = [int(mean_T)]
        for ratio in test_recurrence_ratios or ():
            r = max(1, int(mean_T * float(ratio)))
            recurrence_list.append(r)
        recurrence_list = sorted(set(recurrence_list))

        scaling_results: dict[str, dict[str, float]] = {}
        for recurrence in recurrence_list:
            model._eval_recurrence_override = recurrence
            try:
                eval_result = self._run_evaluation(
                    model,
                    prepared_data,
                    split="test",
                    accelerator=accelerator,
                    model_is_prepared=False,
                )
                scaling_results[str(recurrence)] = eval_result["metrics"]
            finally:
                model._eval_recurrence_override = None
        return scaling_results

    def _log_scaling_results(self, scaling_results: dict[str, dict[str, float]]) -> None:
        if (logger := getattr(self, "_logger", None)) is None:
            return
        flat: dict[str, float] = {}
        for r, metrics in scaling_results.items():
            for name, val in metrics.items():
                flat[f"{name} (r={r})"] = val
        logger.log_test({"protocol": "recurrence_scaling", "metrics": flat})
