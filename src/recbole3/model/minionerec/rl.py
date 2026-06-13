from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import Sampler
from accelerate.utils import set_seed
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper, Trainer

# TRL is optional. Some TRL versions require newer torch FSDP symbols (e.g. FSDPModule),
# so we treat it as best-effort and fall back to fixed reference-model behavior.
try:  # pragma: no cover - runtime dependency
    from trl import SyncRefModelCallback as _TRLSyncRefModelCallback
except Exception:  # noqa: BLE001 - we want to tolerate any TRL import issues at runtime.
    _TRLSyncRefModelCallback = None

from recbole3.model.minionerec.config import MiniOneRecConfig
from recbole3.model.minionerec.logits import (
    MiniOneRecConstrainedLogitsProcessor,
    build_minionerec_prefix_allowed_tokens,
)
from recbole3.model.minionerec.rewards import RewardFunc, normalize_minionerec_rule_text


@dataclass(frozen=True, slots=True)
class _RLConstraintPrefixEntry:
    prefix_allowed_tokens_fn: Any
    prefix_token_count: int
    has_allowed_semantic_ids: bool


class MiniOneRecRepeatRandomSampler(Sampler):
    """Original MiniOneRec sampler: shuffle prompts and repeat each one G times."""

    def __init__(self, data_source: Sized, repeat_count: int, seed: int | None = None) -> None:
        self.data_source = data_source
        self.repeat_count = int(repeat_count)
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def __iter__(self):
        indexes = [
            index
            for index in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self) -> int:
        return self.num_samples * self.repeat_count


class MiniOneRecGRPOTrainer(Trainer):
    """Minimal MiniOneRec GRPO trainer, restricted to the original paper/codepath."""

    def __init__(
        self,
        *,
        config: MiniOneRecConfig,
        ref_model: nn.Module,
        tokenizer: Any,
        semantic_ids: Sequence[str],
        sid_to_item_ids: Mapping[str, Sequence[int]] | None = None,
        prompt2excluded_item_ids: Mapping[str, Sequence[int]] | None = None,
        reward_funcs: Sequence[RewardFunc],
        **kwargs: Any,
    ) -> None:
        self.minionerec_config = config
        self.ref_model = ref_model
        self.processing_class = tokenizer
        self.semantic_ids = tuple(semantic_ids)
        self.sid_to_item_ids = {
            str(sid): tuple(int(item_id) for item_id in item_ids)
            for sid, item_ids in dict(sid_to_item_ids or {}).items()
        }
        self.prompt2excluded_item_ids = {
            str(prompt): tuple(int(item_id) for item_id in item_ids)
            for prompt, item_ids in dict(prompt2excluded_item_ids or {}).items()
        }
        self.reward_funcs = tuple(reward_funcs)
        self.num_generations = int(config.rl_num_generations)
        self.beta = float(config.rl_beta)
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._constraint_prefix_cache: OrderedDict[tuple[int, ...], _RLConstraintPrefixEntry] = OrderedDict()
        super().__init__(processing_class=tokenizer, **kwargs)
        self._validate_batch_size("train", int(self.args.per_device_train_batch_size))
        eval_strategy = getattr(self.args, "eval_strategy", getattr(self.args, "evaluation_strategy", "no"))
        if str(eval_strategy) != "no":
            self._validate_batch_size("eval", int(self.args.per_device_eval_batch_size))
        set_seed(int(self.args.seed), device_specific=True)
        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if bool(config.rl_sync_ref_model):
            if _TRLSyncRefModelCallback is None:
                raise ImportError(
                    "MiniOneRecConfig.rl_sync_ref_model=True requires a compatible `trl` installation, "
                    "but importing `trl.SyncRefModelCallback` failed in this environment. "
                    "Either install a TRL version compatible with your torch, or set rl_sync_ref_model=false."
                )
            self.add_callback(_TRLSyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        self.model_accepts_loss_kwargs = False

    def _set_signature_columns_if_needed(self) -> None:
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "completion", "excluded_item_ids"]

    def _validate_batch_size(self, split: str, per_device_batch_size: int) -> None:
        batch_size = int(per_device_batch_size)
        valid_values = [value for value in range(2, batch_size + 1) if batch_size % value == 0]
        if self.num_generations not in valid_values:
            raise ValueError(
                f"The per-device MiniOneRec GRPO {split} batch size ({per_device_batch_size}) must be divisible by "
                f"rl_num_generations ({self.num_generations}). Valid values: {valid_values}."
            )

    def _get_train_sampler(self, train_dataset: Sized | None = None) -> Sampler[int] | None:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return MiniOneRecRepeatRandomSampler(dataset, self.num_generations, seed=int(self.args.seed))

    def _get_eval_sampler(self, eval_dataset: Sized) -> Sampler[int]:
        return MiniOneRecRepeatRandomSampler(eval_dataset, self.num_generations, seed=int(self.args.seed))

    def _prepare_inputs(self, inputs: list[dict[str, Any]] | dict[str, Any]) -> dict[str, torch.Tensor]:
        prompts_text, excluded_item_ids = self._prompt_rows_from_inputs(inputs)
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        if self.minionerec_config.rl_max_prompt_length is not None:
            max_prompt_length = int(self.minionerec_config.rl_max_prompt_length)
            prompt_ids = prompt_ids[:, -max_prompt_length:]
            prompt_mask = prompt_mask[:, -max_prompt_length:]

        prompt_completion_ids = self._generate_prompt_completions(
            prompt_ids,
            prompt_mask,
            prompts_text=prompts_text,
            excluded_item_ids=excluded_item_ids,
        )
        prompt_length = int(prompt_ids.shape[1])
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = self._completion_mask(completion_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep=int(completion_ids.shape[1]),
            )

        completions_text = _batch_decode(
            self.processing_class,
            completion_ids,
            base_model=self.minionerec_config.model_name_or_path or self.minionerec_config.model_checkpoint_path or "",
        )
        rewards = self._compute_rewards(prompts_text, completions_text, device=prompt_ids.device)
        gathered_rewards = self.accelerator.gather(rewards)
        grouped_rewards = gathered_rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (gathered_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            int(self.accelerator.process_index) * len(prompts_text),
            (int(self.accelerator.process_index) + 1) * len(prompts_text),
        )
        advantages = advantages[process_slice]

        self._metrics["reward"].append(float(gathered_rewards.mean().detach().cpu().item()))
        self._metrics["reward_std"].append(float(std_grouped_rewards.mean().detach().cpu().item()))

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def _prompt_rows_from_inputs(
        self,
        inputs: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        if isinstance(inputs, dict):
            prompts_text = [str(prompt) for prompt in inputs["prompt"]]
            raw_excluded = inputs.get("excluded_item_ids")
            if raw_excluded is None:
                excluded_item_ids = [self.prompt2excluded_item_ids.get(prompt, ()) for prompt in prompts_text]
            else:
                if len(raw_excluded) != len(prompts_text):
                    raise ValueError(
                        "MiniOneRec GRPO batch has mismatched prompt and excluded_item_ids lengths: "
                        f"{len(prompts_text)} prompts vs {len(raw_excluded)} excluded rows."
                    )
                excluded_item_ids = [_normalize_excluded_item_ids(row) for row in raw_excluded]
            return prompts_text, excluded_item_ids

        prompts_text: list[str] = []
        excluded_item_ids: list[tuple[int, ...]] = []
        for example in inputs:
            prompt = str(example["prompt"])
            prompts_text.append(prompt)
            excluded_item_ids.append(
                _normalize_excluded_item_ids(
                    example.get("excluded_item_ids", self.prompt2excluded_item_ids.get(prompt, ()))
                )
            )
        return prompts_text, excluded_item_ids

    def _generate_prompt_completions(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        *,
        prompts_text: Sequence[str],
        excluded_item_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        generation_model = self.accelerator.unwrap_model(self.model)
        if int(prompt_ids.shape[0]) % self.num_generations != 0:
            raise ValueError(
                "MiniOneRec constrained beam search expects sampler-repeated prompt batches divisible by "
                f"rl_num_generations ({self.num_generations})."
            )
        if len(excluded_item_ids) != len(prompts_text):
            raise ValueError(
                "MiniOneRec GRPO constrained rollout received mismatched prompt and excluded history counts: "
                f"{len(prompts_text)} prompts vs {len(excluded_item_ids)} excluded rows."
            )
        excluded_groups = excluded_item_ids[:: self.num_generations]
        constraint_prefix = self._constraint_prefix_for_excluded_groups(excluded_groups)
        constraint_processor = MiniOneRecConstrainedLogitsProcessor(
            constraint_prefix.prefix_allowed_tokens_fn,
            num_beams=self.num_generations,
            prefix_token_count=constraint_prefix.prefix_token_count,
            eos_token_id=self.processing_class.eos_token_id,
        )
        logits_processor = LogitsProcessorList(
            [
                TemperatureLogitsWarper(temperature=float(self.minionerec_config.rl_temperature)),
                constraint_processor,
            ]
        )
        generation_config = GenerationConfig(
            max_new_tokens=int(self.minionerec_config.rl_max_completion_length),
            length_penalty=float(self.minionerec_config.length_penalty),
            num_beams=self.num_generations,
            num_return_sequences=self.num_generations,
            top_k=None,
            top_p=None,
            do_sample=True,
            temperature=float(self.minionerec_config.rl_temperature),
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        generated = generation_model.generate(
            prompt_ids[:: self.num_generations],
            attention_mask=prompt_mask[:: self.num_generations],
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
        self._record_constraint_processor_stats(constraint_processor)
        return generated

    def _constraint_prefix_for_excluded_groups(self, excluded_groups: Sequence[Sequence[int]]) -> _RLConstraintPrefixEntry:
        if not bool(self.minionerec_config.rl_exclude_history):
            return self._constraint_prefix_for_excluded(())

        entries = [self._constraint_prefix_for_excluded(excluded_item_ids) for excluded_item_ids in excluded_groups]
        if not entries:
            return self._constraint_prefix_for_excluded(())
        prefix_token_count = int(
            next(
                (entry.prefix_token_count for entry in entries if entry.has_allowed_semantic_ids),
                entries[0].prefix_token_count,
            )
        )

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: list[int]) -> list[int]:
            entry = entries[int(batch_id)]
            if not entry.has_allowed_semantic_ids:
                return []
            return entry.prefix_allowed_tokens_fn(batch_id, input_ids)

        return _RLConstraintPrefixEntry(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            prefix_token_count=prefix_token_count,
            has_allowed_semantic_ids=any(entry.has_allowed_semantic_ids for entry in entries),
        )

    def _constraint_prefix_for_excluded(self, excluded_item_ids: Sequence[int]) -> _RLConstraintPrefixEntry:
        excluded_key = tuple(sorted({int(item_id) for item_id in excluded_item_ids}))
        cache_size = max(0, int(getattr(self.minionerec_config, "constraint_cache_size", 32)))
        if cache_size == 0:
            return self._build_constraint_prefix_entry(excluded_key)

        cached = self._constraint_prefix_cache.get(excluded_key)
        if cached is not None:
            self._constraint_prefix_cache.move_to_end(excluded_key)
            return cached

        entry = self._build_constraint_prefix_entry(excluded_key)
        self._constraint_prefix_cache[excluded_key] = entry
        if len(self._constraint_prefix_cache) > cache_size:
            self._constraint_prefix_cache.popitem(last=False)
        return entry

    def _build_constraint_prefix_entry(self, excluded_item_ids: Sequence[int]) -> _RLConstraintPrefixEntry:
        allowed_semantic_ids = _allowed_semantic_ids(
            self.semantic_ids,
            self.sid_to_item_ids,
            excluded_item_ids,
        )
        if not allowed_semantic_ids:
            return _RLConstraintPrefixEntry(
                prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: [],
                prefix_token_count=int(self.minionerec_config.constraint_prefix_token_count or 0),
                has_allowed_semantic_ids=False,
            )
        prefix_allowed_tokens_fn, prefix_token_count = build_minionerec_prefix_allowed_tokens(
            self.processing_class,
            allowed_semantic_ids,
            base_model=self.minionerec_config.model_name_or_path or self.minionerec_config.model_checkpoint_path or "",
            prefix_token_count=self.minionerec_config.constraint_prefix_token_count,
        )
        return _RLConstraintPrefixEntry(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            prefix_token_count=int(prefix_token_count),
            has_allowed_semantic_ids=True,
        )

    def _compute_rewards(self, prompts: list[str], completions: list[str], *, device: torch.device) -> torch.Tensor:
        reward_values = torch.zeros(len(prompts), len(self.reward_funcs), dtype=torch.float32, device=device)
        for index, reward_func in enumerate(self.reward_funcs):
            reward_values[:, index] = torch.tensor(
                reward_func(prompts, completions),
                dtype=torch.float32,
                device=device,
            )
        self._record_completion_validity(completions)
        return reward_values.sum(dim=1)

    def _record_constraint_processor_stats(self, processor: MiniOneRecConstrainedLogitsProcessor) -> None:
        for key, value in processor.stats().items():
            self._metrics[key].append(float(value))

    def _record_completion_validity(self, completions: list[str]) -> None:
        valid_sids = {normalize_minionerec_rule_text(sid) for sid in self.semantic_ids}
        if not completions:
            return
        valid_count = sum(1 for completion in completions if normalize_minionerec_rule_text(completion) in valid_sids)
        total = len(completions)
        self._metrics["valid_generation_rate"].append(valid_count / total)
        self._metrics["invalid_generation_rate"].append((total - valid_count) / total)
        self._metrics["invalid_generation_count"].append(float(total - valid_count))

    def compute_loss(self, model: nn.Module, inputs: dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: Any = None) -> torch.Tensor:
        del num_items_in_batch
        if return_outputs:
            raise ValueError("MiniOneRecGRPOTrainer does not support returning outputs.")

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = int(completion_ids.shape[1])

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        safe_lengths = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / safe_lengths).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / safe_lengths).mean()
        self._metrics["completion_length"].append(float(completion_length))
        self._metrics["kl"].append(float(self.accelerator.gather_for_metrics(mean_kl).mean().item()))
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: list[dict[str, str]] | dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        del prediction_loss_only
        del ignore_keys
        prepared_inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, prepared_inputs)
        return loss.detach(), None, None

    def _get_per_token_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        except TypeError:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        return _selective_log_softmax(logits, target_ids)

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.processing_class.eos_token_id
        is_eos = completion_ids == eos_token_id
        eos_index = torch.full((is_eos.shape[0],), is_eos.shape[1], dtype=torch.long, device=completion_ids.device)
        has_eos = is_eos.any(dim=1)
        eos_index[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        sequence_indices = torch.arange(is_eos.shape[1], device=completion_ids.device).expand(is_eos.shape[0], -1)
        return (sequence_indices <= eos_index.unsqueeze(1)).int()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        if logs and next(iter(logs)).startswith("eval_"):
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs = {**logs, **metrics}
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)
        self._metrics.clear()


def _selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    logps = torch.nn.functional.log_softmax(logits, dim=-1)
    return torch.gather(logps, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


def _allowed_semantic_ids(
    semantic_ids: Sequence[str],
    sid_to_item_ids: Mapping[str, Sequence[int]],
    excluded_item_ids: Sequence[int],
) -> tuple[str, ...]:
    excluded = {int(item_id) for item_id in excluded_item_ids}
    if not excluded:
        return tuple(str(sid) for sid in semantic_ids)
    if not sid_to_item_ids:
        return tuple(str(sid) for sid in semantic_ids)
    return tuple(
        str(sid)
        for sid in semantic_ids
        if any(int(item_id) not in excluded for item_id in sid_to_item_ids.get(str(sid), ()))
    )


def _normalize_excluded_item_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return tuple(int(item_id) for item_id in value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (str, bytes)):
        return (int(value),)
    try:
        return tuple(int(item_id) for item_id in value)
    except TypeError:
        return (int(value),)


def _batch_decode(tokenizer: Any, completion_ids: torch.Tensor, *, base_model: str) -> list[str]:
    if "llama" in str(base_model).lower():
        decoded = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    else:
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [str(text).split("Response:\n")[-1] for text in decoded]


__all__ = [
    "MiniOneRecGRPOTrainer",
]
