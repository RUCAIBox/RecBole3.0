from __future__ import annotations

import copy
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from recbole3.dataset import SEEN_ITEM_IDS
from recbole3.evaluation import EvalConfig, MetricSpec
from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.minionerec.config import MiniOneRecConfig
from recbole3.model.minionerec.data import (
    MINIONEREC_SEQREC_INSTRUCTION,
    MiniOneRecSIDCodec,
    MiniOneRecTokenizerAdapter,
    build_minionerec_rl_datasets,
    build_minionerec_sft_datasets,
    load_minionerec_sid_codec,
)
from recbole3.model.minionerec.logits import (
    MiniOneRecConstrainedLogitsProcessor,
    build_minionerec_prefix_allowed_tokens,
)
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig


logger = logging.getLogger(__name__)

_DEFAULT_CONSTRAINT_CACHE_SIZE = 32
_DEFAULT_LARGE_EVAL_WARNING_THRESHOLD = 10_000


def _hf_sft_length_grouping_kwargs(config: MiniOneRecConfig) -> dict[str, Any]:
    """Map group_by_length to the HF TrainingArguments field supported by the installed transformers."""

    import inspect

    from transformers import TrainingArguments

    signature = inspect.signature(TrainingArguments.__init__)
    if "train_sampling_strategy" in signature.parameters:
        strategy = "group_by_length" if bool(config.group_by_length) else "random"
        return {"train_sampling_strategy": strategy}
    if "group_by_length" in signature.parameters:
        return {"group_by_length": bool(config.group_by_length)}
    return {}


@dataclass(frozen=True, slots=True)
class _ConstraintPrefixCacheEntry:
    prefix_allowed_tokens_fn: Any
    prefix_token_count: int
    has_allowed_semantic_ids: bool


class MiniOneRecTrainer:
    """MiniOneRec SFT trainer and constrained-generation evaluator."""

    def __init__(self, config: MiniOneRecConfig):
        self.config = config

    def run(self, task_data: Any, output_dir: str | Path) -> dict[str, Any]:
        stage = str(self.config.pipeline_stage).strip().lower()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not str(self.config.sid_file or "").strip():
            raise ValueError("MiniOneRecConfig.sid_file must point to a MiniOneRec item.index.json file.")

        if stage == "sft":
            return self.run_sft(task_data, output_dir=output_path)
        if stage == "grpo":
            return self.run_rl(task_data, output_dir=output_path)
        if stage == "evaluation":
            checkpoint_path = self.config.model_checkpoint_path
            if not checkpoint_path:
                raise ValueError("MiniOneRecConfig.model_checkpoint_path must be set when pipeline_stage='evaluation'.")
            return self.evaluate(task_data, checkpoint_path=checkpoint_path, output_dir=output_path)
        raise ValueError(f"Unknown MiniOneRec pipeline_stage '{self.config.pipeline_stage}'.")

    def run_sft(self, task_data: Any, *, output_dir: Path) -> dict[str, Any]:
        from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments

        codec = self._load_codec(task_data)
        tokenizer, original_vocab_size = self._load_tokenizer(
            self.config.model_name_or_path,
            codec,
            padding_side="left",
        )
        tokenizer.save_pretrained(output_dir)
        train_dataset, valid_dataset = build_minionerec_sft_datasets(self.config, codec, tokenizer, task_data)
        model = self._load_train_model(tokenizer, original_vocab_size=original_vocab_size)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )
        has_validation = len(valid_dataset) > 0
        training_args = TrainingArguments(
            run_name=Path(output_dir).name,
            per_device_train_batch_size=int(self.config.train_batch_size),
            per_device_eval_batch_size=int(self.config.eval_batch_size),
            gradient_accumulation_steps=int(self.config.gradient_accumulation_steps),
            warmup_steps=int(self.config.warmup_steps),
            num_train_epochs=float(self.config.num_train_epochs),
            learning_rate=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            bf16=bool(self.config.bf16),
            fp16=bool(self.config.fp16),
            logging_steps=int(self.config.logging_steps),
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
            dataloader_num_workers=int(self.config.dataloader_num_workers),
            eval_strategy="steps" if has_validation else "no",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            output_dir=str(output_dir),
            save_total_limit=int(self.config.save_total_limit),
            load_best_model_at_end=bool(self.config.load_best_model_at_end and has_validation),
            **_hf_sft_length_grouping_kwargs(self.config),
            deepspeed=self.config.deepspeed,
            report_to=self.config.report_to,
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset if len(valid_dataset) > 0 else None,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=(
                [EarlyStoppingCallback(early_stopping_patience=int(self.config.early_stopping_patience))]
                if has_validation and int(self.config.early_stopping_patience) > 0
                else None
            ),
        )
        model.config.use_cache = False
        train_result = trainer.train()
        trainer.save_state()
        trainer.save_model(str(output_dir))

        final_dir = output_dir / "final_checkpoint"
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        result: dict[str, Any] = {
            "checkpoint_path": str(final_dir),
            "train": train_result.metrics,
            "num_train_examples": len(train_dataset),
            "num_valid_examples": len(valid_dataset),
        }
        if self.config.evaluate_after_training:
            generation_model = _unwrap_model_for_generation(trainer, trainer.model)
            result["valid"] = self._evaluate_with_model(
                generation_model,
                tokenizer,
                codec,
                task_data,
                split="valid",
                output_dir=output_dir,
            )
            if isinstance(result.get("valid"), dict) and isinstance(result["valid"].get("metrics"), dict):
                logger.info("MiniOneRec eval:valid metrics=%s", result["valid"]["metrics"])
            result["test"] = self._evaluate_with_model(
                generation_model,
                tokenizer,
                codec,
                task_data,
                split="test",
                output_dir=output_dir,
            )
            if isinstance(result.get("test"), dict) and isinstance(result["test"].get("metrics"), dict):
                logger.info("MiniOneRec eval:test metrics=%s", result["test"]["metrics"])
        return result

    def run_rl(self, task_data: Any, *, output_dir: Path) -> dict[str, Any]:
        from transformers import TrainingArguments

        from recbole3.model.minionerec.rewards import build_minionerec_reward_functions
        from recbole3.model.minionerec.rl import MiniOneRecGRPOTrainer

        model_path = self.config.model_checkpoint_path or self.config.model_name_or_path
        if not model_path:
            raise ValueError("MiniOneRec RL requires model_checkpoint_path or model_name_or_path.")
        if int(self.config.rl_num_generations) < 2:
            raise ValueError("MiniOneRec GRPO requires rl_num_generations >= 2 for group-normalized rewards.")

        codec = self._load_codec(task_data)
        tokenizer, _ = self._load_tokenizer(str(model_path), codec, padding_side="left")
        tokenizer.save_pretrained(output_dir)
        rl_datasets = build_minionerec_rl_datasets(self.config, codec, task_data)
        model = self._load_rl_model(str(model_path), tokenizer)
        ref_model = _create_reference_model(model)
        reward_funcs = build_minionerec_reward_functions(
            self.config,
            prompt2history=rl_datasets.prompt2history,
            history2target=rl_datasets.history2target,
        )

        has_validation = len(rl_datasets.eval_dataset) > 0
        training_args = TrainingArguments(
            run_name=Path(output_dir).name,
            per_device_train_batch_size=int(self.config.rl_train_batch_size),
            per_device_eval_batch_size=int(self.config.rl_eval_batch_size),
            gradient_accumulation_steps=int(self.config.rl_gradient_accumulation_steps),
            num_train_epochs=float(self.config.rl_num_train_epochs),
            learning_rate=float(self.config.rl_learning_rate),
            warmup_ratio=float(self.config.rl_warmup_ratio),
            max_grad_norm=float(self.config.rl_max_grad_norm),
            bf16=bool(self.config.bf16),
            fp16=bool(self.config.fp16),
            logging_steps=int(self.config.logging_steps),
            optim=self.config.rl_optim,
            lr_scheduler_type=self.config.rl_lr_scheduler_type,
            eval_strategy="steps" if has_validation else "no",
            eval_steps=self.config.rl_eval_steps,
            save_strategy="steps",
            save_steps=self.config.rl_save_steps,
            output_dir=str(output_dir),
            save_total_limit=int(self.config.rl_save_total_limit),
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
            dataloader_num_workers=int(self.config.dataloader_num_workers),
            report_to=self.config.rl_report_to,
            remove_unused_columns=False,
        )
        trainer = MiniOneRecGRPOTrainer(
            config=self.config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            semantic_ids=tuple(codec.sid_to_item),
            sid_to_item_ids=dict(codec.sid_to_items),
            prompt2excluded_item_ids=rl_datasets.prompt2excluded_item_ids,
            reward_funcs=reward_funcs,
            train_dataset=rl_datasets.train_dataset,
            eval_dataset=rl_datasets.eval_dataset if has_validation else None,
            args=training_args,
            data_collator=lambda features: features,
        )
        train_result = trainer.train()
        trainer.save_model(str(output_dir))

        final_dir = output_dir / "final_checkpoint"
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        result: dict[str, Any] = {
            "checkpoint_path": str(final_dir),
            "train": train_result.metrics,
            "num_train_examples": len(rl_datasets.train_dataset),
            "num_valid_examples": len(rl_datasets.eval_dataset),
            "reward_type": self.config.rl_reward_type,
        }
        if self.config.evaluate_after_rl:
            result.update(
                self._evaluate_after_rl(
                    trainer,
                    tokenizer=tokenizer,
                    codec=codec,
                    task_data=task_data,
                    output_dir=output_dir,
                )
            )
        return result

    def _evaluate_after_rl(
        self,
        trainer: Any,
        *,
        tokenizer: Any,
        codec: MiniOneRecSIDCodec,
        task_data: Any,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run a quick post-GRPO eval on rank 0 only; other ranks wait at a barrier."""
        import torch.distributed as dist

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        skipped = {
            "valid": {"skipped": True, "reason": "MiniOneRec generation evaluation requires a single process"},
            "test": {"skipped": True, "reason": "MiniOneRec generation evaluation requires a single process"},
        }
        if world_size > 1:
            if local_rank == 0:
                logger.error(
                    "MiniOneRec evaluate_after_rl is skipped under WORLD_SIZE=%d. "
                    "Run pipeline_stage=evaluation in a single process for Recall@K/NDCG.",
                    world_size,
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return skipped

        generation_model = _unwrap_model_for_generation(trainer, trainer.model)
        out: dict[str, Any] = {}
        out["valid"] = self._evaluate_with_model(
            generation_model,
            tokenizer,
            codec,
            task_data,
            split="valid",
            output_dir=output_dir,
        )
        if isinstance(out.get("valid"), dict) and isinstance(out["valid"].get("metrics"), dict):
            logger.info("MiniOneRec eval:valid metrics=%s", out["valid"]["metrics"])
        out["test"] = self._evaluate_with_model(
            generation_model,
            tokenizer,
            codec,
            task_data,
            split="test",
            output_dir=output_dir,
        )
        if isinstance(out.get("test"), dict) and isinstance(out["test"].get("metrics"), dict):
            logger.info("MiniOneRec eval:test metrics=%s", out["test"]["metrics"])

        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        return out

    def evaluate(self, task_data: Any, *, checkpoint_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
        codec = self._load_codec(task_data)
        tokenizer, _ = self._load_tokenizer(str(checkpoint_path), codec, padding_side="left")
        model = self._load_eval_model(str(checkpoint_path), tokenizer)
        evaluation_output_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent

        result: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "valid": self._evaluate_with_model(
                model,
                tokenizer,
                codec,
                task_data,
                split="valid",
                output_dir=evaluation_output_dir,
            ),
        }
        if isinstance(result.get("valid"), dict) and isinstance(result["valid"].get("metrics"), dict):
            logger.info("MiniOneRec eval:valid metrics=%s", result["valid"]["metrics"])
        result["test"] = self._evaluate_with_model(
            model,
            tokenizer,
            codec,
            task_data,
            split="test",
            output_dir=evaluation_output_dir,
        )
        if isinstance(result.get("test"), dict) and isinstance(result["test"].get("metrics"), dict):
            logger.info("MiniOneRec eval:test metrics=%s", result["test"]["metrics"])
        return result

    def _evaluate_with_model(
        self,
        model: Any,
        tokenizer: Any,
        codec: MiniOneRecSIDCodec,
        task_data: Any,
        *,
        split: str,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        _ensure_single_process_generation_eval()
        if max(self.config.topk, default=0) > int(self.config.num_beams):
            raise ValueError(
                f"MiniOneRecConfig.num_beams ({self.config.num_beams}) must be >= max(topk) ({max(self.config.topk)})."
            )
        _warn_if_large_generation_eval(self.config, task_data, split=split)
        eval_model = MiniOneRecGenerationRetrievalModel(
            config=self.config,
            generation_model=model,
            tokenizer=tokenizer,
            codec=codec,
        )
        evaluation_trainer = Trainer(_build_recbole_eval_trainer_config(self.config))
        was_training = bool(getattr(model, "training", False))
        try:
            result = evaluation_trainer.evaluate(eval_model, task_data, split=split)
            generation_stats = eval_model.generation_stats()
            result["generation_stats"] = generation_stats
            result["generation_metrics"] = _generation_stats_as_metrics(generation_stats)

            inference_results = result.get("inference_results")
            if bool(self.config.save_evaluation_predictions) and output_dir is not None and inference_results is not None:
                prediction_file = _write_evaluation_predictions(
                    self.config,
                    output_dir=output_dir,
                    split=split,
                    inference_results=inference_results,
                    codec=codec,
                )
                result["prediction_file"] = str(prediction_file)
                result.pop("inference_results", None)
            return result
        finally:
            if hasattr(model, "train"):
                model.train(was_training)

    def _load_codec(self, task_data: Any) -> MiniOneRecSIDCodec:
        return load_minionerec_sid_codec(self.config, task_data)

    def _load_tokenizer(
        self,
        model_or_checkpoint_path: str,
        codec: MiniOneRecSIDCodec,
        *,
        padding_side: str,
    ) -> tuple[Any, int]:
        from transformers import AutoTokenizer

        if not model_or_checkpoint_path:
            raise ValueError("MiniOneRecConfig.model_name_or_path must be specified.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_or_checkpoint_path,
            trust_remote_code=bool(self.config.trust_remote_code),
            padding_side=padding_side,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        original_vocab_size = len(tokenizer)
        if self.config.add_sid_tokens and codec.all_tokens:
            tokenizer.add_tokens(list(codec.all_tokens))
        return tokenizer, int(original_vocab_size)

    def _load_train_model(self, tokenizer: Any, *, original_vocab_size: int) -> Any:
        from transformers import AutoConfig, AutoModelForCausalLM

        if not self.config.model_name_or_path:
            raise ValueError("MiniOneRecConfig.model_name_or_path must be specified for SFT.")
        model_kwargs = self._model_load_kwargs()
        if self.config.train_from_scratch:
            config = AutoConfig.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=bool(self.config.trust_remote_code),
            )
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=bool(self.config.trust_remote_code),
                **model_kwargs,
            )
        model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        if self.config.freeze_llm:
            self._freeze_llm_except_new_token_rows(model, original_vocab_size=original_vocab_size, vocab_size=len(tokenizer))
        return model

    def _load_rl_model(self, model_path: str, tokenizer: Any) -> Any:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=bool(self.config.trust_remote_code),
            **self._model_load_kwargs(),
        )
        if len(tokenizer) != int(model.get_input_embeddings().weight.shape[0]):
            model.resize_token_embeddings(len(tokenizer))
        model.config.use_cache = False
        return model

    def _load_eval_model(self, checkpoint_path: str, tokenizer: Any) -> Any:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=bool(self.config.trust_remote_code),
            **self._model_load_kwargs(),
        )
        if len(tokenizer) != int(model.get_input_embeddings().weight.shape[0]):
            model.resize_token_embeddings(len(tokenizer))
        return model

    def _model_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        torch_dtype = self._resolve_torch_dtype(self.config.torch_dtype)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        return kwargs

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str | None:
        normalized = str(dtype_name or "").strip()
        if not normalized:
            return None
        if normalized == "auto":
            return "auto"
        dtype = getattr(torch, normalized, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype '{dtype_name}'.")
        return dtype

    @staticmethod
    def _freeze_llm_except_new_token_rows(model: Any, *, original_vocab_size: int, vocab_size: int) -> None:
        if int(vocab_size) <= int(original_vocab_size):
            raise ValueError("freeze_llm=True requires add_sid_tokens to add at least one new token.")
        for parameter in model.parameters():
            parameter.requires_grad = False

        seen_params: set[int] = set()
        for embedding in (model.get_input_embeddings(), model.get_output_embeddings()):
            if embedding is None or not hasattr(embedding, "weight"):
                continue
            weight = embedding.weight
            if id(weight) in seen_params:
                continue
            seen_params.add(id(weight))
            weight.requires_grad = True
            weight.register_hook(_new_token_row_gradient_hook(int(original_vocab_size)))


class MiniOneRecGenerationCollator(BaseCollator):
    """Build MiniOneRec prompt tensors for RecBole retrieval evaluation."""

    def __init__(
        self,
        config: MiniOneRecConfig,
        prepared_data: Any,
        tokenizer: Any,
        codec: MiniOneRecSIDCodec,
    ) -> None:
        super().__init__(config, prepared_data=prepared_data)
        self.tokenizer = tokenizer
        self.codec = codec
        self.adapter = MiniOneRecTokenizerAdapter(tokenizer)

    def __call__(self, records: Any) -> dict[str, torch.Tensor]:
        rows = records.to_dict("records") if hasattr(records, "columns") else list(records)
        features = [self._encode_record(record) for record in rows]
        return _collate_generation_features(features, self.tokenizer)

    def _encode_record(self, record: dict[str, Any]) -> dict[str, list[int]]:
        history_item_ids = tuple(int(item_id) for item_id in (record.get(SEEN_ITEM_IDS) or ()))
        user_input = _generation_user_input(self.config, self.codec, history_item_ids)
        prompt = _generate_prompt(user_input, output="")
        tokens = self.adapter.encode(MINIONEREC_SEQREC_INSTRUCTION, bos=True, eos=False)
        tokens = tokens + self.adapter.encode(prompt, bos=False, eos=False)
        tokens = tokens[-int(self.config.eval_max_len) :]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }


class MiniOneRecGenerationRetrievalModel(BaseRetrievalModel):
    """Eval-only RecBole retrieval wrapper around MiniOneRec constrained generation."""

    def __init__(
        self,
        config: MiniOneRecConfig,
        generation_model: Any,
        tokenizer: Any,
        codec: MiniOneRecSIDCodec,
    ) -> None:
        super().__init__(config)
        self.generation_model = generation_model
        self.tokenizer = tokenizer
        self.codec = codec
        self._generation_stats = _empty_generation_stats()
        self._constraint_prefix_cache: OrderedDict[tuple[int, frozenset[int]], _ConstraintPrefixCacheEntry] = OrderedDict()

    def build_train_collator(self, prepared_data: Any) -> BaseCollator:
        raise NotImplementedError("MiniOneRecGenerationRetrievalModel is evaluation-only.")

    def build_eval_collator(self, prepared_data: Any) -> BaseCollator:
        return MiniOneRecGenerationCollator(self.config, prepared_data, self.tokenizer, self.codec)

    def forward(self, batch: Any) -> dict[str, Any]:
        raise NotImplementedError("MiniOneRecGenerationRetrievalModel is evaluation-only.")

    def compute_loss(self, batch: Any, outputs: dict[str, Any]) -> Any:
        raise NotImplementedError("MiniOneRecGenerationRetrievalModel is evaluation-only.")

    def predict(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_item_ids is not None:
            raise ValueError("MiniOneRec is wired to RecBole full evaluation; sampled candidate evaluation is not supported.")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        excluded_rows = _excluded_item_rows(exclude_item_ids, exclude_mask, batch_size=int(input_ids.shape[0]))
        predictions: list[list[int]] = [[-1] * int(k) for _ in excluded_rows]
        batch_stats = _empty_generation_stats()

        for row_indices, excluded in _group_excluded_rows(excluded_rows):
            constraint_prefix, cache_stats = self._constraint_prefix_for_excluded(excluded)
            group_predictions, group_stats = _generate_batch_predictions(
                self.config,
                self.generation_model,
                self.tokenizer,
                self.codec,
                input_ids=input_ids[row_indices],
                attention_mask=attention_mask[row_indices],
                excluded=excluded,
                k=int(k),
                constraint_prefix=constraint_prefix,
            )
            _merge_generation_stats(group_stats, cache_stats)
            for offset, row_index in enumerate(row_indices):
                predictions[row_index] = group_predictions[offset]
            _merge_generation_stats(batch_stats, group_stats)

        _merge_generation_stats(self._generation_stats, batch_stats)
        return torch.as_tensor(predictions, dtype=torch.long, device=input_ids.device)

    def generation_stats(self) -> dict[str, int]:
        return dict(self._generation_stats)

    def _constraint_prefix_for_excluded(self, excluded: set[int]) -> tuple[_ConstraintPrefixCacheEntry, dict[str, int]]:
        cache_size = max(0, int(getattr(self.config, "constraint_cache_size", _DEFAULT_CONSTRAINT_CACHE_SIZE)))
        if cache_size == 0:
            return _build_constraint_prefix_cache_entry(self.config, self.tokenizer, self.codec, excluded), {
                "constraint_cache_misses": 1,
            }

        key = (id(self.codec), frozenset(int(item_id) for item_id in excluded))
        cached = self._constraint_prefix_cache.get(key)
        if cached is not None:
            self._constraint_prefix_cache.move_to_end(key)
            return cached, {"constraint_cache_hits": 1}

        entry = _build_constraint_prefix_cache_entry(self.config, self.tokenizer, self.codec, excluded)
        self._constraint_prefix_cache[key] = entry
        cache_stats = {"constraint_cache_misses": 1}
        if len(self._constraint_prefix_cache) > cache_size:
            self._constraint_prefix_cache.popitem(last=False)
            cache_stats["constraint_cache_evictions"] = 1
        return entry, cache_stats


def _build_recbole_eval_trainer_config(config: MiniOneRecConfig) -> TrainerConfig:
    return TrainerConfig(
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        dataloader_num_workers=int(config.dataloader_num_workers),
        pin_memory=False,
        max_epochs=0,
        save_inference_results=bool(config.save_evaluation_predictions),
        eval=EvalConfig(
            protocol="full",
            metrics=tuple(MetricSpec(name=str(metric), ks=tuple(int(k) for k in config.topk)) for metric in config.metrics),
            neg_sampling_num=0,
            candidate_seed=42,
            exclude_history=bool(config.exclude_history),
        ),
    )


def _ensure_single_process_generation_eval() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    logger.error(
        "MiniOneRec generation-based full evaluation is not distributed-safe yet because RecBole Trainer "
        "does not gather retrieval predictions across ranks before metric computation. "
        "Run this stage with one process, for example CUDA_VISIBLE_DEVICES=0 and pipeline_stage=evaluation."
    )
    raise RuntimeError(
        "MiniOneRec generation evaluation requires a single process. "
        "Run pipeline_stage=evaluation with one GPU/process until prediction gathering is implemented."
    )


def _warn_if_large_generation_eval(config: MiniOneRecConfig, task_data: Any, *, split: str) -> None:
    threshold = int(getattr(config, "large_eval_warning_threshold", _DEFAULT_LARGE_EVAL_WARNING_THRESHOLD))
    if threshold <= 0:
        return
    try:
        eval_size = len(task_data.get_eval_dataset(split))
    except Exception as exc:  # pragma: no cover - dataset backends may expose split sizes differently.
        logger.debug("Could not inspect MiniOneRec %s evaluation split size: %s", split, exc)
        return
    if int(eval_size) < threshold:
        return
    logger.warning(
        "MiniOneRec generation-based full evaluation on split '%s' has %d rows. "
        "This may take a long time because every row runs constrained beam generation "
        "(num_beams=%d, max_new_tokens=%d, exclude_history=%s, constraint_cache_size=%d).",
        split,
        int(eval_size),
        int(config.num_beams),
        int(config.max_new_tokens),
        bool(config.exclude_history),
        int(getattr(config, "constraint_cache_size", _DEFAULT_CONSTRAINT_CACHE_SIZE)),
    )


def _build_constraint_prefix_cache_entry(
    config: MiniOneRecConfig,
    tokenizer: Any,
    codec: MiniOneRecSIDCodec,
    excluded: set[int],
) -> _ConstraintPrefixCacheEntry:
    allowed_semantic_ids = _allowed_semantic_ids(codec, excluded)
    if not allowed_semantic_ids:
        return _ConstraintPrefixCacheEntry(
            prefix_allowed_tokens_fn=None,
            prefix_token_count=0,
            has_allowed_semantic_ids=False,
        )
    prefix_allowed_tokens_fn, prefix_token_count = build_minionerec_prefix_allowed_tokens(
        tokenizer,
        allowed_semantic_ids,
        base_model=config.model_name_or_path or config.model_checkpoint_path or "",
        prefix_token_count=config.constraint_prefix_token_count,
    )
    return _ConstraintPrefixCacheEntry(
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        prefix_token_count=int(prefix_token_count),
        has_allowed_semantic_ids=True,
    )


def _generate_batch_predictions(
    config: MiniOneRecConfig,
    model: Any,
    tokenizer: Any,
    codec: MiniOneRecSIDCodec,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    excluded: set[int],
    k: int,
    constraint_prefix: _ConstraintPrefixCacheEntry,
) -> tuple[list[list[int]], dict[str, int]]:
    from transformers import GenerationConfig, LogitsProcessorList

    num_beams = int(config.num_beams)
    if int(k) > num_beams:
        raise ValueError(f"MiniOneRecConfig.num_beams ({num_beams}) must be >= requested k ({int(k)}).")

    stats = _empty_generation_stats()
    batch_size = int(input_ids.shape[0])
    if not constraint_prefix.has_allowed_semantic_ids:
        stats["padded_predictions"] += int(k) * batch_size
        return [[-1] * int(k) for _ in range(batch_size)], stats

    generation_config = GenerationConfig(
        num_beams=num_beams,
        length_penalty=float(config.length_penalty),
        num_return_sequences=num_beams,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=int(config.max_new_tokens),
        top_k=None,
        top_p=None,
    )
    constraint_processor = MiniOneRecConstrainedLogitsProcessor(
        constraint_prefix.prefix_allowed_tokens_fn,
        num_beams=num_beams,
        prefix_token_count=constraint_prefix.prefix_token_count,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    logits_processor = LogitsProcessorList([constraint_processor])

    device = _infer_model_input_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        logits_processor=logits_processor,
    )
    prompt_length = int(input_ids.shape[1])
    completion_ids = generated.sequences[:, prompt_length:]
    completions = _batch_decode(
        tokenizer,
        completion_ids,
        base_model=config.model_name_or_path or config.model_checkpoint_path or "",
    )
    processor_stats = constraint_processor.stats()
    _merge_generation_stats(
        stats,
        {
            "constraint_total_prefix_checks": int(processor_stats["constraint_total_prefix_checks"]),
            "constraint_valid_prefix_checks": int(processor_stats["constraint_valid_prefix_checks"]),
            "constraint_invalid_prefix_checks": int(processor_stats["constraint_invalid_prefix_checks"]),
            "constraint_forced_eos_count": int(processor_stats["constraint_forced_eos_count"]),
        },
    )
    batch_predictions: list[list[int]] = []
    for row_offset in range(batch_size):
        start = row_offset * num_beams
        row_completions = completions[start : start + num_beams]
        row_predictions, row_stats = _select_generated_item_ids(row_completions, codec, excluded=excluded, k=int(k))
        batch_predictions.append(row_predictions)
        _merge_generation_stats(stats, row_stats)
    return batch_predictions, stats


def _select_generated_item_ids(
    completions: Sequence[str],
    codec: MiniOneRecSIDCodec,
    *,
    excluded: set[int],
    k: int,
) -> tuple[list[int], dict[str, int]]:
    selected: list[int] = []
    seen_valid_items: set[int] = set()
    stats = _empty_generation_stats()
    stats["decoded_generations"] += len(completions)
    for completion in completions:
        candidate_item_ids = codec.decode_sid_candidates(_normalize_completion(completion))
        if not candidate_item_ids:
            stats["invalid_generations"] += 1
            continue
        stats["valid_generations"] += 1
        selectable_item_ids = [
            int(item_id)
            for item_id in candidate_item_ids
            if int(item_id) not in seen_valid_items and int(item_id) not in excluded
        ]
        if not selectable_item_ids and all(int(item_id) in seen_valid_items for item_id in candidate_item_ids):
            stats["duplicate_generations"] += 1
            continue
        if not selectable_item_ids:
            stats["excluded_generations"] += 1
            continue
        item_id = selectable_item_ids[0]
        seen_valid_items.add(int(item_id))
        selected.append(int(item_id))
        stats["selected_generations"] += 1
        if len(selected) == int(k):
            return selected, stats
    while len(selected) < int(k):
        selected.append(-1)
        stats["padded_predictions"] += 1
    return selected, stats


def _allowed_semantic_ids(codec: MiniOneRecSIDCodec, excluded: set[int]) -> tuple[str, ...]:
    if not excluded:
        return tuple(codec.sid_to_item)
    return tuple(
        sid
        for sid, item_ids in codec.sid_to_items.items()
        if any(int(item_id) not in excluded for item_id in item_ids)
    )


def _excluded_item_rows(
    exclude_item_ids: torch.Tensor | None,
    exclude_mask: torch.Tensor | None,
    *,
    batch_size: int,
) -> list[set[int]]:
    if exclude_item_ids is None or exclude_mask is None or exclude_item_ids.numel() == 0:
        return [set() for _ in range(int(batch_size))]
    rows: list[set[int]] = []
    item_rows = exclude_item_ids.detach().cpu()
    mask_rows = exclude_mask.detach().cpu().bool()
    for item_row, mask_row in zip(item_rows, mask_rows, strict=True):
        rows.append({int(item_id) for item_id, keep in zip(item_row.tolist(), mask_row.tolist(), strict=True) if bool(keep)})
    return rows


def _group_excluded_rows(excluded_rows: Sequence[set[int]]) -> list[tuple[list[int], set[int]]]:
    grouped: dict[tuple[int, ...], list[int]] = {}
    for row_index, excluded in enumerate(excluded_rows):
        key = tuple(sorted(int(item_id) for item_id in excluded))
        grouped.setdefault(key, []).append(int(row_index))
    return [(row_indices, set(key)) for key, row_indices in grouped.items()]


def _generation_user_input(config: MiniOneRecConfig, codec: MiniOneRecSIDCodec, history_item_ids: Sequence[int]) -> str:
    truncated_history = tuple(int(item_id) for item_id in history_item_ids[-int(config.history_max_length) :])
    history = ", ".join(codec.item_sid(item_id) for item_id in truncated_history)
    return (
        "Can you predict the next possible item the user may expect, "
        f"given the following chronological interaction history: {history}"
    )


def _generate_prompt(user_input: str, *, output: str) -> str:
    return f"""### User Input: 
{user_input}

### Response:\n{output}"""


def _collate_generation_features(features: Sequence[dict[str, list[int]]], tokenizer: Any) -> dict[str, torch.Tensor]:
    pad_token_id = int(tokenizer.pad_token_id)
    max_length = max((len(feature["input_ids"]) for feature in features), default=0)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    for feature in features:
        input_ids = list(feature["input_ids"])
        attention_mask = list(feature["attention_mask"])
        pad_width = max_length - len(input_ids)
        input_rows.append([pad_token_id] * pad_width + input_ids)
        attention_rows.append([0] * pad_width + attention_mask)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
    }


def _write_evaluation_predictions(
    config: MiniOneRecConfig,
    *,
    output_dir: Path,
    split: str,
    inference_results: dict[str, Any],
    codec: MiniOneRecSIDCodec,
) -> Path:
    configured_dir = config.evaluation_prediction_dir
    prediction_dir = Path(configured_dir) if configured_dir else Path(output_dir) / "evaluation_predictions"
    if configured_dir and not prediction_dir.is_absolute():
        prediction_dir = Path(output_dir) / prediction_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"minionerec_{split}_predictions.json"

    pred_item_ids = inference_results.get("pred_item_ids", [])
    target_item_ids = inference_results.get("target_item_ids", [])
    target_mask = inference_results.get("target_mask", [])
    rows: list[dict[str, Any]] = []
    for row_index, pred_row in enumerate(pred_item_ids):
        predictions = [int(item_id) for item_id in pred_row]
        target_ids = [
            int(item_id)
            for item_id, keep in zip(target_item_ids[row_index], target_mask[row_index], strict=True)
            if bool(keep)
        ]
        hit_rank = next((rank + 1 for rank, item_id in enumerate(predictions) if item_id in set(target_ids)), None)
        rows.append(
            {
                "index": row_index,
                "target_item_ids": target_ids,
                "target_sids": [codec.item_to_sid.get(item_id) for item_id in target_ids],
                "pred_item_ids": predictions,
                "pred_sids": [codec.item_to_sid.get(item_id) if item_id >= 0 else None for item_id in predictions],
                "hit_rank": hit_rank,
            }
        )
    with prediction_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    return prediction_path


def _new_token_row_gradient_hook(original_vocab_size: int):
    def hook(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        gradient[:original_vocab_size].zero_()
        return gradient

    return hook


def _unwrap_model_for_generation(trainer: Any, model: Any) -> Any:
    accelerator = getattr(trainer, "accelerator", None)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        try:
            return unwrap_model(model)
        except Exception as exc:  # pragma: no cover - accelerator wrappers vary by backend.
            logger.warning("Failed to unwrap MiniOneRec generation model for evaluation: %s", exc)
    return model


def _create_reference_model(model: Any) -> Any:
    try:
        from trl.models import create_reference_model

        return create_reference_model(model)
    except Exception as exc:  # pragma: no cover - TRL/torch version mismatches vary.
        logger.warning("Falling back to deepcopy reference model (TRL create_reference_model unavailable): %s", exc)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for parameter in ref_model.parameters():
            parameter.requires_grad = False
        return ref_model


def _infer_model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_decode(tokenizer: Any, completion_ids: torch.Tensor, *, base_model: str) -> list[str]:
    if "llama" in str(base_model).lower():
        decoded = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    else:
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [_normalize_completion(text) for text in decoded]


def _normalize_completion(text: str) -> str:
    return str(text).split("Response:\n")[-1].strip()


def _empty_generation_stats() -> dict[str, int]:
    return {
        "decoded_generations": 0,
        "valid_generations": 0,
        "invalid_generations": 0,
        "excluded_generations": 0,
        "duplicate_generations": 0,
        "selected_generations": 0,
        "padded_predictions": 0,
        "constraint_total_prefix_checks": 0,
        "constraint_valid_prefix_checks": 0,
        "constraint_invalid_prefix_checks": 0,
        "constraint_forced_eos_count": 0,
        "constraint_cache_hits": 0,
        "constraint_cache_misses": 0,
        "constraint_cache_evictions": 0,
    }


def _merge_generation_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = int(target.get(key, 0)) + int(value)


def _generation_stats_as_metrics(stats: dict[str, int]) -> dict[str, float]:
    decoded = max(1, int(stats.get("decoded_generations", 0)))
    selected_or_padding = max(1, int(stats.get("selected_generations", 0)) + int(stats.get("padded_predictions", 0)))
    prefix_checks = max(1, int(stats.get("constraint_total_prefix_checks", 0)))
    cache_lookups = max(1, int(stats.get("constraint_cache_hits", 0)) + int(stats.get("constraint_cache_misses", 0)))
    return {
        "generation_valid_rate": int(stats.get("valid_generations", 0)) / decoded,
        "generation_invalid_rate": int(stats.get("invalid_generations", 0)) / decoded,
        "generation_selected_rate": int(stats.get("selected_generations", 0)) / decoded,
        "generation_padding_rate": int(stats.get("padded_predictions", 0)) / selected_or_padding,
        "constraint_success_rate": int(stats.get("constraint_valid_prefix_checks", 0)) / prefix_checks,
        "constraint_invalid_prefix_rate": int(stats.get("constraint_invalid_prefix_checks", 0)) / prefix_checks,
        "constraint_cache_hit_rate": int(stats.get("constraint_cache_hits", 0)) / cache_lookups,
    }


__all__ = [
    "MiniOneRecTrainer",
]
