from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

import torch
from accelerate import Accelerator

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.testing_utils import torch_device

from recbole3.model.lcrec.config import LCRecConfig
from recbole3.model.lcrec.data import LCRecItemTokenizer, get_lcrec_sft_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LCRecTrainer:
    """LCRec trainer: handles LLM loading, SFT training via HF Trainer, and beam-search evaluation."""

    def __init__(self, config: LCRecConfig):
        self.config = config

    def _is_main_process(self) -> bool:
        """Check if current process is rank 0."""
        return int(os.environ.get("RANK", "0")) == 0


    def _log(self, msg: str, *args: Any, level: str = "info") -> None:
        """Log a message only on the main process.

        Args:
            msg: Log message (supports %-style formatting).
            *args: Format arguments for the message.
            level: Logging level — 'info', 'warning', 'error', 'debug'.
        """
        if not self._is_main_process():
            return
        getattr(logger, level)(msg, *args)

    # ------------------------------------------------------------------
    # Tokenizer loading
    # ------------------------------------------------------------------

    def _load_llm_tokenizer(
        self, item_tokenizer: LCRecItemTokenizer, padding_side: str = "right"
    ) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, use_fast=False, padding_side=padding_side
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        added = tokenizer.add_tokens(item_tokenizer.all_tokens)
        self._log("Added %d item code tokens to LLM tokenizer", added)
        return tokenizer

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, llm_tokenizer: AutoTokenizer, device_map: str | dict = "auto") -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            attn_implementation=self.config.attn_implementation,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)
        model.config.use_cache = False

        if self.config.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=list(self.config.lora_target_modules),
                modules_to_save=list(self.config.lora_modules_to_save),
                lora_dropout=self.config.lora_dropout,
                bias="none",
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log("Model architecture:", model)
        self._log("Model parameters: total=%d, trainable=%d", n_params, n_trainable)
        return model

    # ------------------------------------------------------------------
    # Training flow
    # ------------------------------------------------------------------

    def run(self, task_data: Any, output_dir: str) -> dict[str, Any]:
        """Run the full LCRec training pipeline.

        Args:
            task_data: Prepared BaseTaskDataset from RecBole3.0.
            output_dir: Directory to save model checkpoints.

        Returns:
            Dictionary with training results.
        """

        # 1. Init item tokenizer
        item_tokenizer = LCRecItemTokenizer(self.config)

        # 2. Load LLM tokenizer
        llm_tokenizer = self._load_llm_tokenizer(item_tokenizer, padding_side="right")
        if self._is_main_process():
            llm_tokenizer.save_pretrained(output_dir)

        # 3. Build SFT datasets
        train_dataset, val_dataset, test_dataset = get_lcrec_sft_datasets(
            self.config, item_tokenizer, llm_tokenizer, task_data
        )

        # 4. Load model
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            device_map: str | dict = {"": local_rank}
            torch.cuda.set_device(local_rank)
        else:
            # Avoid device_map="auto" (may shard/offload and kill throughput on single-node training).
            device_map = {"": 0} if torch.cuda.is_available() else "auto"
            if torch.cuda.is_available():
                torch.cuda.set_device(0)

        model = self._load_model(llm_tokenizer, device_map=device_map)

        # 5. Data collator
        data_collator = DataCollatorForSeq2Seq(llm_tokenizer, pad_to_multiple_of=8, padding="longest")

        # 6. Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.train_dataloader_num_workers,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            load_best_model_at_end=True,
            deepspeed=self.config.deepspeed,
            report_to="none",
        )

        # 7. HF Trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            processing_class=llm_tokenizer,
            data_collator=data_collator,
        )

        # 8. Train & save
        trainer.train()
        trainer.save_state()
        trainer.save_model()
        self._log("Model checkpoint saved to %s", output_dir)

        # # 9. Optionally evaluate
        results: dict[str, Any] = {"checkpoint_path": output_dir}
        # if test_dataset is not None and len(test_dataset) > 0:
        #     eval_results = self._evaluate_with_model(
        #         model, llm_tokenizer, item_tokenizer, test_dataset
        #     )
        #     results["eval"] = eval_results

        return results

    # ------------------------------------------------------------------
    # Evaluation flow (standalone, loads from checkpoint)
    # ------------------------------------------------------------------

    def evaluate(self, task_data: Any, checkpoint_path: str) -> dict[str, Any]:
        """Evaluate a trained LCRec model.

        Args:
            task_data: Prepared BaseTaskDataset.
            checkpoint_path: Path to trained model checkpoint.
            output_dir: Output directory for results.

        Returns:
            Dictionary with evaluation results.
        """
        item_tokenizer = LCRecItemTokenizer(self.config)

        # Load tokenizer from checkpoint (has added item tokens)
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, use_fast=False, padding_side="left"
            )
        except (OSError, ValueError):
            self._log(
                "Could not load tokenizer from checkpoint; loading from base model and re-adding tokens.",
                level="warning",
            )
            llm_tokenizer = self._load_llm_tokenizer(item_tokenizer, padding_side="left")

        # Build test dataset
        _, _, test_dataset = get_lcrec_sft_datasets(
            self.config, item_tokenizer, llm_tokenizer, task_data
        )

        # Load model
        model = self._load_eval_model(llm_tokenizer, checkpoint_path)

        results = self._evaluate_with_model(model, llm_tokenizer, item_tokenizer, test_dataset)
        return results

    def _load_eval_model(self, llm_tokenizer: AutoTokenizer, checkpoint_path: str) -> Any:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            device_map: str | dict = {"": local_rank}
            torch.cuda.set_device(local_rank)
        else:
            device_map = "auto"

        if self.config.use_lora:
            from peft import PeftModel

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                attn_implementation=self.config.attn_implementation,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            model.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)
            model = PeftModel.from_pretrained(
                model,
                checkpoint_path,
                torch_device=local_rank,
                torch_dtype=getattr(torch, self.config.torch_dtype)
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                attn_implementation=self.config.attn_implementation,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
        return model

    # ------------------------------------------------------------------
    # Evaluation core
    # ------------------------------------------------------------------

    def _evaluate_with_model(
        self,
        model: Any,
        llm_tokenizer: AutoTokenizer,
        item_tokenizer: LCRecItemTokenizer,
        test_dataset: Any,
    ) -> dict[str, Any]:
        """Run beam-search evaluation on the test dataset."""
        model.eval()

        # Parse test_prompt_ids
        test_prompt_ids = self._parse_test_prompt_ids()

        collator = DataCollatorForSeq2Seq(llm_tokenizer, padding="longest")
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=collator,
            pin_memory=True,
            num_workers=self.config.eval_dataloader_num_workers,
        )

        # DDP support
        accelerator = Accelerator()
        use_ddp = accelerator.num_processes > 1
        device = accelerator.device
        model, dataloader = accelerator.prepare(model, dataloader)


        maxk = max(self.config.topk)
        n_digit = item_tokenizer.n_digit
        eos_token_id = llm_tokenizer.eos_token_id


        all_prompt_results: list[dict[str, float]] = []

        for prompt_id in test_prompt_ids:
            self._log("Evaluating prompt %d / %d", prompt_id+1, len(test_prompt_ids))
            # Mutates dataset state — safe because DataLoader uses num_workers=0
            test_dataset.set_prompt_id(prompt_id)

            all_results: dict[str, list[torch.Tensor]] = {f"{m}@{k}": [] for m in self.config.metrics for k in self.config.topk}

            pbar = tqdm(dataloader, desc=f"Eval-Prompt-{prompt_id}", disable=not self._is_main_process())
            for batch in pbar:

                batch = {k: v.to(device) for k, v in batch.items()}

                gen_model = accelerator.unwrap_model(model)

                with torch.no_grad():
                    outputs = gen_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=n_digit + 1,
                        num_beams=self.config.num_beams,
                        num_return_sequences=maxk,
                        output_scores=False,
                        early_stopping=False,
                    )

                B, inputs_len = batch["input_ids"].shape[:2]
                preds = outputs[:, inputs_len : inputs_len + n_digit].reshape(B, maxk, -1)
                labels = batch["labels"]
                labels = labels[labels != -100].reshape(B, -1)

                if use_ddp:
                    preds, labels = accelerator.gather_for_metrics((preds, labels))

                batch_metrics = self._calculate_metrics(preds, labels, eos_token_id, self.config.metrics, self.config.topk)
                for key, value in batch_metrics.items():
                    all_results[key].append(value)

                # Live progress
                live: dict[str, float] = {}
                for metric in self.config.metrics:
                    for k in self.config.topk:
                        key = f"{metric}@{k}"
                        live[key] = torch.cat(all_results[key]).mean().item()
                pbar.set_postfix(live)

            prompt_results: dict[str, float] = {}
            for metric in self.config.metrics:
                for k in self.config.topk:
                    key = f"{metric}@{k}"
                    prompt_results[key] = torch.cat(all_results[key]).mean().item()

            self._log("Prompt %d results: %s", prompt_id, prompt_results)
            all_prompt_results.append(prompt_results)

        # Aggregate across prompts: mean / min / max
        mean_results: dict[str, float] = OrderedDict()
        min_results: dict[str, float] = OrderedDict()
        max_results: dict[str, float] = OrderedDict()
        for metric in self.config.metrics:
            for k in self.config.topk:
                key = f"{metric}@{k}"
                vals = [r[key] for r in all_prompt_results]
                mean_results[key] = sum(vals) / len(vals)
                min_results[key] = min(vals)
                max_results[key] = max(vals)

        for key in mean_results:
            self._log("Mean %s: %.4f  (min=%.4f, max=%.4f)", key, mean_results[key], min_results[key], max_results[key])

        return {"mean": mean_results, "min": min_results, "max": max_results, "per_prompt": all_prompt_results}

    def _parse_test_prompt_ids(self) -> list[int]:
        from recbole3.model.lcrec.prompts import seqrec_prompts

        tpi = self.config.test_prompt_ids
        if tpi == "all":
            return list(range(len(seqrec_prompts)))

        ids = [int(x.strip()) for x in tpi.split(",")]
        # Validate against seqrec_prompts
        for pid in ids:
            if pid < 0 or pid >= len(seqrec_prompts):
                raise ValueError(
                    f"prompt_id {pid} out of range [0, {len(seqrec_prompts) - 1}]"
                )
        return ids

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_metrics(
        preds: torch.Tensor,
        labels: torch.Tensor,
        eos_token_id: int,
        metrics: tuple[str, ...] = ("recall", "ndcg"),
        topk: tuple[int, ...] = (5, 10, 20),
    ) -> dict[str, torch.Tensor]:
        """Calculate Recall@K and NDCG@K for a batch.

        Args:
            preds: (B, maxk, n_digit) predicted token sequences.
            labels: (B, n_digit) ground truth token sequences.
            eos_token_id: EOS token ID for label truncation.
            metrics: Metric names to compute.
            topk: Top-K values to compute.

        Returns:
            Dict mapping metric name -> per-sample scores tensor.
        """
        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()
        B = preds_cpu.shape[0]
        maxk = preds_cpu.shape[1]
        n_digit = preds_cpu.shape[2]

        # Build position index: pos_index[i, j] = True if preds[i, j] matches labels[i]
        # Truncate labels to n_digit tokens (SID tokens only, stripping any trailing EOS)
        compare_labels = labels_cpu[:, :n_digit]  # (B, n_digit)
        # Vectorized: compare all (B, maxk, n_digit) predictions against (B, 1, n_digit) labels
        matches = (preds_cpu == compare_labels.unsqueeze(1)).all(dim=-1)  # (B, maxk)
        # Mark only the first matching position per sample
        pos_index = matches.cumsum(dim=1).eq(1) & matches  # (B, maxk)

        results: dict[str, torch.Tensor] = {}

        # Compute DCG weights once if not pre-computed
        dcg_weights = 1.0 / torch.log2(torch.arange(1, maxk + 1).float() + 1)

        for k in topk:
            if k > maxk:
                continue
            if "recall" in metrics:
                results[f"recall@{k}"] = pos_index[:, :k].sum(dim=1).float()
            if "ndcg" in metrics:
                results[f"ndcg@{k}"] = torch.where(
                    pos_index, dcg_weights.unsqueeze(0).expand(B, -1), torch.zeros_like(pos_index).float()
                )[:, :k].sum(dim=1)

        return results
