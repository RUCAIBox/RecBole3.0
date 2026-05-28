# STARec Reproduction Example

This directory contains bash recipes for reproducing the STARec training and
evaluation flow with RecBole3.0, LlamaFactory, VeRL, and a vLLM-compatible
serving endpoint.

The scripts are intentionally dataset-specific. They target the Amazon 2014
retrieval dataset with `dataset.category=CDs_and_Vinyl`, matching the CD/music
product setting used by STARec-style experiments. Edit the variables at the top
of each script if you want to run another dataset or change the model sizes.

The default recipe uses newer model choices for a practical current run. Apart
from those model names, the disclosed paper settings are encoded directly where
possible: 1,000 teacher users, 1,000 heldout evaluation users, maximum history
length 40, SFT cutoff length 16,384, SFT 3 epochs, RL learning rate `1.0e-6`,
RL 1 epoch, GRPO batch size 64, KL coefficient `1.0e-3`, 8 rollouts, and RL
input/output lengths 4,096/16,384. The SFT script uses full fine-tuning rather
than LoRA. For a paper-faithful model setup, edit the script variables to use
DeepSeek-R1-Distill-Qwen-32B as the teacher and Qwen2.5-7B as the primary
SFT/RL base model, with the matching LlamaFactory template.

## Requirements

- `uv` and this RecBole3.0 repository.
- The RecBole STARec extra for teacher/evaluation LLM calls:

```bash
uv sync --extra starec
```

- LlamaFactory on `PATH` as `llamafactory-cli` for SFT.
- VeRL importable as `python -m verl.trainer.main_ppo` for RL.
- A teacher LLM endpoint compatible with the OpenAI SDK.
- A vLLM OpenAI-compatible endpoint for final student evaluation.

For committed examples, keep `API_KEY=""`. For a local run, either set it in
your shell or fill it directly in `run_cds_teacher_trace.sh`:

```bash
export STAREC_API_KEY="your-teacher-api-key"
```

The default SFT export uses `SFT_REASONING_MODE=think-tags`, so the teacher
endpoint must return `message.reasoning_content`. If you use a non-reasoning
teacher, set `SFT_REASONING_MODE="answer-only"` in
`export_cds_training_data.sh`.

## Pipeline

Run the scripts from the repository root, or directly by path; each script will
change into the repository root before running.

### 1. Select teacher and heldout users

```bash
examples/starec/prepare_cds_user_split.sh
```

This writes:

- `outputs/starec_cds_training/users/teacher_users.jsonl`
- `outputs/starec_cds_training/users/heldout_eval_users.jsonl`
- `outputs/starec_cds_training/users/user_split_manifest.json`

By default this samples 1,000 teacher users and 1,000 heldout evaluation users,
with `history_max_length=40`, `history_min_length=30`, and
`train_init_interactions=20`. The CDs scripts use `dataset.metadata_mode=fields`,
`model.feedback_score_field=overall`, and `model.feedback_positive_threshold=3`
so STARec can interpret Amazon ratings with the paper's `rating > 3` rule
without changing the dataset's `label` field.

### 2. Run the teacher trace

```bash
examples/starec/run_cds_teacher_trace.sh
```

This runs STARec on the teacher users and writes a train-only trace to:

```text
outputs/starec_cds_training/teacher_trace/teacher_trace.jsonl
```

The teacher trace script sets `model.temperature=1.0`, `model.top_p=1.0`, and
`model.item_text_template='{title}. Artist/brand: {brand}'`.

### 3. Export SFT and RL data

```bash
examples/starec/export_cds_training_data.sh
```

This writes:

- `outputs/starec_cds_training/export/starec_sft.jsonl`
- `outputs/starec_cds_training/export/dataset_info.json`
- `outputs/starec_cds_training/export/starec_verl_rl.jsonl`
- rejected-row JSONL files for auditing filters

SFT ranking samples are filtered by their own `target_rank <= 5`; SFT
init/reflection samples are filtered by the next ranking probe from the same
user. RL samples use ranking turns as prompts and are not filtered by the
teacher model's target rank by default.

### 4. Train the SFT model with LlamaFactory

```bash
examples/starec/train_cds_sft_llamafactory.sh
```

The default base model is `Qwen/Qwen3-4B-Instruct-2507`, and the default
LlamaFactory template is `qwen3`. The script uses the paper's primary 7B-style
SFT learning rate (`1.0e-5`), cutoff length 16,384, and 3 training epochs. It
full fine-tunes the model to:

```text
outputs/starec_cds_training/sft/full
```

### 5. Train the RL model with VeRL

```bash
examples/starec/train_cds_rl_verl.sh
```

The script converts the exported RL JSONL to parquet and calls VeRL with
`recbole3.model.starec.reward.compute_score` as the custom reward function. The
VeRL call uses GRPO with train batch size 64, 8 rollouts, learning rate
`1.0e-6`, 1 epoch, KL coefficient `1.0e-3`, and maximum prompt/response lengths
4,096/16,384. The default script is written for 8 visible GPUs.

### 6. Serve and evaluate the student

Start a vLLM OpenAI-compatible server for your SFT/RL model, then run:

```bash
examples/starec/run_cds_vllm_eval.sh
```

If your serving URL or model name differs, edit `API_BASE_URL` and
`API_MODEL_NAME` at the top of `run_cds_vllm_eval.sh`.

The final evaluation uses heldout users from:

```text
outputs/starec_cds_training/users/heldout_eval_users.jsonl
```

and writes results under:

```text
outputs/starec_cds_training/vllm_eval
```

The vLLM evaluation script sets `model.temperature=0.2`.

## Editing

The scripts are meant to be edited directly. For example, change
`TEACHER_USER_COUNT` in `prepare_cds_user_split.sh`, `API_BATCH` in
`run_cds_teacher_trace.sh`, or `BASE_MODEL_PATH` in
`train_cds_sft_llamafactory.sh`. The default output root is
`outputs/starec_cds_training`.
