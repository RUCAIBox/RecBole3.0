#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/starec_cds_training"
ACTOR_MODEL_PATH="$OUTPUT_ROOT/sft/full"
RL_JSONL_PATH="$OUTPUT_ROOT/export/starec_verl_rl.jsonl"
RL_PARQUET_PATH="$OUTPUT_ROOT/rl/starec_verl_rl.parquet"
REWARD_PATH="src/recbole3/model/starec/reward.py"
OUTPUT_DIR="$OUTPUT_ROOT/rl/checkpoints"
PROJECT_NAME="starec"
EXPERIMENT_NAME="starec-cds-rl"
CUDA_VISIBLE_DEVICES_VALUE="0,1,2,3,4,5,6,7"
PYTHON_BIN="python"

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=16384
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_N=8
LEARNING_RATE="1e-6"
TOTAL_EPOCHS=1
KL_LOSS_COEF="1.0e-3"
SAVE_FREQ=5
TEST_FREQ=5
GPUS_PER_NODE=8
TENSOR_MODEL_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.60

to_hydra_path() {
  printf '%s\n' "${1//\\//}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -e "$ACTOR_MODEL_PATH" ]]; then
  echo "Missing SFT-start model: $ACTOR_MODEL_PATH. Run examples/starec/train_cds_sft_llamafactory.sh first, or set ACTOR_MODEL_PATH." >&2
  exit 1
fi
if [[ ! -f "$RL_JSONL_PATH" ]]; then
  echo "Missing RL data: $RL_JSONL_PATH. Run examples/starec/export_cds_training_data.sh first." >&2
  exit 1
fi
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "Missing reward function: $REWARD_PATH." >&2
  exit 1
fi

if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$RL_PARQUET_PATH")"
RL_JSONL_ABS="$(cd -- "$(dirname -- "$RL_JSONL_PATH")" && pwd)/$(basename -- "$RL_JSONL_PATH")"
if [[ "$RL_PARQUET_PATH" = /* ]]; then
  RL_PARQUET_ABS="$RL_PARQUET_PATH"
else
  RL_PARQUET_ABS="$REPO_ROOT/$RL_PARQUET_PATH"
fi

echo "[starec:rl] converting JSONL to parquet: $RL_PARQUET_PATH"
RL_JSONL_ABS="$RL_JSONL_ABS" RL_PARQUET_ABS="$RL_PARQUET_ABS" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

src = Path(os.environ["RL_JSONL_ABS"])
dst = Path(os.environ["RL_PARQUET_ABS"])
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"No RL rows found in {src}")
dst.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(dst, index=False)
print(f"[starec:rl] wrote {len(rows)} rows to {dst}")
PY

TRAIN_FILE="$(to_hydra_path "$RL_PARQUET_PATH")"
ACTOR_MODEL="$(to_hydra_path "$ACTOR_MODEL_PATH")"
REWARD_FILE="$(to_hydra_path "$REWARD_PATH")"
LOCAL_DIR="$(to_hydra_path "$OUTPUT_DIR")"

echo "[starec:rl] actor model: $ACTOR_MODEL_PATH"
echo "[starec:rl] output dir: $OUTPUT_DIR"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TRAIN_FILE" \
  data.prompt_key=prompt \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$TRAIN_BATCH_SIZE" \
  data.shuffle=true \
  data.filter_overlong_prompts=false \
  data.truncation=left \
  actor_rollout_ref.model.path="$ACTOR_MODEL" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  reward_model.enable=false \
  custom_reward_function.path="$REWARD_FILE" \
  custom_reward_function.name=compute_score \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  "trainer.logger=['console']" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.default_local_dir="$LOCAL_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=true \
  trainer.critic_warmup=0

echo "[starec:rl] done"
