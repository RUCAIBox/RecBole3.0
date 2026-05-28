#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
TEMPLATE="qwen3"
CUDA_VISIBLE_DEVICES_VALUE="0"
USE_TORCHRUN=false

OUTPUT_ROOT="outputs/starec_cds_training"
DATASET_DIR="$OUTPUT_ROOT/export"
DATASET_NAME="starec_sft"
SFT_OUTPUT_DIR="$OUTPUT_ROOT/sft/full"
CONFIG_DIR="$OUTPUT_ROOT/configs"

CUTOFF_LEN=16384
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE="1.0e-5"
NUM_TRAIN_EPOCHS="3.0"
SAVE_STEPS=100
PREPROCESSING_WORKERS=8
DATALOADER_WORKERS=2
BF16=true

to_yaml_path() {
  printf '%s\n' "${1//\\//}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$DATASET_DIR/starec_sft.jsonl" ]]; then
  echo "Missing SFT data. Run examples/starec/export_cds_training_data.sh first." >&2
  exit 1
fi
if [[ ! -f "$DATASET_DIR/dataset_info.json" ]]; then
  echo "Missing LlamaFactory dataset_info.json. Run examples/starec/export_cds_training_data.sh first." >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR"

if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
fi
if [[ "$USE_TORCHRUN" == true ]]; then
  export FORCE_TORCHRUN=1
else
  unset FORCE_TORCHRUN || true
fi

SFT_CONFIG_PATH="$CONFIG_DIR/llamafactory_starec_cds_sft.yaml"
cat > "$SFT_CONFIG_PATH" <<EOF
model_name_or_path: $(to_yaml_path "$BASE_MODEL_PATH")
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full

dataset: $DATASET_NAME
dataset_dir: $(to_yaml_path "$DATASET_DIR")
template: $TEMPLATE
cutoff_len: $CUTOFF_LEN
overwrite_cache: true
preprocessing_num_workers: $PREPROCESSING_WORKERS
dataloader_num_workers: $DATALOADER_WORKERS

output_dir: $(to_yaml_path "$SFT_OUTPUT_DIR")
logging_steps: 1
save_steps: $SAVE_STEPS
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE
gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS
learning_rate: $LEARNING_RATE
num_train_epochs: $NUM_TRAIN_EPOCHS
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: $BF16
ddp_timeout: 180000000
resume_from_checkpoint: null
EOF

echo "[starec:sft] config: $SFT_CONFIG_PATH"
echo "[starec:sft] training full model: $SFT_OUTPUT_DIR"
llamafactory-cli train "$SFT_CONFIG_PATH"

echo "[starec:sft] done"
