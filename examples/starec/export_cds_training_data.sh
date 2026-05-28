#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/starec_cds_training"
TRACE_PATH="$OUTPUT_ROOT/teacher_trace/teacher_trace.jsonl"
EXPORT_DIR="$OUTPUT_ROOT/export"
SFT_PATH="$EXPORT_DIR/starec_sft.jsonl"
SFT_REJECTED_PATH="$EXPORT_DIR/starec_sft_rejected.jsonl"
DATASET_INFO_PATH="$EXPORT_DIR/dataset_info.json"
RL_PATH="$EXPORT_DIR/starec_verl_rl.jsonl"
RL_REJECTED_PATH="$EXPORT_DIR/starec_verl_rl_rejected.jsonl"
RANK_THRESHOLD=5
MAX_DESCRIPTION_WORDS=120
SFT_REASONING_MODE="think-tags"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$TRACE_PATH" ]]; then
  echo "Missing teacher trace: $TRACE_PATH. Run examples/starec/run_cds_teacher_trace.sh first." >&2
  exit 1
fi

uv run python -m recbole3.tools.starec_training export-sft \
  --trace-path "$TRACE_PATH" \
  --output-path "$SFT_PATH" \
  --rejected-path "$SFT_REJECTED_PATH" \
  --dataset-info-path "$DATASET_INFO_PATH" \
  --dataset-name starec_sft \
  --rank-threshold "$RANK_THRESHOLD" \
  --max-description-words "$MAX_DESCRIPTION_WORDS" \
  --sft-reasoning-mode "$SFT_REASONING_MODE"

uv run python -m recbole3.tools.starec_training export-rl \
  --trace-path "$TRACE_PATH" \
  --output-path "$RL_PATH" \
  --rejected-path "$RL_REJECTED_PATH" \
  --data-source starec_ranking
