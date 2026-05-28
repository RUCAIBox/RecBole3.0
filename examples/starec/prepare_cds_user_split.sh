#!/usr/bin/env bash
set -euo pipefail

TEACHER_USER_COUNT=1000
HELDOUT_EVAL_USER_COUNT=1000
USER_SPLIT_SEED=42
OUTPUT_ROOT="outputs/starec_cds_training"
OUTPUT_DIR="$OUTPUT_ROOT/users"

HISTORY_MAX_LENGTH=40
HISTORY_MIN_LENGTH=30
TRAIN_INIT_INTERACTIONS=20

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

uv run python -m recbole3.tools.starec_training split-users \
  --teacher-user-count "$TEACHER_USER_COUNT" \
  --heldout-eval-user-count "$HELDOUT_EVAL_USER_COUNT" \
  --seed "$USER_SPLIT_SEED" \
  --output-dir "$OUTPUT_DIR" \
  dataset=amazon2014_retrieval \
  dataset.category=CDs_and_Vinyl \
  dataset.metadata_mode=fields \
  model.history_max_length="$HISTORY_MAX_LENGTH" \
  model.history_min_length="$HISTORY_MIN_LENGTH" \
  model.train_init_interactions="$TRAIN_INIT_INTERACTIONS" \
  model.feedback_score_field=overall \
  model.feedback_positive_threshold=3 \
  runtime.device=cpu
