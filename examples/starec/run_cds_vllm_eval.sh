#!/usr/bin/env bash
set -euo pipefail

API_MODEL_NAME="starec-student"
API_BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"
RUNTIME_API_KEY_ENV_NAME="STAREC_STUDENT_RUNTIME_API_KEY"
API_BATCH=5
ASYNC_DISPATCH=true
ITEM_TEXT_TEMPLATE="'{title}. Artist/brand: {brand}'"

OUTPUT_ROOT="outputs/starec_cds_training"
HELDOUT_USERS_PATH="$OUTPUT_ROOT/users/heldout_eval_users.jsonl"
EVAL_OUTPUT_DIR="$OUTPUT_ROOT/vllm_eval"

HISTORY_MAX_LENGTH=40
HISTORY_MIN_LENGTH=30
TRAIN_INIT_INTERACTIONS=20

if [[ -n "${STAREC_STUDENT_API_KEY:-}" ]]; then
  API_KEY="$STAREC_STUDENT_API_KEY"
fi

export "${RUNTIME_API_KEY_ENV_NAME}=${API_KEY}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$HELDOUT_USERS_PATH" ]]; then
  echo "Missing heldout users: $HELDOUT_USERS_PATH. Run examples/starec/prepare_cds_user_split.sh first." >&2
  exit 1
fi

uv run --extra starec python -m recbole3.run \
  dataset=amazon2014_retrieval \
  model=starec \
  dataset.category=CDs_and_Vinyl \
  dataset.metadata_mode=fields \
  model.selected_user_count=-1 \
  model.selected_user_ids_path="$HELDOUT_USERS_PATH" \
  model.history_max_length="$HISTORY_MAX_LENGTH" \
  model.history_min_length="$HISTORY_MIN_LENGTH" \
  model.train_init_interactions="$TRAIN_INIT_INTERACTIONS" \
  model.item_text_template="$ITEM_TEXT_TEMPLATE" \
  model.feedback_score_field=overall \
  model.feedback_positive_threshold=3 \
  model.backend=openai \
  model.api_model_name="$API_MODEL_NAME" \
  model.api_base_url="$API_BASE_URL" \
  model.api_key_env="$RUNTIME_API_KEY_ENV_NAME" \
  model.temperature=0.2 \
  model.api_batch="$API_BATCH" \
  model.async_dispatch="$ASYNC_DISPATCH" \
  model.candidate_source=random \
  model.has_gt=true \
  model.refresh_candidate_cache=true \
  runtime.device=cpu \
  runtime.output_dir="$EVAL_OUTPUT_DIR"
