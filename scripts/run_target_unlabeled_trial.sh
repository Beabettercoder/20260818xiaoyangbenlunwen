#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_target_unlabeled_trial.sh DATA_ROOT [TARGET] [SHOT] [GPU_ID]" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=$(cd "$1" 2>/dev/null && pwd || true)
TARGET=${2:-AID}
SHOT=${3:-5}
GPU_ID=${4:-4}

PYTHON_BIN=${PYTHON_BIN:-python3}
SAVE_DIR=${SAVE_DIR:-"$REPO_ROOT/output"}
LOG_ROOT=${LOG_ROOT:-"$REPO_ROOT/logs/target_unlabeled_trial"}
WARMUP_NAME=${WARMUP_NAME:-baseline}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-1}
TRAIN_EPISODES=${TRAIN_EPISODES:-10}
TEST_EPISODES=${TEST_EPISODES:-100}
TARGET_SSL_WEIGHT=${TARGET_SSL_WEIGHT:-0.25}
TRAIN_N_QUERY=${TRAIN_N_QUERY:-5}
TEST_N_QUERY=${TEST_N_QUERY:-15}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-32}

if [[ -z "$DATA_ROOT" || ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: $1" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/$TARGET/unlabeled.json" || ! -f "$DATA_ROOT/$TARGET/novel.json" ]]; then
  echo "[ERROR] $TARGET requires unlabeled.json and novel.json under $DATA_ROOT/$TARGET" >&2
  exit 2
fi
if [[ "$SHOT" != "1" && "$SHOT" != "5" ]]; then
  echo "[ERROR] SHOT must be 1 or 5" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$SAVE_DIR/checkpoints"
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git)
SUMMARY_FILE="$LOG_ROOT/summary_${TARGET}_${SHOT}shot_${RUN_TAG}.txt"

echo "[Protocol] labeled NWPU base+val plus $TARGET unlabeled; evaluate $TARGET novel"
echo "[Protocol] two arms share the same source split and checkpoint rule"
echo "[Config] epochs=$EPOCHS train_episodes=$TRAIN_EPISODES test_episodes=$TEST_EPISODES gpu=$GPU_ID"

for ARM in pairwise_control target_feature; do
  SSL_WEIGHT=0
  if [[ "$ARM" == "target_feature" ]]; then
    SSL_WEIGHT=$TARGET_SSL_WEIGHT
  fi

  RUN_NAME="trial_${ARM}_NWPU_to_${TARGET}_${SHOT}shot_${RUN_TAG}"
  RUN_DIR="$SAVE_DIR/checkpoints/$RUN_NAME"
  TRAIN_LOG="$LOG_ROOT/${RUN_NAME}_train.log"
  TEST_LOG="$LOG_ROOT/${RUN_NAME}_test.log"

  if [[ -e "$RUN_DIR" ]]; then
    echo "[ERROR] refusing to overwrite $RUN_DIR" >&2
    exit 2
  fi

  echo "===== Train $ARM ====="
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$REPO_ROOT/metatrain_StyleAdv_RN.py" \
    --data_dir "$DATA_ROOT" \
    --source_dataset NWPU \
    --target_dataset "$TARGET" \
    --name "$RUN_NAME" \
    --save_dir "$SAVE_DIR" \
    --warmup "$WARMUP_NAME" \
    --require_warmup 1 \
    --n_shot "$SHOT" \
    --n_query "$TEST_N_QUERY" \
    --train_n_query "$TRAIN_N_QUERY" \
    --train_aug \
    --style_attack_mode progressive \
    --stop_epoch "$EPOCHS" \
    --train_episodes "$TRAIN_EPISODES" \
    --val_episodes 1 \
    --do_bscdfsl 0 \
    --target_ssl_mode feature_consistency \
    --target_ssl_weight "$SSL_WEIGHT" \
    --target_ssl_ramp_epochs 60 \
    --semantic_anchor 0 \
    --semantic_drift_control 0 \
    --text_guide_epsilon 0 \
    --text_guide_gradient 0 \
    --skip_source_test 1 \
    --train_num_workers 2 \
    --eval_num_workers 4 \
    --feature_batch_size "$FEATURE_BATCH_SIZE" \
    --pin_memory 1 \
    --persistent_workers 1 \
    2>&1 | tee "$TRAIN_LOG"

  if [[ ! -f "$RUN_DIR/best_model.tar" ]]; then
    echo "[ERROR] training ended without best_model.tar: $RUN_DIR" >&2
    exit 1
  fi

  echo "===== Test $ARM ====="
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$REPO_ROOT/test_function_bscdfsl_benchmark.py" \
    --data_dir "$DATA_ROOT" \
    --source_dataset NWPU \
    --target_datasets "$TARGET" \
    --name "$RUN_NAME" \
    --save_dir "$SAVE_DIR" \
    --n_shot "$SHOT" \
    --n_query "$TEST_N_QUERY" \
    --n_episodes_test "$TEST_EPISODES" \
    --feature_batch_size "$FEATURE_BATCH_SIZE" \
    --eval_num_workers 4 \
    --text_guide_epsilon 0 \
    --text_guide_gradient 0 \
    2>&1 | tee "$TEST_LOG"

  {
    echo "run=$RUN_NAME commit=$COMMIT arm=$ARM source=NWPU target=$TARGET shot=$SHOT"
    grep -E 'test iterations.*Acc =' "$RUN_DIR/acc_bscdfsl.txt" || true
    grep -E 'Epoch [0-9]+ \| Mean Loss' "$TRAIN_LOG" | tail -1 || true
    echo
  } | tee -a "$SUMMARY_FILE"
done

echo "[DONE] $SUMMARY_FILE"
