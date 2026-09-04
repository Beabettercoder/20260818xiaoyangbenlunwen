#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_submission_source_only.sh DATA_ROOT [MODE] [SHOTS] [GPU_ID]

Arguments:
  DATA_ROOT  Parent directory containing NWPU, AID, UCM and EuroSAT.
  MODE       styleadv_global (control) or semantic_global (current candidate).
             Default: semantic_global
  SHOTS      Comma- or space-separated shot counts. Default: "5 1"
  GPU_ID     Physical GPU index. Default: 4

Activate the project Python environment first, or set PYTHON_BIN explicitly.

Optional environment variables:
  PYTHON_BIN, SAVE_DIR, LOG_ROOT, WARMUP_NAME, TARGETS, RUN_TAG,
  EPOCHS, TRAIN_EPISODES, VAL_EPISODES, TEST_EPISODES,
  TRAIN_N_QUERY, TEST_N_QUERY, TRAIN_WORKERS, EVAL_WORKERS,
  FEATURE_BATCH_SIZE, DRY_RUN.

Examples:
  bash scripts/run_submission_source_only.sh /path/to/datasets semantic_global "5 1" 4
  DRY_RUN=1 bash scripts/run_submission_source_only.sh /path/to/datasets styleadv_global 5 4
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=$(cd "$1" 2>/dev/null && pwd || true)
MODE=${2:-semantic_global}
SHOTS_RAW=${3:-"5 1"}
GPU_ID=${4:-4}

PYTHON_BIN=${PYTHON_BIN:-python3}
SAVE_DIR=${SAVE_DIR:-"$REPO_ROOT/output"}
LOG_ROOT=${LOG_ROOT:-"$REPO_ROOT/logs/submission_source_only"}
WARMUP_NAME=${WARMUP_NAME:-baseline}
TARGETS=${TARGETS:-AID,UCM,EuroSAT}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

EPOCHS=${EPOCHS:-200}
TRAIN_EPISODES=${TRAIN_EPISODES:-100}
VAL_EPISODES=${VAL_EPISODES:-100}
TEST_EPISODES=${TEST_EPISODES:-1000}
TRAIN_N_QUERY=${TRAIN_N_QUERY:-5}
TEST_N_QUERY=${TEST_N_QUERY:-15}
TRAIN_WORKERS=${TRAIN_WORKERS:-2}
EVAL_WORKERS=${EVAL_WORKERS:-4}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-32}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$DATA_ROOT" || ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: $1" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN" >&2
  echo "Activate the environment or set PYTHON_BIN=/absolute/path/to/python3" >&2
  exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] GPU_ID must be a non-negative integer: $GPU_ID" >&2
  exit 2
fi

case "$MODE" in
  styleadv_global)
    SEMANTIC_ANCHOR=0
    SEMANTIC_DRIFT=0
    ;;
  semantic_global)
    SEMANTIC_ANCHOR=1
    SEMANTIC_DRIFT=1
    ;;
  *)
    echo "[ERROR] Unknown MODE: $MODE" >&2
    echo "Use styleadv_global or semantic_global." >&2
    exit 2
    ;;
esac

for split_file in \
  "$DATA_ROOT/NWPU/base.json" \
  "$DATA_ROOT/NWPU/val.json" \
  "$DATA_ROOT/AID/novel.json" \
  "$DATA_ROOT/UCM/novel.json" \
  "$DATA_ROOT/EuroSAT/novel.json"; do
  if [[ ! -f "$split_file" ]]; then
    echo "[ERROR] Required split file is missing: $split_file" >&2
    exit 2
  fi
done

WARMUP_DIR="$SAVE_DIR/checkpoints/$WARMUP_NAME"
if ! find "$WARMUP_DIR" -maxdepth 1 -type f -name '*.tar' -print -quit 2>/dev/null | grep -q .; then
  echo "[ERROR] Warm-up checkpoint not found under: $WARMUP_DIR" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$SAVE_DIR/checkpoints"
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git)
SUMMARY_FILE="$LOG_ROOT/summary_${MODE}_${RUN_TAG}.txt"
SHOTS_RAW=${SHOTS_RAW//,/ }
read -r -a SHOTS <<< "$SHOTS_RAW"

echo "[Protocol] source-only: labeled NWPU only; no target images enter training"
echo "[Protocol] one checkpoint per shot; the same checkpoint tests AID, UCM and EuroSAT"
echo "[Config] mode=$MODE gpu=$GPU_ID shots=${SHOTS[*]} commit=$COMMIT"
echo "[Config] data=$DATA_ROOT save=$SAVE_DIR"
echo "[Config] train=${EPOCHS} epochs x ${TRAIN_EPISODES} episodes; test=${TEST_EPISODES} episodes"
echo "[Truth] attack inner loss is global classifier CE in the current repository"

run_and_log() {
  local log_file=$1
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@" 2>&1 | tee "$log_file"
  fi
}

for SHOT in "${SHOTS[@]}"; do
  if [[ "$SHOT" != "1" && "$SHOT" != "5" ]]; then
    echo "[ERROR] Only 1-shot and 5-shot are supported: $SHOT" >&2
    exit 2
  fi

  RUN_NAME="submit_${MODE}_NWPU_${SHOT}shot_${RUN_TAG}"
  RUN_DIR="$SAVE_DIR/checkpoints/$RUN_NAME"
  TRAIN_LOG="$LOG_ROOT/${RUN_NAME}_train.log"
  TEST_LOG="$LOG_ROOT/${RUN_NAME}_test.log"

  if [[ -e "$RUN_DIR" ]]; then
    echo "[ERROR] Refusing to overwrite an existing run: $RUN_DIR" >&2
    exit 2
  fi

  echo
  echo "===== Train $MODE, NWPU, ${SHOT}-shot on physical GPU $GPU_ID ====="
  TRAIN_CMD=(
    env "CUDA_VISIBLE_DEVICES=$GPU_ID" "PYTHONUNBUFFERED=1"
    "$PYTHON_BIN" "$REPO_ROOT/metatrain_StyleAdv_RN.py"
    --data_dir "$DATA_ROOT"
    --source_dataset NWPU
    --name "$RUN_NAME"
    --save_dir "$SAVE_DIR"
    --warmup "$WARMUP_NAME"
    --require_warmup 1
    --n_shot "$SHOT"
    --n_query "$TEST_N_QUERY"
    --train_n_query "$TRAIN_N_QUERY"
    --train_aug
    --style_attack_mode progressive
    --stop_epoch "$EPOCHS"
    --train_episodes "$TRAIN_EPISODES"
    --val_episodes "$VAL_EPISODES"
    --do_bscdfsl 0
    --target_ssl_weight 0
    --semantic_anchor "$SEMANTIC_ANCHOR"
    --semantic_drift_control "$SEMANTIC_DRIFT"
    --text_guide_epsilon 0
    --text_guide_gradient 0
    --skip_source_test 1
    --train_num_workers "$TRAIN_WORKERS"
    --eval_num_workers "$EVAL_WORKERS"
    --feature_batch_size "$FEATURE_BATCH_SIZE"
    --pin_memory 1
    --persistent_workers 1
  )
  run_and_log "$TRAIN_LOG" "${TRAIN_CMD[@]}"

  if [[ "$DRY_RUN" != "1" && ! -f "$RUN_DIR/best_model.tar" ]]; then
    echo "[ERROR] Training ended without best_model.tar: $RUN_DIR" >&2
    exit 1
  fi

  echo
  echo "===== Test the same ${SHOT}-shot checkpoint on $TARGETS ====="
  TEST_CMD=(
    env "CUDA_VISIBLE_DEVICES=$GPU_ID" "PYTHONUNBUFFERED=1"
    "$PYTHON_BIN" "$REPO_ROOT/test_function_bscdfsl_benchmark.py"
    --data_dir "$DATA_ROOT"
    --source_dataset NWPU
    --target_datasets "$TARGETS"
    --name "$RUN_NAME"
    --save_dir "$SAVE_DIR"
    --n_shot "$SHOT"
    --n_query "$TEST_N_QUERY"
    --n_episodes_test "$TEST_EPISODES"
    --feature_batch_size "$FEATURE_BATCH_SIZE"
    --eval_num_workers "$EVAL_WORKERS"
    --text_guide_epsilon 0
    --text_guide_gradient 0
  )
  run_and_log "$TEST_LOG" "${TEST_CMD[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  RESULT_FILE="$RUN_DIR/acc_bscdfsl.txt"
  if [[ ! -f "$RESULT_FILE" ]] || ! grep -q 'Acc =' "$RESULT_FILE"; then
    echo "[ERROR] Evaluation did not produce complete accuracy output: $RESULT_FILE" >&2
    exit 1
  fi

  {
    echo "run=$RUN_NAME commit=$COMMIT mode=$MODE source=NWPU shot=$SHOT"
    grep 'Acc =' "$RESULT_FILE"
    echo
  } | tee -a "$SUMMARY_FILE"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[PASS] Command construction completed; no training was started."
else
  echo "[PASS] All requested source-only runs completed."
  echo "[PASS] Summary: $SUMMARY_FILE"
fi
