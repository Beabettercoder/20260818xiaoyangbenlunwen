#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end check for the current semantic-guided progressive method.
# This is intentionally short: it verifies data loading, CLIP/text modules,
# semantic anchor, drift control, one training epoch, checkpointing, and target
# evaluation without starting a formal experiment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_DIR="${1:-}"
GPU_ID="${2:-7}"
TARGET_DATASET="${3:-AID}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 DATA_DIR [GPU_ID] [TARGET_DATASET]" >&2
  echo "Example: $0 /server/datasets 0 AID" >&2
  exit 2
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] Dataset directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

SOURCE_DATASET="NWPU"
RUN_NAME="smoke_${SOURCE_DATASET}_to_${TARGET_DATASET}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${REPO_ROOT}/output/smoke_tests"
CHECKPOINT_DIR="${SAVE_DIR}/checkpoints/${RUN_NAME}"
ACC_FILE="${CHECKPOINT_DIR}/acc_${TARGET_DATASET}.txt"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
}

echo "[SmokeTest] repo=${REPO_ROOT}"
echo "[SmokeTest] source=${SOURCE_DATASET} target=${TARGET_DATASET} gpu=${GPU_ID}"
echo "[SmokeTest] data=${DATA_DIR}"

# Strict pairwise protocol: source uses labeled base/val; target supplies
# unlabeled data for SSL and novel data for the final episodic evaluation.
require_file "${DATA_DIR}/${SOURCE_DATASET}/base.json"
require_file "${DATA_DIR}/${SOURCE_DATASET}/val.json"
require_file "${DATA_DIR}/${SOURCE_DATASET}/novel.json"
require_file "${DATA_DIR}/${TARGET_DATASET}/unlabeled.json"
require_file "${DATA_DIR}/${TARGET_DATASET}/novel.json"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is not available in the selected environment")
print(f"[SmokeTest] torch={torch.__version__}")
print(f"[SmokeTest] torch_cuda={torch.version.cuda}")
print(f"[SmokeTest] visible_gpu={torch.cuda.get_device_name(0)}")
PY

cd "${REPO_ROOT}"

# One epoch, one training episode, and two evaluation episodes.  Workers are
# disabled so failures are reported in the foreground and do not get hidden by
# DataLoader worker processes.  The innovation path is enabled; the old CLIP
# gradient gate is explicitly disabled for attribution clarity.
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" metatrain_StyleAdv_RN.py \
  --data_dir "${DATA_DIR}" \
  --source_dataset "${SOURCE_DATASET}" \
  --target_dataset "${TARGET_DATASET}" \
  --name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --n_shot 5 \
  --style_attack_mode progressive \
  --semantic_anchor 1 \
  --semantic_drift_control 1 \
  --text_guide_epsilon 0 \
  --text_guide_gradient 0 \
  --target_ssl_weight 1.0 \
  --target_ssl_ramp_epochs 1 \
  --target_unlabeled_batch_size 8 \
  --stop_epoch 1 \
  --train_episodes 1 \
  --val_episodes 1 \
  --n_episodes_test 2 \
  --n_query 2 \
  --train_num_workers 0 \
  --eval_num_workers 0 \
  --persistent_workers 0 \
  --pin_memory 0 \
  --feature_batch_size 8 \
  --skip_source_test 1 \
  --do_bscdfsl 1

if [[ ! -f "${CHECKPOINT_DIR}/last_epoch.tar" && ! -f "${CHECKPOINT_DIR}/best_model.tar" ]]; then
  echo "[ERROR] Training returned but no checkpoint was created: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ACC_FILE}" ]]; then
  echo "[ERROR] Training returned but target result was not created: ${ACC_FILE}" >&2
  exit 1
fi

if ! grep -Eq 'Acc = [-+]?[0-9]+([.][0-9]+)?%' "${ACC_FILE}"; then
  echo "[ERROR] Target evaluation did not produce an accuracy line: ${ACC_FILE}" >&2
  cat "${ACC_FILE}" >&2
  exit 1
fi

echo "[PASS] Smoke test completed successfully."
echo "[PASS] checkpoint=${CHECKPOINT_DIR}"
echo "[PASS] target_result=${ACC_FILE}"
cat "${ACC_FILE}"
