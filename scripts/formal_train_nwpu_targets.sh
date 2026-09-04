#!/usr/bin/env bash
set -Eeuo pipefail

# Formal strict-pairwise experiment:
#   source: NWPU
#   targets: AID UCM EuroSAT (one independent run per target)
#
# Usage:
#   bash scripts/formal_train_nwpu_targets.sh DATA_ROOT [GPU_ID] [TARGETS]
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python
#   EPOCHS=200
#   TRAIN_EPISODES=100
#   TEST_EPISODES=1000
#   N_SHOT=5
#   SAVE_ROOT=/path/to/output/formal_nwpu
#   RUN_TAG=...
#   TRAIN_WORKERS=4
#   EVAL_WORKERS=4
#   SKIP_SOURCE_TEST=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_ROOT="${1:-}"
GPU_ID="${2:-7}"
TARGETS_TEXT="${3:-AID UCM EuroSAT}"

EPOCHS="${EPOCHS:-200}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
TEST_EPISODES="${TEST_EPISODES:-1000}"
N_SHOT="${N_SHOT:-5}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
SKIP_SOURCE_TEST="${SKIP_SOURCE_TEST:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/output/formal_nwpu/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/formal_nwpu/${RUN_TAG}}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: $0 DATA_ROOT [GPU_ID] [TARGETS]" >&2
  echo "Example: $0 /server/datasets 0 'AID UCM EuroSAT'" >&2
  exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] Dataset root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

# Accept either space-separated or comma-separated target names.
TARGETS_TEXT="${TARGETS_TEXT//,/ }"
read -r -a TARGETS <<< "${TARGETS_TEXT}"
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "[ERROR] No target dataset was supplied." >&2
  exit 2
fi

SOURCE_DATASET="NWPU"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 1
  fi
}

require_file "${DATA_ROOT}/${SOURCE_DATASET}/base.json"
require_file "${DATA_ROOT}/${SOURCE_DATASET}/val.json"
require_file "${DATA_ROOT}/${SOURCE_DATASET}/novel.json"

for target in "${TARGETS[@]}"; do
  if [[ "${target}" == "${SOURCE_DATASET}" ]]; then
    echo "[ERROR] Target must differ from source: ${target}" >&2
    exit 1
  fi
  require_file "${DATA_ROOT}/${target}/unlabeled.json"
  require_file "${DATA_ROOT}/${target}/novel.json"
done

mkdir -p "${SAVE_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

echo "[Formal] repo=${REPO_ROOT}"
echo "[Formal] source=${SOURCE_DATASET}"
echo "[Formal] targets=${TARGETS[*]}"
echo "[Formal] data=${DATA_ROOT}"
echo "[Formal] gpu=${GPU_ID}"
echo "[Formal] epochs=${EPOCHS}, train_episodes=${TRAIN_EPISODES}, test_episodes=${TEST_EPISODES}"
echo "[Formal] save_root=${SAVE_ROOT}"
echo "[Formal] log_root=${LOG_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit('[ERROR] CUDA is not available in the selected environment')
print(f'[Formal] torch={torch.__version__}')
print(f'[Formal] torch_cuda={torch.version.cuda}')
print(f'[Formal] visible_gpu={torch.cuda.get_device_name(0)}')
PY

for target in "${TARGETS[@]}"; do
  run_name="formal_${SOURCE_DATASET}_to_${target}_${N_SHOT}shot_${RUN_TAG}"
  run_save_dir="${SAVE_ROOT}"
  checkpoint_dir="${run_save_dir}/checkpoints/${run_name}"
  log_file="${LOG_ROOT}/${run_name}.log"
  acc_file="${checkpoint_dir}/acc_${target}.txt"

  echo
  echo "[Formal] ===== ${SOURCE_DATASET} -> ${target} ====="
  echo "[Formal] run=${run_name}"
  echo "[Formal] log=${log_file}"

  # The target is trained independently under the strict pairwise protocol.
  # The old CLIP epsilon/gate path stays off so the reported gain is from the
  # semantic anchor + progressive risk budget + drift feedback path.
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" metatrain_StyleAdv_RN.py \
    --data_dir "${DATA_ROOT}" \
    --source_dataset "${SOURCE_DATASET}" \
    --target_dataset "${target}" \
    --name "${run_name}" \
    --save_dir "${run_save_dir}" \
    --n_shot "${N_SHOT}" \
    --style_attack_mode progressive \
    --semantic_anchor 1 \
    --semantic_anchor_weight 1.0 \
    --semantic_drift_control 1 \
    --semantic_risk_beta 1.0 \
    --semantic_budget_temperature 1.0 \
    --semantic_drift_threshold 0.0 \
    --semantic_dual_step_size 0.1 \
    --semantic_lambda_max 10.0 \
    --text_guide_epsilon 0 \
    --text_guide_gradient 0 \
    --target_ssl_weight 1.0 \
    --target_ssl_ramp_epochs 60 \
    --target_unlabeled_batch_size 64 \
    --train_episodes "${TRAIN_EPISODES}" \
    --val_episodes 100 \
    --n_episodes_test "${TEST_EPISODES}" \
    --n_query 15 \
    --train_num_workers "${TRAIN_WORKERS}" \
    --eval_num_workers "${EVAL_WORKERS}" \
    --pin_memory 1 \
    --persistent_workers 1 \
    --feature_batch_size 64 \
    --skip_source_test "${SKIP_SOURCE_TEST}" \
    --do_bscdfsl 1 \
    --stop_epoch "${EPOCHS}" \
    2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] Training failed for ${target}; see ${log_file}" >&2
    exit "${status}"
  fi

  if [[ ! -f "${checkpoint_dir}/best_model.tar" && ! -f "${checkpoint_dir}/last_epoch.tar" ]]; then
    echo "[ERROR] No checkpoint was created for ${target}: ${checkpoint_dir}" >&2
    exit 1
  fi

  if [[ ! -f "${acc_file}" ]] || ! grep -Eq 'Acc = [-+]?[0-9]+([.][0-9]+)?%' "${acc_file}"; then
    echo "[ERROR] No valid target accuracy was produced for ${target}: ${acc_file}" >&2
    [[ -f "${acc_file}" ]] && cat "${acc_file}" >&2 || true
    exit 1
  fi

  echo "[Formal] RESULT ${target}"
  cat "${acc_file}"
done

echo
echo "[Formal] ALL TARGETS COMPLETED"
echo "[Formal] results=${SAVE_ROOT}"
echo "[Formal] logs=${LOG_ROOT}"
