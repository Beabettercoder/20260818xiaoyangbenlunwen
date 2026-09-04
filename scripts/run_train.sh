#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATA_DIR [additional metatrain arguments...]" >&2
  exit 2
fi

DATA_DIR="$1"
shift
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" metatrain_StyleAdv_RN.py --data_dir "${DATA_DIR}" "$@"
