#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv cache clean

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  uv pip install --system -r "${ROOT_DIR}/requirements.txt"
fi
