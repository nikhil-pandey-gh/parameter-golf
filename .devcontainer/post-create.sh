#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# Install global npm packages (tsgo = Go-based TypeScript 7 native compiler)
npm install -g wrangler @typescript/native-preview
npm install -g bun
npm install -g @github/copilot@prerelease
npm install -g @openai/codex@alpha

# Install Python dependencies via uv
uv pip install --system -r "${ROOT_DIR}/requirements.txt"

echo
echo "=== tool versions ==="
node --version
npm --version
python3 --version
uv --version
bun --version
tsgo --version
wrangler --version
copilot --version
codex --version
docker --version
cargo --version
gh --version
