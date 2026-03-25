#!/usr/bin/env bash
set -euo pipefail

# Rerun commands for pairs that had fixes/remediations applied.
# This script only prints commands by default; copy-run manually as needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
COMPARE_SCRIPT="experiments/adapt_tsl_lib/compare_tsl_liulian.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: $PYTHON_BIN not found or not executable"
  exit 1
fi

cat <<'EOF'
# Early-stopping divergence reruns
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ECL_PatchTST --disable-es
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Exchange_PatchTST --disable-es
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ILI_PatchTST --disable-es
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Exchange_DLinear --disable-es
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ILI_DLinear --disable-es

# OOM continuity fixes (rerun with OOM fallback + disable-es)
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Traffic_TimesNet --disable-es --oom-fallback
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ILI_TimesNet --disable-es --oom-fallback
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Traffic_TimeXer --disable-es --oom-fallback

# Inventory-expansion reruns for default-backed TSL pairs
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Exchange_TimeMixer
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ILI_TimeMixer
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs Exchange_TimeXer
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ILI_TimeXer

# Static parity remediation fix
.venv/bin/python experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs ECL_Transformer --disable-es
EOF
