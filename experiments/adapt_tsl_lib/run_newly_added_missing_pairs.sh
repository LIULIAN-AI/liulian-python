#!/usr/bin/env bash
set -euo pipefail

# Run all newly added missing pairs introduced in compare_tsl_liulian.py.
#
# Usage:
#   bash experiments/adapt_tsl_lib/run_newly_added_missing_pairs.sh
#   bash experiments/adapt_tsl_lib/run_newly_added_missing_pairs.sh --dry-run
#   bash experiments/adapt_tsl_lib/run_newly_added_missing_pairs.sh --remaining-only
#   bash experiments/adapt_tsl_lib/run_newly_added_missing_pairs.sh --dry-run --remaining-only
#
# Notes:
# - This script uses --pairs explicitly as requested.
# - Most pairs are expected to be skipped by design due to no canonical TSL
#   counterpart, task mismatch, or missing adapter/dataset integration.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: $PYTHON_BIN not found or not executable"
  echo "Please create/activate the project virtual environment first."
  exit 1
fi

PAIRS=(
  ETTh1_TimeLLM ETTh2_TimeLLM ETTm1_TimeLLM ETTm2_TimeLLM Weather_TimeLLM ECL_TimeLLM Traffic_TimeLLM Exchange_TimeLLM ILI_TimeLLM
  ETTh1_TimeMoE ETTh2_TimeMoE ETTm1_TimeMoE ETTm2_TimeMoE Weather_TimeMoE ECL_TimeMoE Traffic_TimeMoE Exchange_TimeMoE ILI_TimeMoE
  ETTh1_ETSformer ETTh2_ETSformer ETTm1_ETSformer ETTm2_ETSformer Weather_ETSformer ECL_ETSformer Traffic_ETSformer Exchange_ETSformer ILI_ETSformer
  ETTh1_Stationary ETTh2_Stationary ETTm1_Stationary ETTm2_Stationary Weather_Stationary ECL_Stationary Traffic_Stationary Exchange_Stationary ILI_Stationary
  Solar_PatchTST Solar_DLinear
  PEMS03_PatchTST PEMS03_DLinear PEMS04_PatchTST PEMS04_DLinear PEMS07_PatchTST PEMS07_DLinear PEMS08_PatchTST PEMS08_DLinear
  CovidDeaths_PatchTST CovidDeaths_DLinear NYCTaxi_PatchTST NYCTaxi_DLinear NN5_PatchTST NN5_DLinear FREDMD_PatchTST FREDMD_DLinear
)

echo "Running ${#PAIRS[@]} newly added pairs..."
exec "$PYTHON_BIN" experiments/adapt_tsl_lib/compare_tsl_liulian.py --pairs "${PAIRS[@]}" "$@"
