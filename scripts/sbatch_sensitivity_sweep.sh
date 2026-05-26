#!/bin/bash
#SBATCH --job-name=sensit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/sbatch_sensitivity_%j.log
#SBATCH --error=outputs/sbatch_sensitivity_%j.log

# =============================================================================
# Threshold + cost-budget sensitivity -- reproduces Tables 5 & 6 (Appendix D).
#
# Two independent sweeps, both on the deployed sklearn GBM head:
#
#   * RASER-2 (Table 5):
#       sweep threshold  theta in {0.10, 0.15, 0.20, 0.25, 0.30}
#       (deployed: theta = 0.20)
#
#   * RASER-3 (Table 6):
#       sweep cost-budget fraction in {0.33, 0.50, 0.60, 0.75, 1.00}
#       (deployed: fraction = 0.60; lambda is then derived from the
#        cost-budget rule on the training fold per (LLM, dataset) cell)
#
# Output:    console tables + outputs/sensitivity/summary.json
# Compute:   CPU only, ~2 min on 8 cores.
# Requires:  trace files already in outputs/sweep_*/ (Step 1 in README).
# =============================================================================

set -e

# Run from the RASER package root.
cd "$(dirname "$0")/.."

# Optional: activate your Python venv before running.
# source .venv/bin/activate

python -m src.eval.sensitivity_sweep 2>&1
