#!/bin/bash
#SBATCH --job-name=router_abl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/sbatch_router_model_ablation_%j.log
#SBATCH --error=outputs/sbatch_router_model_ablation_%j.log

# =============================================================================
# Router head ablation -- reproduces Table 12 in the paper (Appendix C).
#
# Runs RASER-2 (binary classifier) and RASER-3 (three regressors) with six
# alternative head model families, keeping everything else fixed (six features,
# same 5-fold CV per (LLM, dataset) cell, deployed hyperparameters).
#
# RASER-2 heads compared: sklearn GBM (deployed), Logistic Regression,
#                         MLP-32, XGBoost, LightGBM, CatBoost.
# RASER-3 heads compared: sklearn GBM (deployed), Ridge,
#                         MLP-32, XGBoost, LightGBM, CatBoost.
#
# Output:    console table + outputs/router_model_ablation/summary.json
# Compute:   CPU only, ~2 min on 8 cores.
# Requires:  trace files already in outputs/sweep_*/ (Step 1 in README).
# =============================================================================

set -e

# Run from the RASER package root.
cd "$(dirname "$0")/.."

# Optional: activate your Python venv before running.
# source .venv/bin/activate

python -m src.eval.router_model_ablation 2>&1
