#!/bin/bash
#SBATCH --job-name=score_backfill
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/sbatch_score_backfill_%j.log
#SBATCH --error=outputs/sbatch_score_backfill_%j.log

# =============================================================================
# Rebuild dense cosine scores in STOP / one-shot RAG trace files (one-off).
#
# Used by both RASER-2 and RASER-3 indirectly: the `score_gap` and `score_top1`
# features are computed from these scores. If you re-run the dense retriever
# with a new embedding model after the trace files were produced, the stored
# `score` field becomes stale and needs to be rebuilt.
#
# The script uses Nomic-Embed-Text-v1.5 to recompute cosine(question, chunk)
# for each chunk in every (LLM, dataset) STOP trace file, writes the new
# score back in-place, and keeps a backup at *.bak.pre_score_backfill.
#
# Output:    in-place update of every outputs/sweep_*/*naive_*.jsonl
# Compute:   1 GPU; ~30 min for 18 cells.
# Requires:  trace files in outputs/sweep_*/ from Step 1.
# When you need it: only if you change the retriever after generating traces;
#                   skip otherwise.
# =============================================================================

set -e

# Run from the RASER package root.
cd "$(dirname "$0")/.."

# Optional: activate your Python venv before running.
# source .venv/bin/activate

python -m src.eval.backfill_dense_scores 2>&1

echo "---"
echo "Backfill complete. Verifying one file:"
python -c "
import json, statistics as st
fn = 'outputs/traces_holdout/musique/naive_bm25_dense_nomic_holdout_musique_traces.jsonl'
gaps, t1s = [], []
for line in open(fn):
    r = json.loads(line)
    ev = r.get('text_evidence') or []
    s = [float(c.get('score') or 0) for c in ev[:5]]
    if len(s) >= 5:
        gaps.append(s[0] - s[4]); t1s.append(s[0])
print(f'   score_top1: mean={st.mean(t1s):.3f} stdev={st.stdev(t1s):.3f} '
      f'min={min(t1s):.3f} max={max(t1s):.3f}')
print(f'   score_gap : mean={st.mean(gaps):.3f} stdev={st.stdev(gaps):.3f} '
      f'min={min(gaps):.3f} max={max(gaps):.3f}')
"
