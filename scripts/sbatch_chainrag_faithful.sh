#!/bin/bash
#SBATCH --job-name=cr_faith
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/sbatch_chainrag_faithful_%j.log
#SBATCH --error=outputs/sbatch_chainrag_faithful_%j.log

# =============================================================================
# Faithful ChainRAG baseline (Liu et al., ACL 2025) -- BASELINE only.
# Used in the main results table (Table 2) as the "ChainRAG" column.
# Not part of RASER itself.
#
# What it does: builds a sentence-level graph (similarity / positional /
#               entity edges), expands seed sentences via multi-hop graph
#               walks, rewrites sub-questions with entity completion,
#               and synthesizes the final answer.
#
# Usage (one (reader, dataset) cell at a time):
#   sbatch --export=DATASET=musique,DATA_DIR=data/processed/musique,\
#          READER_TAG=gpt,LLM_MODEL=kit.gpt-oss-120b,N=300,\
#          OUT=outputs/chainrag_faithful/gpt-oss-120b/musique \
#     scripts/sbatch_chainrag_faithful.sh
#
# Output:    JSONL trace file (one record per question) under $OUT
# Compute:   1 GPU (for embeddings + spaCy NER); ~30-60 min per cell.
# Requires:  prepared corpus + holdout question list under $DATA_DIR.
# =============================================================================

set -e

# Run from the RASER package root.
cd "$(dirname "$0")/.."

DATASET=${DATASET:-musique}
DATA_DIR=${DATA_DIR:-data/processed/musique_demo}
READER_TAG=${READER_TAG:-gpt}
LLM_MODEL=${LLM_MODEL:-kit.gpt-oss-120b}
N=${N:-300}
OUT=${OUT:-outputs/chainrag_faithful/${READER_TAG}/${DATASET}}
QFILE=${QFILE:-${DATA_DIR}/holdout_qids.txt}

mkdir -p "$OUT"

# Optional: if reader is vLLM-served, discover endpoint URL
SLUG=$(echo "$LLM_MODEL" | tr '/.: ' '____')
URL_FILE="outputs/vllm_endpoints/${SLUG}.url"
if [[ "$LLM_MODEL" != kit.* ]] && [ -f "$URL_FILE" ]; then
  deadline=$(($(date +%s) + 600))
  while [ $(date +%s) -lt $deadline ]; do
    BASE_URL=$(cat "$URL_FILE" 2>/dev/null | tr -d '[:space:]')
    if [ -n "$BASE_URL" ] && curl -fsS --max-time 5 "${BASE_URL}/models" > /dev/null 2>&1; then
      export HAGRID_LLM_BASE_URL="$BASE_URL"
      export HAGRID_LLM_API_KEY="dummy"
      echo "  endpoint live: $BASE_URL"
      break
    fi
    echo "  waiting for endpoint..."
    sleep 15
  done
fi

echo "============================================"
echo "Faithful ChainRAG"
echo "Reader: $LLM_MODEL  Dataset: $DATASET  N: $N"
echo "OUT: $OUT"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "============================================"

python -u -m src.eval.baselines \
  --baseline chain_rag_faithful --data-dir "$DATA_DIR" --n "$N" --qid-file "$QFILE" \
  --retriever-mode dense --output-dir "$OUT" \
  --label-suffix "_chainrag_faithful_${READER_TAG}" \
  --llm-model "$LLM_MODEL"

echo ""
echo "============================================"
echo "Trace counts:"
for f in $OUT/*chain_rag_faithful*.jsonl; do
  printf "  %-90s %4d records\n" "$f" "$(wc -l < $f)"
done
echo "End: $(date)"
echo "============================================"
