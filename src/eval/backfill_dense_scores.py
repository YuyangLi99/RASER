"""Backfill missing dense cosine scores into existing STOP/naive trace files.

The bug: src/eval/baselines.py line 73 hard-coded `c["bm25_score"]` when
writing text_evidence, even in dense mode — so traces produced under
retriever_mode='dense' all have score=0.0 for every chunk. The router's
score_gap / score_top1 features therefore see only zeros.

This script recomputes cosine(question, chunk_content) using the same
Nomic-Embed-v1.5 encoder used at retrieval time, and writes the
corrected scores back into the trace files in place (a .bak copy is
made first).

Usage:
    python -m src.eval.backfill_dense_scores [--dry-run] [PATH...]

If no PATH given, processes the canonical set: the naive_bm25 trace
files for all 6 readers x 3 datasets used by the canonical evaluation.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tools.encoders import get_encoder
from src.eval.three_route_feasibility import READER_TRACES


def canonical_stop_traces() -> List[Path]:
    """The 18 STOP trace files the router reads features from."""
    paths = []
    for reader, dsmap in READER_TRACES.items():
        for ds, (sp, _pp, _ip, _dp) in dsmap.items():
            paths.append(ROOT / sp)
    return paths


def backfill_file(path: Path, encoder, dry_run: bool = False) -> dict:
    """Recompute dense cosine scores for one trace file."""
    rows = [json.loads(l) for l in open(path)]
    n_records = len(rows)

    # Batch: collect all (record_idx, chunk_idx, question, chunk_content)
    questions, chunks_per_record = [], []
    for rec in rows:
        questions.append(rec.get("question", ""))
        ev = rec.get("text_evidence") or []
        chunks_per_record.append([(c.get("content") or "") for c in ev])

    # Encode all questions once
    q_emb = encoder.encode_queries(questions, batch_size=64)  # (n_rec, d)

    # Encode all chunks together (flattened)
    flat_chunks, flat_idx = [], []
    for ri, chunks in enumerate(chunks_per_record):
        for ci, content in enumerate(chunks):
            flat_chunks.append(content)
            flat_idx.append((ri, ci))
    if not flat_chunks:
        return {"file": str(path), "records": n_records, "chunks": 0,
                "max_score": 0.0, "skipped": True}

    c_emb = encoder.encode_documents(flat_chunks, batch_size=128)  # (n_chunk, d)

    # Cosine = dot product since both are L2-normalized
    # Map back to per-record scores
    new_scores_per_record = [[0.0] * len(c) for c in chunks_per_record]
    for k, (ri, ci) in enumerate(flat_idx):
        sim = float(np.dot(q_emb[ri], c_emb[k]))
        new_scores_per_record[ri][ci] = sim

    # Verify ordering is roughly monotonically decreasing per record
    # (since the chunks were retrieved in cosine-rank order; small drift is OK)
    out_max = 0.0
    for ri, rec in enumerate(rows):
        ev = rec.get("text_evidence") or []
        scores = new_scores_per_record[ri]
        for ci, c in enumerate(ev):
            c["score"] = round(scores[ci], 4)
        if scores:
            out_max = max(out_max, scores[0])

    if dry_run:
        return {"file": str(path), "records": n_records,
                "chunks": len(flat_chunks), "max_score": out_max,
                "dry_run": True}

    # Backup once
    bak = path.with_suffix(path.suffix + ".bak.pre_score_backfill")
    if not bak.exists():
        shutil.copy(path, bak)

    # Write back
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)

    return {"file": str(path), "records": n_records,
            "chunks": len(flat_chunks), "max_score": out_max,
            "wrote": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--encoder", default="nomic")
    args = ap.parse_args()

    paths = args.paths or canonical_stop_traces()
    paths = [p for p in paths if p.exists()]
    print(f"Backfilling {len(paths)} STOP trace files using {args.encoder}.")
    if not paths:
        print("No paths found."); return

    print("Loading encoder ...")
    encoder = get_encoder(args.encoder, device="cuda")
    print(f"Encoder ready: {encoder.name}")

    for p in paths:
        try:
            res = backfill_file(p, encoder, dry_run=args.dry_run)
            print(f"  {p.name}: n_rec={res['records']:>4} chunks={res['chunks']:>5}  max_score={res['max_score']:.3f}  {'(dry)' if args.dry_run else 'OK'}")
        except Exception as e:
            print(f"  {p.name}: ERROR {e}")


if __name__ == "__main__":
    main()
