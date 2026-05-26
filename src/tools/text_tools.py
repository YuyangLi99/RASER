"""Text retrieval tools: BM25, dense, and hybrid (RRF) retrieval over chunked passages."""

import json
import os
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional


class TextRetriever:
    """Text chunk retriever supporting BM25 / dense / hybrid (RRF) modes.

    mode:
        "bm25"   - BM25 only (default, no encoder needed)
        "dense"  - dense only (requires encoder + pre-encoded index)
        "hybrid" - BM25 + dense fused via Reciprocal Rank Fusion (k=60)
    """

    def __init__(self, processed_dir: str, mode: str = "bm25",
                 encoder=None, encoder_name: Optional[str] = None,
                 rrf_k: int = 60):
        assert mode in ("bm25", "dense", "hybrid")
        self.mode = mode
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_texts: List[str] = []
        self.bm25 = None
        self.dense_emb: Optional[np.ndarray] = None  # (N, D) L2-normalized
        self._load(processed_dir, encoder_name)

    def _load(self, processed_dir: str, encoder_name: Optional[str]):
        graphed_file = os.path.join(processed_dir, "graphed.jsonl")
        with open(graphed_file) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                for chunk in rec.get("text_chunks", []):
                    self.chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "question_id": qid,
                        "title": chunk.get("title", ""),
                        "text": chunk["text"],
                        "passage_idx": chunk.get("passage_idx", -1),
                        "entities": chunk.get("entities", []),
                    })
                    self.chunk_texts.append(chunk["text"])

        tokenized = [doc.lower().split() for doc in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        if self.mode in ("dense", "hybrid"):
            if encoder_name is None:
                raise ValueError("encoder_name required for dense/hybrid mode")
            emb_path = os.path.join(processed_dir, f"dense_{encoder_name}_chunks.npy")
            meta_path = os.path.join(processed_dir, f"dense_{encoder_name}_chunks_meta.json")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Missing dense index {emb_path}. Run build_dense_index.py first.")
            self.dense_emb = np.load(emb_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert len(meta) == len(self.chunks), \
                f"Index/data mismatch: {len(meta)} vs {len(self.chunks)}"
            for i, m in enumerate(meta):
                assert m["chunk_id"] == self.chunks[i]["chunk_id"], \
                    f"chunk_id mismatch at {i}"

        print(f"TextRetriever[{self.mode}]: {len(self.chunks)} chunks loaded")

    def _bm25_ranks(self, query_str: str, candidate_indices: List[int]) -> Dict[int, int]:
        """Return rank (1-indexed) of each candidate by BM25."""
        scores = self.bm25.get_scores(query_str.lower().split())
        cand_sorted = sorted(candidate_indices, key=lambda i: -scores[i])
        return {idx: r + 1 for r, idx in enumerate(cand_sorted)}, scores

    def _dense_ranks(self, query_str: str, candidate_indices: List[int]) -> Dict[int, int]:
        q_emb = self.encoder.encode_queries([query_str])[0]  # (D,)
        sims = self.dense_emb @ q_emb  # (N,)
        cand_sorted = sorted(candidate_indices, key=lambda i: -sims[i])
        return {idx: r + 1 for r, idx in enumerate(cand_sorted)}, sims

    def retrieve(self, query, top_k: int = 5, question_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve top-k text chunks for a query."""
        if isinstance(query, list):
            query = " ".join(str(q) for q in query)
        query = str(query)

        # Candidate pool: all chunks for this question (or all chunks if no qid filter)
        if question_id:
            candidates = [i for i, c in enumerate(self.chunks) if c["question_id"] == question_id]
        else:
            candidates = list(range(len(self.chunks)))

        if self.mode == "bm25":
            scores = self.bm25.get_scores(query.lower().split())
            scored = [(i, scores[i]) for i in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = []
            for idx, s in scored[:top_k]:
                chunk = self.chunks[idx].copy()
                chunk["bm25_score"] = float(s)
                results.append(chunk)
            return results

        if self.mode == "dense":
            q_emb = self.encoder.encode_queries([query])[0]
            sims = self.dense_emb @ q_emb
            scored = [(i, sims[i]) for i in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = []
            for idx, s in scored[:top_k]:
                chunk = self.chunks[idx].copy()
                chunk["dense_score"] = float(s)
                chunk["bm25_score"] = 0.0
                results.append(chunk)
            return results

        # hybrid: RRF fusion
        bm25_ranks, bm25_scores = self._bm25_ranks(query, candidates)
        dense_ranks, dense_sims = self._dense_ranks(query, candidates)
        rrf = {}
        for i in candidates:
            rrf[i] = 1.0 / (self.rrf_k + bm25_ranks[i]) + 1.0 / (self.rrf_k + dense_ranks[i])
        scored = sorted(rrf.items(), key=lambda x: -x[1])
        results = []
        for idx, s in scored[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = float(s)
            chunk["bm25_score"] = float(bm25_scores[idx])
            chunk["dense_score"] = float(dense_sims[idx])
            results.append(chunk)
        return results
