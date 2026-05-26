"""Graph retrieval tools: expand subgraph, find bridges, linearize triples.

Supports BM25 / dense / hybrid (RRF) modes for node retrieval.
"""

import json
import os
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional


class GraphRetriever:
    """Per-question graph retrieval supporting BM25 / dense / hybrid modes."""

    def __init__(self, processed_dir: str, mode: str = "bm25",
                 encoder=None, encoder_name: Optional[str] = None,
                 rrf_k: int = 60):
        assert mode in ("bm25", "dense", "hybrid")
        self.mode = mode
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.graphs = {}       # question_id -> networkx.Graph
        self.node_data = {}    # question_id -> {node_id: node_dict}
        self.all_nodes = []    # flat list
        self.all_node_texts = []
        self.bm25 = None
        self.dense_emb: Optional[np.ndarray] = None
        # Per-question candidate index lists for fast filtering
        self.qid_to_indices: Dict[str, List[int]] = {}
        self._load(processed_dir, encoder_name)

    def _load(self, processed_dir: str, encoder_name: Optional[str]):
        graphed_file = os.path.join(processed_dir, "graphed.jsonl")
        with open(graphed_file) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                graph_data = rec.get("graph", {})
                if not graph_data:
                    continue

                G = nx.Graph()
                node_map = {}
                for node in graph_data.get("nodes", []):
                    nid = node["node_id"]
                    G.add_node(nid, **node)
                    node_map[nid] = node
                    linearized = self._linearize_node(node)
                    self.qid_to_indices.setdefault(qid, []).append(len(self.all_nodes))
                    self.all_nodes.append({"question_id": qid, "node": node, "linearized": linearized})
                    self.all_node_texts.append(linearized)

                for edge in graph_data.get("edges", []):
                    G.add_edge(edge["source"], edge["target"], **edge)

                self.graphs[qid] = G
                self.node_data[qid] = node_map

        tokenized = [t.lower().split() for t in self.all_node_texts]
        self.bm25 = BM25Okapi(tokenized)

        if self.mode in ("dense", "hybrid"):
            if encoder_name is None:
                raise ValueError("encoder_name required for dense/hybrid mode")
            emb_path = os.path.join(processed_dir, f"dense_{encoder_name}_nodes.npy")
            meta_path = os.path.join(processed_dir, f"dense_{encoder_name}_nodes_meta.json")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Missing dense node index {emb_path}. Run build_dense_index.py first.")
            self.dense_emb = np.load(emb_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert len(meta) == len(self.all_nodes), \
                f"Index/data mismatch: {len(meta)} vs {len(self.all_nodes)}"
            for i, m in enumerate(meta):
                assert m["node_id"] == self.all_nodes[i]["node"]["node_id"], \
                    f"node_id mismatch at {i}"

        print(f"GraphRetriever[{self.mode}]: {len(self.graphs)} graphs, {len(self.all_nodes)} nodes")

    def _linearize_node(self, node: dict) -> str:
        """Convert a graph node to natural language for retrieval and LLM consumption."""
        label = node.get("label", "")
        ntype = node.get("node_type", "")
        prov = node.get("provenance", "")
        if ntype == "entity":
            return f"Entity: {label}"
        elif ntype == "passage":
            return f"Passage titled '{label}': {prov[:200]}" if prov else f"Passage: {label}"
        elif ntype == "sentence":
            return f"Supporting sentence from '{label}': {prov}" if prov else f"Sentence: {label}"
        return f"{label} {prov}"

    def expand_cheap(self, query, question_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Cheap expansion: BM25/dense/hybrid search over graph nodes for this question."""
        if isinstance(query, list):
            query = " ".join(str(q) for q in query)
        query = str(query)
        candidates = self.qid_to_indices.get(question_id, [])

        if self.mode == "bm25":
            scores = self.bm25.get_scores(query.lower().split())
            scored = [(i, scores[i]) for i in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [(i, {"bm25_score": float(s)}) for i, s in scored[:top_k]]

        elif self.mode == "dense":
            q_emb = self.encoder.encode_queries([query])[0]
            sims = self.dense_emb @ q_emb
            scored = [(i, sims[i]) for i in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [(i, {"dense_score": float(s)}) for i, s in scored[:top_k]]

        else:  # hybrid (RRF)
            bm25_scores = self.bm25.get_scores(query.lower().split())
            q_emb = self.encoder.encode_queries([query])[0]
            sims = self.dense_emb @ q_emb
            bm25_sorted = sorted(candidates, key=lambda i: -bm25_scores[i])
            dense_sorted = sorted(candidates, key=lambda i: -sims[i])
            bm25_rank = {idx: r + 1 for r, idx in enumerate(bm25_sorted)}
            dense_rank = {idx: r + 1 for r, idx in enumerate(dense_sorted)}
            rrf = {i: 1.0 / (self.rrf_k + bm25_rank[i]) + 1.0 / (self.rrf_k + dense_rank[i])
                   for i in candidates}
            scored = sorted(rrf.items(), key=lambda x: -x[1])
            top = [(i, {"rrf_score": float(s),
                        "bm25_score": float(bm25_scores[i]),
                        "dense_score": float(sims[i])}) for i, s in scored[:top_k]]

        results = []
        for idx, score_dict in top:
            entry = self.all_nodes[idx].copy()
            entry.update(score_dict)
            # Default bm25_score for downstream code that expects it
            if "bm25_score" not in entry:
                entry["bm25_score"] = 0.0
            nid = entry["node"]["node_id"]
            G = self.graphs.get(question_id)
            if G and G.has_node(nid):
                neighbors = list(G.neighbors(nid))[:5]
                entry["neighbors"] = [
                    self.node_data[question_id][n]
                    for n in neighbors
                    if n in self.node_data.get(question_id, {})
                ]
            else:
                entry["neighbors"] = []
            results.append(entry)
        return results

    def expand_beam(self, query, question_id: str, top_k: int = 5, beam_width: int = 3) -> List[Dict[str, Any]]:
        """Beam expansion: BM25 seed + multi-hop neighbor expansion."""
        # Start with cheap expansion
        seeds = self.expand_cheap(query, question_id, top_k=beam_width)

        G = self.graphs.get(question_id)
        if not G:
            return seeds

        # Expand 2 hops from seed nodes
        visited = set()
        expanded_nodes = []
        for seed in seeds:
            nid = seed["node"]["node_id"]
            visited.add(nid)
            expanded_nodes.append(seed)

            # 1-hop
            for n1 in list(G.neighbors(nid))[:beam_width]:
                if n1 not in visited:
                    visited.add(n1)
                    node_info = self.node_data[question_id].get(n1, {})
                    if node_info:
                        expanded_nodes.append({
                            "question_id": question_id,
                            "node": node_info,
                            "linearized": self._linearize_node(node_info),
                            "bm25_score": 0.0,
                            "neighbors": [],
                            "hop": 1,
                        })
                    # 2-hop
                    for n2 in list(G.neighbors(n1))[:2]:
                        if n2 not in visited:
                            visited.add(n2)
                            node_info2 = self.node_data[question_id].get(n2, {})
                            if node_info2:
                                expanded_nodes.append({
                                    "question_id": question_id,
                                    "node": node_info2,
                                    "linearized": self._linearize_node(node_info2),
                                    "bm25_score": 0.0,
                                    "neighbors": [],
                                    "hop": 2,
                                })

        # Re-rank all expanded nodes by BM25 relevance to query
        for entry in expanded_nodes:
            text = entry["linearized"]
            tokens = text.lower().split()
            query_tokens = query.lower().split()
            # Simple overlap score
            overlap = len(set(tokens) & set(query_tokens))
            entry["relevance_score"] = entry.get("bm25_score", 0) + overlap * 0.5

        expanded_nodes.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return expanded_nodes[:top_k]

    def get_subgraph_triples(self, question_id: str, node_ids: List[str]) -> List[str]:
        """Linearize edges between given nodes as natural language triples."""
        G = self.graphs.get(question_id)
        if not G:
            return []

        triples = []
        node_set = set(node_ids)
        for u, v, data in G.edges(data=True):
            if u in node_set or v in node_set:
                u_label = self.node_data[question_id].get(u, {}).get("label", u)
                v_label = self.node_data[question_id].get(v, {}).get("label", v)
                edge_type = data.get("edge_type", "related_to")
                triples.append(f"{u_label} --[{edge_type}]--> {v_label}")

        return triples[:20]
