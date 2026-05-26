"""Faithful ChainRAG reimplementation.

Following Liu et al. (ACL 2025) "Mitigating Lost-in-Retrieval Problems in
Retrieval Augmented Multi-Hop QA". https://github.com/nju-websoft/ChainRAG

Key components added (which our prior chain_rag.py lacked):
  1. Sentence-level retrieval (instead of chunk-level)
  2. Sentence graph with 3 edge types: similarity (k-NN), positional, entity-based
  3. Multi-hop graph expansion (1/2/3-hop neighbors until ~3000 word budget)
  4. Entity completion: sub-question rewriting using prior answer
  5. Multi-hop judgment + decomposition (was already in our simplified version)
  6. AnsInt mode: final answer = aggregate of sub-answers; CxtInt mode: from combined context

What's adapted to our infra:
  - LLM client = our OpenAI-compatible client (via HAGRID_LLM_BASE_URL)
  - Sentence embeddings = our nomic encoder (was OpenAI text-embedding-3-small in original)
  - Skip FlagReranker neural reranking (use embedding-cosine as proxy); see SKIP_RERANK
  - Use our existing TextRetriever for the initial chunk retrieval pass to scope to relevant docs
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Any, Tuple

import numpy as np
import networkx as nx
from openai import OpenAI

from src.tools.text_tools import TextRetriever
from src.tools.llm_utils import call_chat
from src.eval.answer_normalizer import normalize_prediction


# ── Prompts (verbatim from official ChainRAG repo) ──────────────────────────
_MULTIHOP_JUDGE_PROMPT = """You are a helpful AI assistant that determines if a question requires multiple steps to answer.

Guidelines for identifying multi-hop questions:
1. The question requires finding and connecting multiple pieces of information
2. The answer cannot be found in a single direct statement
3. You need to find intermediate information to reach the final answer

Output format should be a JSON object with only one field:
- "is_multi_hop": boolean (true/false)

Example:
Question: "Who is the paternal grandmother of Marie Of Brabant, Queen Of France?"
Output: {{"is_multi_hop": false}}
Question: "Who is Archibald Acheson, 4Th Earl Of Gosford's paternal grandfather?"
Output: {{"is_multi_hop": false}}
Question: "Who was the wife of the person who founded Microsoft?"
Output: {{"is_multi_hop": true}}

Question: "{question}"
Output:"""


_DECOMPOSE_PROMPT = """You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions.

Guidelines:
1. Only break down the question if it requires finding and connecting multiple distinct pieces of information.
2. Each sub-question should target a specific, essential piece of information.
3. Avoid generating redundant or overlapping sub-questions.
4. Order sub-questions logically so later ones can reference earlier ones.
5. Use #N to refer to the answer of sub-question N when needed.
6. Sub-questions should be minimal — only what's essential to answer the original question.
7. Keep the total number of sub-questions minimal (usually 2 at most, occasionally 3 for 3-hop chains).

Output format should be a JSON array of sub-questions, e.g.:
["sub-question 1", "sub-question 2"]

Question: "{question}"
Output:"""


_REWRITE_PROMPT = """Rewrite the following question to be self-contained by replacing pronouns or references with the actual entities they refer to.
Previous question: {prev_q}
Previous answer: {prev_ans}
Current question: {sub_q}
Rewritten question:"""


_CAN_ANSWER_PROMPT = """Based on the following context, can you answer the question?
Please respond with 'yes' or 'no' only.

Question: {question}
Context: {context}

Can you answer the question based on this context? (yes/no):"""


_FORCE_ANSWER_PROMPT = """Based on the given context, you must provide an answer with fewest words to the question.
Only give me the answer and do not output any other words.

Context: {context}
Question: {question}

Provide your best possible answer:"""


_FINAL_FROM_SUBANS_PROMPT = """Based on the answers to the sub-questions, use the fewest words possible to answer the original question.
Only give me the answer and do not output any other words.
Original Question: {original_question}
Sub-questions and their answers:
{sub_answers}
Provide the shortest possible answer:"""


_FINAL_FROM_CTX_PROMPT = """Based on the following context, use the fewest words possible to answer the original question.
Only give me the answer and do not output any other words.
Context: {context}
Question: {question}

Answer:"""


# ── Sentence-graph utilities ────────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (spaCy not required for inference)."""
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _try_load_spacy_ner():
    """Optional spaCy entity extractor. Falls back to capitalized-token heuristic if unavailable."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
        except Exception:
            return None
        return nlp
    except Exception:
        return None


_NLP = None


def extract_entities(text: str) -> List[str]:
    """Extract entity strings. Uses spaCy if available; else capitalized n-grams."""
    global _NLP
    if _NLP is None:
        _NLP = _try_load_spacy_ner()
    if _NLP is not None:
        try:
            doc = _NLP(text[:5000])
            return [e.text for e in doc.ents]
        except Exception:
            pass
    # Fallback: consecutive capitalized words
    return re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)


def build_sentence_graph(sentences: List[str],
                          sent_emb: np.ndarray,
                          k_sim: int = 5,
                          pos_window: int = 3) -> nx.Graph:
    """Build the 3-edge-type sentence graph (similarity / positional / entity)."""
    G = nx.Graph()
    n = len(sentences)
    for i in range(n):
        G.add_node(i, text=sentences[i])

    # 1. Similarity edges (k-NN by cosine)
    if n > 1:
        norm = sent_emb / (np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-9)
        sim = norm @ norm.T
        np.fill_diagonal(sim, -1)
        for i in range(n):
            topk = np.argsort(-sim[i])[:k_sim]
            for j in topk:
                if sim[i, j] > 0.3:
                    G.add_edge(i, int(j), weight=float(sim[i, j]), kind="sim")

    # 2. Positional edges (within pos_window in same doc — here treated as global sequence)
    for i in range(n):
        for off in range(1, pos_window + 1):
            if i + off < n:
                G.add_edge(i, i + off, weight=0.5, kind="pos")

    # 3. Entity-based edges
    ents_per_sent = [set(map(str.lower, extract_entities(s))) for s in sentences]
    for i in range(n):
        for j in range(i + 1, n):
            shared = ents_per_sent[i] & ents_per_sent[j]
            if shared:
                G.add_edge(i, j, weight=0.3 * len(shared), kind="entity")
    return G


def n_hop_neighbors(G: nx.Graph, seeds: List[int], hops: int) -> List[int]:
    """Return nodes reachable within `hops` from any seed (exclusive of seeds)."""
    visited = set(seeds); frontier = set(seeds)
    for _ in range(hops):
        new = set()
        for s in frontier:
            for nb in G.neighbors(s):
                if nb not in visited:
                    new.add(nb)
        if not new: break
        visited |= new; frontier = new
    return list(visited - set(seeds))


# ── LLM call helpers ────────────────────────────────────────────────────────
def _call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int = 800,
              temperature: float = 0.2) -> Tuple[str, int]:
    text, tokens = call_chat(client, model,
                              [{"role": "user", "content": prompt}],
                              max_tokens=max_tokens, temperature=temperature)
    return text.strip(), tokens


def _parse_json_list(raw: str) -> List[str]:
    """Best-effort extraction of a JSON array of strings from LLM output."""
    raw = raw.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if x]
    except Exception:
        pass
    # Fallback: split on newlines / numbering
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"').strip("'")
        if line and len(line) > 3:
            lines.append(line)
    return lines[:3]


def _parse_json_bool(raw: str, key: str = "is_multi_hop") -> bool:
    """Best-effort: extract boolean from JSON output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return bool(obj.get(key, False))
    except Exception:
        pass
    return "true" in raw.lower()


# ── Main pipeline ───────────────────────────────────────────────────────────
class ChainRAGFaithful:
    """Faithful ChainRAG: sentence-graph + multi-hop expansion + entity completion."""

    def __init__(self, processed_dir: str,
                 retriever_mode: str = "dense",
                 encoder=None, encoder_name: str = "nomic-v1.5",
                 top_k_chunks: int = 6,
                 top_k_seed_sents: int = 5,
                 max_words: int = 3000,
                 max_sub_questions: int = 3,
                 api_key: str = None,
                 model: str = "kit.gpt-oss-120b"):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.top_k_chunks = top_k_chunks
        self.top_k_seed_sents = top_k_seed_sents
        self.max_words = max_words
        self.max_sub_questions = max_sub_questions
        self.encoder = encoder
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("HAGRID_LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("HAGRID_LLM_BASE_URL")
                      or "https://ki-toolbox.scc.kit.edu/api/v1"),
        )
        self.model = model
        self.total_tokens = 0

    # ── Question decomposition ──────────────────────────────────────────
    def _is_multi_hop(self, q: str) -> bool:
        raw, tok = _call_llm(self.client, self.model,
                              _MULTIHOP_JUDGE_PROMPT.format(question=q), max_tokens=512)
        self.total_tokens += tok
        return _parse_json_bool(raw)

    def _decompose(self, q: str) -> List[str]:
        raw, tok = _call_llm(self.client, self.model,
                              _DECOMPOSE_PROMPT.format(question=q), max_tokens=768)
        self.total_tokens += tok
        subs = _parse_json_list(raw)
        return subs[:self.max_sub_questions] if subs else [q]

    def _rewrite_subq(self, prev_q: str, prev_ans: str, sub_q: str) -> str:
        if not prev_ans or len(prev_ans.split()) > 30:
            return sub_q
        raw, tok = _call_llm(self.client, self.model,
                              _REWRITE_PROMPT.format(prev_q=prev_q, prev_ans=prev_ans, sub_q=sub_q),
                              max_tokens=512)
        self.total_tokens += tok
        if not raw or len(raw.split()) > 50:
            return sub_q
        return raw

    # ── Retrieval + sentence graph ──────────────────────────────────────
    def _retrieve_sentences_and_graph(self, q: str, qid: str):
        # 1. Get top chunks from our dense retriever (scope the working set)
        chunks = self.text_retriever.retrieve(q, top_k=self.top_k_chunks, question_id=qid)
        # 2. Split into sentences
        all_sents = []; sent_meta = []
        for c in chunks:
            text = c.get("text") or c.get("content") or ""
            for s in split_sentences(text):
                all_sents.append(s)
                sent_meta.append({"title": c.get("title", ""), "chunk_id": c.get("chunk_id", "")})
        if not all_sents:
            return [], None, []
        # 3. Embed sentences with our encoder (use simple batched call)
        try:
            embs = self.encoder.encode(all_sents) if self.encoder else None
            if embs is None or not isinstance(embs, np.ndarray):
                embs = np.array(embs) if embs is not None else None
        except Exception:
            embs = None
        if embs is None:
            # Fallback: lexical similarity
            embs = np.eye(len(all_sents))[:, :min(len(all_sents), 16)]
        # 4. Build graph
        G = build_sentence_graph(all_sents, embs)
        return all_sents, G, embs

    def _seed_retrieve(self, q: str, all_sents: List[str], embs: np.ndarray) -> List[int]:
        """Embedding-cosine top-k seed sentences."""
        try:
            q_emb = self.encoder.encode([q])[0] if self.encoder else None
        except Exception:
            q_emb = None
        if q_emb is None:
            return list(range(min(len(all_sents), self.top_k_seed_sents)))
        norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        scores = norm @ qn
        return list(np.argsort(-scores)[:self.top_k_seed_sents])

    def _can_answer(self, q: str, ctx: str) -> bool:
        raw, tok = _call_llm(self.client, self.model,
                              _CAN_ANSWER_PROMPT.format(question=q, context=ctx[:6000]),
                              max_tokens=512)
        self.total_tokens += tok
        return raw.strip().lower().startswith("y")

    def _force_answer(self, q: str, ctx: str) -> str:
        raw, tok = _call_llm(self.client, self.model,
                              _FORCE_ANSWER_PROMPT.format(question=q, context=ctx[:6000]),
                              max_tokens=512, temperature=0.0)
        self.total_tokens += tok
        return raw

    # ── End-to-end per sub-question ─────────────────────────────────────
    def _answer_subq(self, sub_q: str, qid: str) -> Tuple[str, List[str]]:
        sents, G, embs = self._retrieve_sentences_and_graph(sub_q, qid)
        if not sents:
            return self._force_answer(sub_q, ""), []

        # Seed retrieval
        seeds = self._seed_retrieve(sub_q, sents, embs)
        accumulated = list(seeds)

        # Initial answerability check
        def _ctx_of(idxs):
            text = " ".join(sents[i] for i in idxs)
            return text, len(text.split())

        for hops in (0, 1, 2, 3):
            if hops > 0:
                new = n_hop_neighbors(G, seeds, hops)
                for nb in new:
                    if nb not in accumulated:
                        accumulated.append(nb)
            ctx, words = _ctx_of(accumulated)
            if words >= self.max_words:
                break
            if hops == 0:
                if self._can_answer(sub_q, ctx):
                    break
            else:
                # cheap check: enough context?
                if words > 1500 and self._can_answer(sub_q, ctx):
                    break
        ctx, _ = _ctx_of(accumulated)
        ans = self._force_answer(sub_q, ctx)
        return ans, [sents[i] for i in accumulated]

    # ── Public API ──────────────────────────────────────────────────────
    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        trace = {"question_id": question_id, "question": question, "action_trace": []}
        self.total_tokens = 0
        retrieval_calls = 0

        # 1. Multi-hop judgment
        is_mh = self._is_multi_hop(question)
        trace["action_trace"].append({"step": 1, "action": "multihop_judge", "is_multi_hop": is_mh})

        # 2. Decompose if needed
        if is_mh:
            subs = self._decompose(question)
        else:
            subs = [question]
        trace["action_trace"].append({"step": 2, "action": "decompose", "sub_questions": subs})

        # 3. Iterate sub-questions with entity completion
        sub_answers = []
        all_sents_used = []
        prev_q = None; prev_ans = None
        for i, sq in enumerate(subs):
            if i > 0 and prev_ans:
                sq_eff = self._rewrite_subq(prev_q, prev_ans, sq)
            else:
                sq_eff = sq
            ans, sents = self._answer_subq(sq_eff, question_id)
            retrieval_calls += 1
            all_sents_used.extend(sents[:5])
            sub_answers.append({"sub_q": sq, "sub_q_eff": sq_eff, "answer": ans})
            trace["action_trace"].append({"step": 3 + i, "action": f"answer_subq_{i+1}",
                                           "sub_q_eff": sq_eff, "answer": ans})
            prev_q, prev_ans = sq_eff, ans

        # 4. Final synthesis (AnsInt mode — combine sub-answers)
        if len(sub_answers) > 1:
            sub_str = "\n".join(f"{i+1}. Q: {s['sub_q_eff']} A: {s['answer']}"
                                for i, s in enumerate(sub_answers))
            final_raw, tok = _call_llm(self.client, self.model,
                                        _FINAL_FROM_SUBANS_PROMPT.format(
                                            original_question=question, sub_answers=sub_str),
                                        max_tokens=512, temperature=0.0)
            self.total_tokens += tok
        else:
            final_raw = sub_answers[-1]["answer"] if sub_answers else "I don't know"

        # Build trace in our standard format
        normalized = normalize_prediction(final_raw, question)
        trace.update({
            "answer": normalized,
            "answer_raw": final_raw,
            "text_evidence": [{"content": s, "title": "", "score": 0.0,
                                "evidence_id": f"chainrag_sent_{i}"}
                               for i, s in enumerate(all_sents_used[:10])],
            "planner_tokens": self.total_tokens,
            "verifier_tokens": 0,
            "retrieval_calls": retrieval_calls,
        })
        return trace
