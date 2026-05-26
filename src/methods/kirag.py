"""IRCoT-style iterative retrieval primitive (legacy filename: kirag.py).

This file IS NOT a faithful reimplementation of KiRAG. Despite the historical
filename, it is more accurately described as an IRCoT-style iterative
retrieval baseline (Trivedi et al., ACL 2023, https://arxiv.org/abs/2212.10509)
with a triple-based state representation instead of free-form CoT text.

What this implementation does:
  1. Initial retrieval (top-k chunks)
  2. Zero-shot LLM extracts knowledge triples from retrieved chunks
  3. Zero-shot LLM identifies what's "missing" → next retrieval query
  4. Re-retrieve with augmented query (gap + entity context)
  5. Repeat up to max_iterations (default 2)
  6. Final answer from accumulated triples + chunks

What this implementation does NOT do (which the original KiRAG, Fang et al.,
ACL 2025, https://aclanthology.org/2025.acl-long.929/, requires):
  - Trained Reasoning Chain Aligner (E5/BGE-initialized retriever model
    supervised on bridging triples for HotpotQA / 2Wiki / MuSiQue)
  - Pre-computed triple-level KG corpus
  - Triple-relevance-based document re-ranking

Therefore in paper text we describe this primitive as:
  "IRCoT-style iterative retrieval with a triple-state representation"
not as:
  "KiRAG reimplementation"

The trace files produced by this module (kirag_*_traces.jsonl) are used as
the ITER route in the 3-route RASER feasibility study; the paper uses the
generic name ITER (or IRCoT-style) to avoid misrepresenting full KiRAG.

References:
  - IRCoT (the closest prior work to this implementation):
    Trivedi et al., 2023. https://aclanthology.org/2023.acl-long.557/
  - KiRAG (the inspiration for the triple representation, NOT reimplemented):
    Fang et al., 2025. https://aclanthology.org/2025.acl-long.929/
"""

import json
import re
from typing import Dict, List, Any, Optional

from openai import OpenAI
from src.tools.text_tools import TextRetriever
from src.tools.llm_utils import call_chat
from src.eval.answer_normalizer import normalize_prediction


_EXTRACT_TRIPLES_PROMPT = """Extract the most important knowledge triples from these passages that are relevant to answering the question. A triple is (subject, relation, object).

Question: {question}

Passages:
{context}

Output up to 5 triples, one per line, in format: subject | relation | object
Only include triples directly relevant to the question.

Triples:"""

_IDENTIFY_GAP_PROMPT = """You are building a reasoning chain to answer a multi-hop question. Given the question and known facts, identify what knowledge is STILL MISSING.

Question: {question}

Known facts:
{known_facts}

If the facts are sufficient to answer, reply "SUFFICIENT".
Otherwise, describe the missing link as a short search query (≤15 words) that would help find the bridging information.

Missing knowledge:"""

_ANSWER_PROMPT = """Answer the following question concisely based on the reasoning chain and evidence.
If you cannot find the answer, reply "I don't know".

Reasoning chain (known facts):
{chain}

Supporting evidence:
{context}

Question: {question}

Answer (be concise, just the answer):"""


def _parse_triples(raw: str) -> List[Dict[str, str]]:
    """Parse 'subject | relation | object' lines."""
    triples = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            triples.append({
                "subject": parts[0],
                "relation": parts[1],
                "object": parts[2],
            })
        elif len(parts) == 2:
            # Try comma/dash split
            triples.append({
                "subject": parts[0],
                "relation": "related_to",
                "object": parts[1],
            })
    return triples[:5]


class KiRAG:
    """KiRAG: knowledge-triple-driven iterative retrieval."""

    def __init__(self, processed_dir: str, top_k: int = 10,
                 api_key: str = None, model: str = "kit.gpt-oss-120b",
                 retriever_mode: str = "dense", encoder_name: str = "nomic-v1.5",
                 encoder=None, max_iterations: int = 2):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.top_k = top_k
        self.max_iter = max_iterations
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("HAGRID_LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("HAGRID_LLM_BASE_URL")
                      or "https://ki-toolbox.scc.kit.edu/api/v1"),
        )
        self.model = model
        self.total_tokens = 0

    def _llm_call(self, prompt: str, max_tokens: int = 800) -> tuple:
        return call_chat(self.client, self.model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=0.0)

    def _extract_triples(self, question: str, chunks: List[Dict]) -> tuple:
        """Extract knowledge triples from passages."""
        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in chunks[:8]
        )
        prompt = _EXTRACT_TRIPLES_PROMPT.format(question=question, context=ctx)
        raw, tokens = self._llm_call(prompt, max_tokens=1500)
        triples = _parse_triples(raw)
        return triples, tokens

    def _identify_gap(self, question: str, known_triples: List[Dict]) -> tuple:
        """Identify missing knowledge in the reasoning chain."""
        facts = "\n".join(
            f"- {t['subject']} | {t['relation']} | {t['object']}"
            for t in known_triples
        ) if known_triples else "(none yet)"
        prompt = _IDENTIFY_GAP_PROMPT.format(question=question, known_facts=facts)
        raw, tokens = self._llm_call(prompt, max_tokens=800)

        if "SUFFICIENT" in raw.upper():
            return None, tokens

        # Take first line as gap query
        query = raw.splitlines()[0].strip() if raw else None
        if query and len(query) > 100:
            query = query[:100]
        return query, tokens

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        total_tokens = 0
        retrieval_calls = 0
        all_triples = []
        all_chunks = []
        iterations = []

        # Initial retrieval
        chunks = self.text_retriever.retrieve(question, top_k=self.top_k,
                                               question_id=question_id)
        retrieval_calls += 1
        all_chunks.extend(chunks)

        # Extract initial triples
        triples, ext_tok = self._extract_triples(question, chunks)
        total_tokens += ext_tok
        all_triples.extend(triples)

        iterations.append({
            "iteration": 0,
            "action": "initial_extract",
            "n_triples": len(triples),
        })

        # Iterative gap-filling
        for it in range(self.max_iter):
            gap_query, gap_tok = self._identify_gap(question, all_triples)
            total_tokens += gap_tok

            if not gap_query:
                iterations.append({
                    "iteration": it + 1,
                    "action": "sufficient",
                })
                break

            # Retrieve with gap query augmented with known entities
            entity_context = " ".join(
                t["subject"] + " " + t["object"]
                for t in all_triples[-3:]  # last 3 triples
            )
            augmented_query = question + " " + gap_query + " " + entity_context
            new_chunks = self.text_retriever.retrieve(
                augmented_query, top_k=self.top_k,
                question_id=question_id
            )
            retrieval_calls += 1

            # Extract new triples
            new_triples, ext_tok = self._extract_triples(question, new_chunks)
            total_tokens += ext_tok

            # Dedupe triples
            existing = {(t["subject"].lower(), t["object"].lower()) for t in all_triples}
            added = 0
            for t in new_triples:
                key = (t["subject"].lower(), t["object"].lower())
                if key not in existing:
                    existing.add(key)
                    all_triples.append(t)
                    added += 1

            all_chunks.extend(new_chunks)
            iterations.append({
                "iteration": it + 1,
                "action": "gap_fill",
                "gap_query": gap_query,
                "new_triples": added,
            })

            if added == 0:
                break

        # Final answer with reasoning chain
        chain = "\n".join(
            f"- {t['subject']} → {t['relation']} → {t['object']}"
            for t in all_triples
        ) if all_triples else "(no facts extracted)"

        # Dedupe chunks
        seen = set()
        union = []
        for c in all_chunks:
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                union.append(c)

        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in union[:12]
        )
        prompt = _ANSWER_PROMPT.format(chain=chain, context=ctx, question=question)
        answer, ans_tok = self._llm_call(prompt, max_tokens=800)
        total_tokens += ans_tok
        self.total_tokens += total_tokens

        return {
            "question_id": question_id,
            "question": question,
            "status": "answer" if answer and answer.lower() != "i don't know" else "abstain",
            "answer": normalize_prediction(answer, question) if answer else None,
            "answer_raw": answer,
            "text_evidence": [
                {"evidence_id": c["chunk_id"], "content": c["text"],
                 "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                 "metadata": {"title": c.get("title", "")}}
                for c in union[:8]
            ],
            "graph_evidence": [],
            "action_trace": [{
                "step": 1,
                "final_action": "kirag",
                "iterations": iterations,
                "n_triples": len(all_triples),
                "triples": [
                    f"{t['subject']} | {t['relation']} | {t['object']}"
                    for t in all_triples
                ],
            }],
            "budget_used": {
                "steps_used": len(iterations) + 1,
                "tool_calls_used": retrieval_calls,
                "tokens_used": total_tokens,
                "verifications_used": 0,
            },
            "planner_tokens": total_tokens,
            "verifier_tokens": 0,
        }
