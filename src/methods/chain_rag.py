"""Self-Ask-style decomposition retrieval primitive (legacy filename: chain_rag.py).

This file IS NOT a faithful reimplementation of ChainRAG. Despite the filename,
it is more accurately described as a Self-Ask-style decomposition baseline
(Press et al., Findings of EMNLP 2023, https://arxiv.org/abs/2210.03350)
with sequential retrieval per sub-question.

What this implementation does:
  1. LLM decomposes question into 2-4 sub-questions (one-shot prompt)
  2. For each sub-question:
     a. Build query = sub-question + prior answers (Self-Ask state propagation)
     b. Retrieve top-k passages
     c. LLM answers sub-question from passages
  3. Final answer = last sub-answer (AnsInt mode) or LLM integration

What this implementation does NOT do (which the original ChainRAG, Liu et al.,
ACL 2025, https://aclanthology.org/2025.acl-long.1089/, requires):
  - Sentence-level graph construction (3 edge types between sentences)
  - Key-entity completion to fix "lost-in-retrieval"
  - Progressive retrieval-and-rewriting over the sentence graph

Therefore in paper text we describe this primitive as:
  "Self-Ask-style decomposition baseline with sequential retrieval"
not as:
  "ChainRAG reimplementation"

References:
  - Self-Ask (the closest prior work to this implementation):
    Press et al., 2023. https://aclanthology.org/2023.findings-emnlp.378/
  - Least-to-Most prompting (related decomposition strategy):
    Zhou et al., 2023. https://arxiv.org/abs/2205.10625
  - ChainRAG (the inspiration, NOT reimplemented):
    Liu et al., 2025. https://aclanthology.org/2025.acl-long.1089/
"""

import json
import re
from typing import Dict, List, Any

from openai import OpenAI
from src.tools.text_tools import TextRetriever
from src.tools.llm_utils import call_chat
from src.eval.answer_normalizer import normalize_prediction


_DECOMPOSE_PROMPT = """Decompose the following multi-hop question into 2-4 simple sub-questions that can be answered one at a time. Each sub-question should build on the previous answer.

Output format: one sub-question per line, numbered. Use #N to refer to the answer of sub-question N.

Example:
Question: What is the capital of the country where the author of Harry Potter was born?
1. Who is the author of Harry Potter?
2. In which country was #1 born?
3. What is the capital of #2?

Question: {question}

Sub-questions:"""

_ANSWER_SUB_PROMPT = """Answer the following question concisely based on the given passages.
If you cannot find the answer, reply "unknown".

Passages:
{context}

Question: {question}

Answer (be concise, just the answer):"""

_INTEGRATE_PROMPT = """Based on the sub-question answers below, give the final answer to the original question.

Original question: {question}

Sub-question answers:
{chain}

Final answer (be concise, just the answer):"""


class ChainRAG:
    """ChainRAG: sub-question decomposition + iterative retrieval."""

    def __init__(self, processed_dir: str, top_k: int = 10,
                 api_key: str = None, model: str = "gpt-oss-120b",
                 retriever_mode: str = "dense", encoder_name: str = "nomic-v1.5",
                 encoder=None, max_sub_questions: int = 4):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.top_k = top_k
        self.max_sub = max_sub_questions
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("LLM_BASE_URL")
                      or ""),
        )
        self.model = model
        self.total_tokens = 0

    def _llm_call(self, prompt: str, max_tokens: int = 800) -> tuple:
        # Route through the shared call_chat() so we get the null-response retry
        # that KiRAG and abv_bridge use. The KIT gateway returns 200 with a null
        # body on transient 5xx, and the raw client crashes / returns empty —
        # which is multiplicatively bad for ChainRAG (5+ LLM calls per question
        # → cascading empty answers → F1 collapses to ~0.1).
        return call_chat(self.client, self.model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=0.0)

    def _decompose(self, question: str) -> tuple:
        """Decompose question into sub-questions. Returns (list[str], tokens)."""
        prompt = _DECOMPOSE_PROMPT.format(question=question)
        raw, tokens = self._llm_call(prompt, max_tokens=1500)
        sub_questions = []
        for line in raw.strip().splitlines():
            line = line.strip()
            # Match numbered lines like "1. ...", "2) ...", etc.
            m = re.match(r'^[\d]+[.)]\s*(.+)', line)
            if m:
                sub_questions.append(m.group(1).strip())
        # Cap at max_sub
        return sub_questions[:self.max_sub], tokens

    def _rewrite_question(self, sub_q: str, prior_answers: Dict[int, str]) -> str:
        """Replace #N references with actual answers."""
        result = sub_q
        for idx, ans in prior_answers.items():
            result = result.replace(f"#{idx}", ans)
            result = result.replace(f"#{idx} ", f"{ans} ")
        return result

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        total_tokens = 0
        retrieval_calls = 0
        chain = []

        # Step 1: Decompose
        sub_questions, dec_tok = self._decompose(question)
        total_tokens += dec_tok

        if not sub_questions:
            # Fallback: treat as single-hop
            sub_questions = [question]

        # Step 2: Iteratively answer each sub-question
        prior_answers = {}
        all_chunks = []

        for i, sub_q in enumerate(sub_questions):
            # Rewrite with prior answers
            rewritten = self._rewrite_question(sub_q, prior_answers)

            # Build query: rewritten sub-question + original question context
            query = rewritten
            if prior_answers:
                query = question + " " + " ".join(prior_answers.values()) + " " + rewritten

            # Retrieve
            chunks = self.text_retriever.retrieve(query, top_k=self.top_k,
                                                   question_id=question_id)
            retrieval_calls += 1
            all_chunks.extend(chunks)

            # Answer sub-question
            ctx = "\n\n".join(f"[{c.get('title', '')}]: {c.get('text', '')}"
                              for c in chunks[:8])
            prompt = _ANSWER_SUB_PROMPT.format(context=ctx, question=rewritten)
            sub_answer, ans_tok = self._llm_call(prompt, max_tokens=800)
            total_tokens += ans_tok

            # Take first line as answer
            if sub_answer:
                first_line = sub_answer.splitlines()[0].strip()
                if len(first_line) <= 80:
                    sub_answer = first_line

            prior_answers[i + 1] = sub_answer
            chain.append({
                "sub_question": sub_q,
                "rewritten": rewritten,
                "answer": sub_answer,
            })

        # Step 3: Final integration (AnsInt mode)
        if len(sub_questions) == 1:
            final_answer = prior_answers.get(1, "")
        else:
            chain_text = "\n".join(
                f"Q{i+1}: {c['rewritten']}\nA{i+1}: {c['answer']}"
                for i, c in enumerate(chain)
            )
            prompt = _INTEGRATE_PROMPT.format(question=question, chain=chain_text)
            final_answer, int_tok = self._llm_call(prompt, max_tokens=800)
            total_tokens += int_tok

        self.total_tokens += total_tokens

        # Dedupe chunks
        seen = set()
        union = []
        for c in all_chunks:
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                union.append(c)

        return {
            "question_id": question_id,
            "question": question,
            "status": "answer" if final_answer and final_answer.lower() != "i don't know" else "abstain",
            "answer": normalize_prediction(final_answer, question) if final_answer else None,
            "answer_raw": final_answer,
            "text_evidence": [
                {"evidence_id": c["chunk_id"], "content": c["text"],
                 "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                 "metadata": {"title": c.get("title", "")}}
                for c in union[:8]
            ],
            "graph_evidence": [],
            "action_trace": [{
                "step": 1,
                "final_action": "chain_rag",
                "n_sub_questions": len(sub_questions),
                "chain": chain,
            }],
            "budget_used": {
                "steps_used": len(sub_questions) + 1,
                "tool_calls_used": retrieval_calls,
                "tokens_used": total_tokens,
                "verifications_used": 0,
            },
            "planner_tokens": total_tokens,
            "verifier_tokens": 0,
        }
