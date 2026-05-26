"""Final Synthesizer: produces the answer from curated evidence.

Key design:
  - Receives one-shot evidence + surviving branch evidence
  - Bridge entities are passed as authoritative intermediate facts
  - Output is a short answer string
"""

from typing import Dict, List, Any

from .llm_client import LLMClient


_SYNTH_PROMPT = """Answer the following question concisely based on the given evidence.
If you cannot find the answer, reply "I don't know".

{bridge_block}Evidence:
{context}

Question: {question}

Answer (be concise, just the answer):"""


class FinalSynthesizer:
    """Produces the final answer from curated evidence and bridge facts."""

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def synthesize(self, question: str,
                   one_shot_chunks: List[Dict],
                   branch_chunks: List[Dict],
                   bridges: List[str],
                   max_evidence: int = 15) -> tuple:
        """Synthesize final answer.

        Args:
            question: original question
            one_shot_chunks: from initial retrieval
            branch_chunks: from surviving branches (already pruned)
            bridges: list of authoritative bridge entity strings
            max_evidence: max chunks to include in prompt

        Returns:
            (answer: str, tokens: int)
        """
        # Merge evidence: branch chunks first (higher value), then one-shot
        seen = set()
        merged = []
        for c in branch_chunks:
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                merged.append(c)
        for c in one_shot_chunks:
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                merged.append(c)
        merged = merged[:max_evidence]

        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in merged
        )

        bridge_block = ""
        if bridges:
            unique = []
            for b in bridges:
                if b and b not in unique:
                    unique.append(b)
            if unique:
                bridge_block = (
                    "Intermediate facts established by decomposition "
                    "(treat as authoritative bridge entities):\n"
                    + "\n".join(f"- {b}" for b in unique)
                    + "\n\n"
                )

        prompt = _SYNTH_PROMPT.format(
            bridge_block=bridge_block,
            context=ctx,
            question=question,
        )
        answer, tokens = self.llm.call(prompt, max_tokens=800)
        return answer, tokens
