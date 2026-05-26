"""Local Repair: one-step entity replacement for failed branches.

When a branch is marked "repair" by the verifier:
  - Re-read the one-shot evidence with a more targeted prompt
  - Ask LLM to propose an alternative bridge entity
  - Only ONE repair attempt allowed (no cascading)
"""

from typing import Dict, List, Any, Optional

from .llm_client import LLMClient


_REPAIR_PROMPT = """A previous attempt to decompose this question identified "{old_entity}" as an intermediate entity, but the evidence retrieved using it was not helpful.

Please read the passages again and suggest ONE alternative intermediate entity (a different name, date, or place) that might better connect the question to its answer.

Passages:
{context}

Question: {question}

Reply with ONLY the alternative entity (≤5 words), or "NONE" if you cannot find a better one.
Alternative entity:"""


class LocalRepair:
    """Attempts one-step entity replacement for a failed branch."""

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def repair(self, question: str, failed_proposal: Dict,
               one_shot_chunks: List[Dict]) -> tuple:
        """Try to replace a failed bridge entity.

        Returns:
            (repaired_proposal: Dict or None, tokens_used: int)
        """
        old_entity = failed_proposal.get("bridge_entity", "")
        if not old_entity:
            return None, 0

        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in one_shot_chunks[:8]
        )
        prompt = _REPAIR_PROMPT.format(
            old_entity=old_entity,
            context=ctx,
            question=question,
        )
        raw, tokens = self.llm.call(prompt, max_tokens=800)

        # Parse: take first line, strip
        if raw:
            entity = raw.strip().splitlines()[0].strip()
            if entity and entity.upper() != "NONE" and entity.lower() != old_entity.lower():
                return {
                    "bridge_entity": entity,
                    "bridge_relation": failed_proposal.get("bridge_relation", ""),
                    "missing_slot": failed_proposal.get("missing_slot", ""),
                    "confidence": 0.3,  # Lower confidence for repairs
                    "repaired_from": old_entity,
                }, tokens

        return None, tokens
