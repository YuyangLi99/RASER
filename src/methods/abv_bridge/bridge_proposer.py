"""Bridge Proposer: extracts top-k structured bridge proposals from one-shot evidence.

Output format per proposal:
    {
        "bridge_entity": str,       # The intermediate entity/fact
        "bridge_relation": str,     # What role it plays (e.g. "birthplace of X")
        "missing_slot": str,        # What we still need to find
        "confidence": float,        # LLM self-assessed confidence 0-1
    }

Design:
  - Ask LLM to decompose the question into sub-questions.
  - Extract the first unanswered sub-question's answer as bridge entity.
  - Generate up to top_k proposals ranked by confidence.
  - Strict format enforcement: parse JSON, fall back to regex if needed.
"""

import json
import re
from typing import Dict, List, Any

from .llm_client import LLMClient


_PROPOSER_PROMPT = """You are decomposing a multi-hop question to find intermediate bridge entities.

Given the question and retrieved passages, identify up to {top_k} intermediate facts that could help answer the question. Each fact should be a short entity, name, date, or place that connects one part of the question to another.

For each bridge proposal, output a JSON object on its own line with these fields:
- "bridge_entity": the intermediate fact (short string, ≤5 words)
- "bridge_relation": what role this entity plays (e.g., "author of the book", "capital of the state")
- "missing_slot": what we still need to find after this bridge (e.g., "birth year of this person")
- "confidence": your confidence that this bridge is correct (0.0 to 1.0)

Output ONLY the JSON objects, one per line. No other text.

Passages:
{context}

Question: {question}
Current candidate answer: {current_answer}

Bridge proposals (one JSON per line):"""


def _parse_proposals(raw: str, top_k: int) -> List[Dict[str, Any]]:
    """Parse LLM output into structured bridge proposals."""
    proposals = []
    # Try JSON parsing line by line
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip markdown code fences
        if line.startswith("```"):
            continue
        # Try to extract JSON from the line
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "bridge_entity" in obj:
                proposals.append({
                    "bridge_entity": str(obj.get("bridge_entity", "")).strip(),
                    "bridge_relation": str(obj.get("bridge_relation", "")).strip(),
                    "missing_slot": str(obj.get("missing_slot", "")).strip(),
                    "confidence": float(obj.get("confidence", 0.5)),
                })
        except (json.JSONDecodeError, ValueError):
            # Try regex fallback for "bridge_entity": "..." pattern
            m = re.search(r'"bridge_entity"\s*:\s*"([^"]+)"', line)
            if m:
                proposals.append({
                    "bridge_entity": m.group(1).strip(),
                    "bridge_relation": "",
                    "missing_slot": "",
                    "confidence": 0.4,
                })
    # Also try parsing the entire output as a JSON array
    if not proposals:
        try:
            arr = json.loads(raw.strip())
            if isinstance(arr, list):
                for obj in arr:
                    if isinstance(obj, dict) and "bridge_entity" in obj:
                        proposals.append({
                            "bridge_entity": str(obj.get("bridge_entity", "")).strip(),
                            "bridge_relation": str(obj.get("bridge_relation", "")).strip(),
                            "missing_slot": str(obj.get("missing_slot", "")).strip(),
                            "confidence": float(obj.get("confidence", 0.5)),
                        })
        except (json.JSONDecodeError, ValueError):
            pass

    # Deduplicate by bridge_entity
    seen = set()
    deduped = []
    for p in proposals:
        key = p["bridge_entity"].lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)

    # Sort by confidence descending, take top_k
    deduped.sort(key=lambda x: -x["confidence"])
    return deduped[:top_k]


class BridgeProposer:
    """Generates top-k structured bridge proposals from question + evidence."""

    def __init__(self, llm: LLMClient = None, top_k: int = 3):
        self.llm = llm or LLMClient()
        self.top_k = top_k

    def propose(self, question: str, chunks: List[Dict],
                current_answer: str = "") -> tuple:
        """Returns (proposals: List[Dict], tokens_used: int)."""
        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in chunks[:8]
        )
        prompt = _PROPOSER_PROMPT.format(
            top_k=self.top_k,
            context=ctx,
            question=question,
            current_answer=current_answer or "(none)",
        )
        raw, tokens = self.llm.call(prompt, max_tokens=1500)
        proposals = _parse_proposals(raw, self.top_k)
        return proposals, tokens
