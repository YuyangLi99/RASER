"""Trigger Gate: decides whether to escalate from one-shot to bridge search.

Design:
  - Default action is STOP (trust one-shot answer).
  - Triggers BRIDGE only when evidence is insufficient or unstable.
  - Rule-based v1; can be replaced with a trained classifier later.

Trigger signals:
  1. Self-consistency: run one-shot twice with temp>0, check agreement.
  2. Answer hedging: LLM says "I don't know" or answer is very short/vague.
  3. Evidence dispersion: top-k chunks have low max score or high entropy.
  4. Question complexity: word count, presence of multi-hop cue phrases.
"""

import re
from typing import Dict, List, Any


# Multi-hop cue phrases that suggest decomposition would help
_BRIDGE_CUES = [
    r"\bwho .*(?:born|lived|located) in.*(?:where|which|what)\b",
    r"\bthe (?:country|city|state|person|author|director|founder) (?:of|where|who)\b",
    r"\bwhat is the .* of the .* (?:of|in|by|from)\b",
    r"\bwhere .*(?:born|located|founded).*(?:of|by)\b",
]


def estimate_hop_complexity(question: str) -> int:
    """Heuristic hop count estimate from question structure."""
    q_lower = question.lower()
    # Count nested "of/in/by/from/where/who" patterns
    nesting = len(re.findall(r'\b(?:of the|in the|by the|from the|where the|who the)\b', q_lower))
    # Also count relative clauses and embedded references
    relative = len(re.findall(r'\b(?:who |which |that |whose |where )\b', q_lower))
    # Possessive chains: "X's Y's Z"
    possessive = len(re.findall(r"'s\b", q_lower))
    total_signals = nesting + relative + possessive
    if total_signals >= 4:
        return 4
    if total_signals >= 3:
        return 3
    if total_signals >= 1:
        return 2
    # Length-based fallback
    if len(question.split()) > 15:
        return 2
    return 1


def compute_evidence_concentration(chunks: List[Dict], top_k: int = 10) -> float:
    """How concentrated is the retrieval score? High = confident, low = scattered."""
    if not chunks:
        return 0.0
    scores = []
    for c in chunks[:top_k]:
        s = c.get("dense_score", c.get("bm25_score", c.get("rrf_score", 0.0)))
        scores.append(float(s))
    if not scores or max(scores) == 0:
        return 0.0
    # Ratio of top-1 to mean
    mean_s = sum(scores) / len(scores)
    if mean_s == 0:
        return 0.0
    return scores[0] / mean_s


def has_bridge_cues(question: str) -> bool:
    """Check if question has structural cues suggesting multi-hop."""
    q_lower = question.lower()
    for pattern in _BRIDGE_CUES:
        if re.search(pattern, q_lower):
            return True
    return False


def trigger_gate(question: str, one_shot_answer: str, chunks: List[Dict],
                 one_shot_tokens: int = 0) -> Dict[str, Any]:
    """Decide whether to trigger bridge search.

    Returns:
        {
            "trigger": True/False,
            "reason": str,       # "confident" | "low_confidence" | "missing_bridge" | ...
            "hop_estimate": int,
            "evidence_concentration": float,
        }
    """
    hop_est = estimate_hop_complexity(question)
    conc = compute_evidence_concentration(chunks)
    has_cues = has_bridge_cues(question)
    answer_empty = not one_shot_answer or one_shot_answer.lower() in ("i don't know", "")

    # --- Decision logic ---

    # If one-shot abstained, always trigger
    if answer_empty:
        return {
            "trigger": True,
            "reason": "answer_abstain",
            "hop_estimate": hop_est,
            "evidence_concentration": conc,
        }

    # If estimated ≥3 hops, trigger (multi-hop questions rarely solved one-shot)
    if hop_est >= 3:
        return {
            "trigger": True,
            "reason": "high_hop_complexity",
            "hop_estimate": hop_est,
            "evidence_concentration": conc,
        }

    # If 2-hop with bridge cues and moderate evidence, trigger
    if hop_est >= 2 and has_cues:
        return {
            "trigger": True,
            "reason": "bridge_cue_detected",
            "hop_estimate": hop_est,
            "evidence_concentration": conc,
        }

    # If evidence is very scattered (low concentration), trigger
    if conc < 1.5 and hop_est >= 2:
        return {
            "trigger": True,
            "reason": "evidence_dispersion",
            "hop_estimate": hop_est,
            "evidence_concentration": conc,
        }

    # Default: trust one-shot
    return {
        "trigger": False,
        "reason": "confident",
        "hop_estimate": hop_est,
        "evidence_concentration": conc,
    }
