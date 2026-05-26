"""Policy Gate: conditional routing for ABV-Bridge.

Instead of a binary trigger (STOP vs BRIDGE), the policy gate routes each
question to the cheapest-sufficient strategy:

    STOP              → trust one-shot answer (0 extra LLM calls)
    TOP1_BRIDGE       → single focused bridge (2 extra LLM calls)
    TOP2_BRIDGE       → parallel top-2 bridge, no pruning (2 extra)
    TOP2_PRUNE        → top-2 + verifier pruning (2 extra)
    TOP2_PRUNE_REPAIR → top-2 + prune + local repair (3 extra)

Routing features:
    1. hop_estimate       — heuristic hop complexity (1-4)
    2. evidence_conc      — retrieval score concentration (higher = more confident)
    3. answer_type        — yes_no / date / count / entity
    4. has_bridge_cues    — structural multi-hop patterns in question
    5. one_shot_confidence — proxy: is answer empty/hedged?
    6. answer_length      — very short answers on entity questions may be wrong
"""

from typing import Dict, List, Any

from .trigger_gate import (
    estimate_hop_complexity,
    compute_evidence_concentration,
    has_bridge_cues,
)
from src.eval.answer_normalizer import classify_question_type


# Route enum
STOP = "STOP"
TOP1 = "TOP1_BRIDGE"
TOP2 = "TOP2_BRIDGE"
TOP2_PRUNE = "TOP2_PRUNE"
TOP2_REPAIR = "TOP2_PRUNE_REPAIR"


def _answer_confidence(answer: str, question: str) -> float:
    """Proxy confidence from answer surface form. 0=no confidence, 1=high."""
    if not answer or answer.strip().lower() in ("i don't know", ""):
        return 0.0

    qtype = classify_question_type(question)

    # Yes/no answers on yes/no questions are usually confident
    if qtype == "yes_no" and answer.strip().lower() in ("yes", "no"):
        return 0.9

    # Very short entity answers might be guesses, but dates/counts are usually precise
    if qtype in ("date", "count"):
        return 0.8

    # Entity answers: length heuristic
    words = answer.strip().split()
    if len(words) <= 2:
        return 0.6  # short but plausible
    if len(words) >= 10:
        return 0.3  # essay-like answer = low confidence
    return 0.7


def policy_gate(question: str, one_shot_answer: str,
                chunks: List[Dict]) -> Dict[str, Any]:
    """Route a question to the cheapest-sufficient ABV-Bridge strategy.

    Returns:
        {
            "route": str,          # STOP / TOP1_BRIDGE / ... / TOP2_PRUNE_REPAIR
            "reason": str,
            "features": dict,      # all routing features for logging
            "top_k_bridges": int,  # 0, 1, or 2
            "use_verifier": bool,
            "use_repair": bool,
        }
    """
    hop_est = estimate_hop_complexity(question)
    conc = compute_evidence_concentration(chunks)
    cues = has_bridge_cues(question)
    qtype = classify_question_type(question)
    ans_conf = _answer_confidence(one_shot_answer, question)

    features = {
        "hop_estimate": hop_est,
        "evidence_concentration": round(conc, 3),
        "has_bridge_cues": cues,
        "question_type": qtype,
        "answer_confidence": round(ans_conf, 3),
    }

    # --- Routing logic ---

    # Rule 1: If one-shot abstained, always escalate to full pipeline
    if ans_conf == 0.0:
        return _route(TOP2_REPAIR, "answer_abstain", features)

    # Rule 2: Comparison / yes-no questions with high confidence → STOP
    # These rarely benefit from bridging
    if qtype == "yes_no" and ans_conf >= 0.8:
        return _route(STOP, "confident_yesno", features)

    # Rule 3: High evidence concentration + confident answer → STOP
    if conc >= 2.0 and ans_conf >= 0.7:
        return _route(STOP, "high_concentration_confident", features)

    # Rule 4: Simple questions (hop=1, no cues) with decent confidence → STOP
    if hop_est <= 1 and not cues and ans_conf >= 0.6:
        return _route(STOP, "simple_confident", features)

    # Rule 5: Moderate difficulty (hop=2, some cues) → TOP1_BRIDGE
    if hop_est == 2 and not cues and conc >= 1.5:
        return _route(TOP1, "moderate_focused", features)

    # Rule 6: Clear 2-hop with bridge cues → TOP1_BRIDGE
    if hop_est == 2 and cues:
        return _route(TOP1, "2hop_with_cues", features)

    # Rule 7: 3-hop → TOP1_BRIDGE (ablation showed top1 is best for 3hop)
    if hop_est == 3 and conc >= 1.2:
        return _route(TOP1, "3hop_focused", features)

    # Rule 8: 3-hop with scattered evidence → TOP2_PRUNE
    if hop_est == 3:
        return _route(TOP2_PRUNE, "3hop_scattered", features)

    # Rule 9: 4-hop → TOP2_PRUNE_REPAIR (repair only helps here)
    if hop_est >= 4:
        return _route(TOP2_REPAIR, "4hop_needs_repair", features)

    # Rule 10: Scattered evidence (low concentration) → TOP2_PRUNE
    if conc < 1.5 and hop_est >= 2:
        return _route(TOP2_PRUNE, "evidence_dispersion", features)

    # Default: moderate escalation
    if ans_conf < 0.6:
        return _route(TOP1, "low_confidence_default", features)

    return _route(STOP, "confident_default", features)


def _route(route: str, reason: str, features: dict) -> Dict[str, Any]:
    """Build a routing decision dict."""
    config = {
        STOP:        {"top_k_bridges": 0, "use_verifier": False, "use_repair": False},
        TOP1:        {"top_k_bridges": 1, "use_verifier": False, "use_repair": False},
        TOP2:        {"top_k_bridges": 2, "use_verifier": False, "use_repair": False},
        TOP2_PRUNE:  {"top_k_bridges": 2, "use_verifier": True,  "use_repair": False},
        TOP2_REPAIR: {"top_k_bridges": 2, "use_verifier": True,  "use_repair": True},
    }
    return {
        "route": route,
        "reason": reason,
        "features": features,
        **config[route],
    }
