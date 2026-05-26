"""Conditional router for ABV-Bridge: LLM router + safety gate.

Design (2026-04-13, driven by oracle/feature analysis on MuSiQue N=100):

    Inner: LLM router over a structured, ANSWER-SIDE state
           → outputs one of 5 discrete routes
    Outer: hard rule safety gate
           → enforces budget + strong empirical rules (date/yes_no → STOP)

Why answer-side state:
    Feature analysis (Cohen's d) found the strongest separators between
    "bridge strictly helps" vs "one-shot is fine" are all on the answer
    side:
        answer_len_words       (d=0.794)
        has_bridge_cues        (d=0.670)
        answer_confidence      (d=0.526)
        retrieval_score_gap    (d=0.490)
        question_type_is_date  (d=0.553, hard rule: bridge_wins has 0% date)
    `hop_estimate` — the old policy_gate's core feature — was found to be
    empirically useless (d=0.115). We drop it as a routing signal.

Five routes (same as policy_gate):
    STOP              → trust one-shot answer (cheapest)
    TOP1_BRIDGE       → single focused bridge
    TOP2_BRIDGE       → parallel top-2 bridge, no pruning
    TOP2_PRUNE        → top-2 + verifier
    TOP2_PRUNE_REPAIR → top-2 + verifier + local repair
"""

import json
import re
from typing import Dict, List, Any, Optional

from .trigger_gate import (
    compute_evidence_concentration,
    has_bridge_cues,
)
from .llm_client import LLMClient
from src.eval.answer_normalizer import classify_question_type


# --- Route constants ---
STOP = "STOP"
TOP1 = "TOP1_BRIDGE"
TOP2 = "TOP2_BRIDGE"
TOP2_PRUNE = "TOP2_PRUNE"
TOP2_REPAIR = "TOP2_PRUNE_REPAIR"
ABSTAIN = "ABSTAIN"

_ALL_ROUTES = (STOP, TOP1, TOP2, TOP2_PRUNE, TOP2_REPAIR)

# Selective 3-route space: used by the SelectiveLLMRouter. Motivated by the
# N=300 route-wise analysis: TOP1_BRIDGE is dominated by STOP on both datasets
# (router mis-escalates unrecoverable hard cases), while TOP2_PRUNE captures
# the genuine bridge-wins regime. ABSTAIN is a recoverability signal — the
# router declares the sample unrecoverable and passes through the one-shot
# answer with an abstain flag so selective-prediction metrics (AURC, risk@
# coverage) can reward correct unrecoverable detection.
_SELECTIVE_ROUTES = (STOP, TOP2_PRUNE, ABSTAIN)

_ROUTE_CONFIG = {
    STOP:        {"top_k_bridges": 0, "use_verifier": False, "use_repair": False},
    TOP1:        {"top_k_bridges": 1, "use_verifier": False, "use_repair": False},
    TOP2:        {"top_k_bridges": 2, "use_verifier": False, "use_repair": False},
    TOP2_PRUNE:  {"top_k_bridges": 2, "use_verifier": True,  "use_repair": False},
    TOP2_REPAIR: {"top_k_bridges": 2, "use_verifier": True,  "use_repair": True},
    ABSTAIN:     {"top_k_bridges": 0, "use_verifier": False, "use_repair": False},
}

# Abstract cost (LLM-call units) used by budget-aware downgrade
_ROUTE_COST = {STOP: 1, TOP1: 4, TOP2: 5, TOP2_PRUNE: 6, TOP2_REPAIR: 7,
               ABSTAIN: 1}

# Downgrade chain under budget pressure
_DOWNGRADE = {TOP2_REPAIR: TOP2_PRUNE, TOP2_PRUNE: TOP2, TOP2: TOP1,
              TOP1: STOP, STOP: STOP, ABSTAIN: ABSTAIN}


# =============================================================================
# Structured state builder
# =============================================================================

def _answer_confidence_proxy(answer: str, qtype: str) -> float:
    if not answer or answer.strip().lower() in ("i don't know", ""):
        return 0.0
    if qtype == "yes_no" and answer.strip().lower() in ("yes", "no"):
        return 0.9
    if qtype in ("date", "count"):
        return 0.8
    words = answer.strip().split()
    if len(words) <= 2:
        return 0.6
    if len(words) >= 10:
        return 0.3
    return 0.7


def build_router_state(question: str, one_shot_answer: str,
                       chunks: List[Dict]) -> Dict[str, Any]:
    """Pack the features that matter for routing into a JSON-serializable state.

    Deliberately excludes retrieved titles/text — the router's job is routing,
    not re-reading evidence. Keeping the state narrow prevents the LLM from
    drifting into implicit QA reasoning.
    """
    qtype = classify_question_type(question)
    scores = [float(c.get("dense_score", 0.0)) for c in chunks[:10]]
    top1 = scores[0] if scores else 0.0
    top5 = scores[4] if len(scores) >= 5 else 0.0
    return {
        "question": question,
        "one_shot_answer": one_shot_answer or "(empty)",
        "question_type": qtype,
        "answer_confidence_proxy": round(_answer_confidence_proxy(one_shot_answer, qtype), 3),
        "answer_len_words": len((one_shot_answer or "").split()),
        "has_bridge_cues": bool(has_bridge_cues(question)),
        "retrieval_score_top1": round(top1, 4),
        "retrieval_score_gap": round(top1 - top5, 4),
        "evidence_concentration": round(compute_evidence_concentration(chunks), 3),
    }


# =============================================================================
# Rule-based router (diagnostic baseline)
# =============================================================================

def rule_router(state: Dict[str, Any]) -> Dict[str, Any]:
    """Interpretable if/else router based on feature separability analysis.

    Not intended as the final method — serves as a transparent baseline
    to show how much of the oracle headroom is reachable from hand-written
    rules alone. The features with the highest Cohen's d are used.
    """
    qtype = state["question_type"]
    ans_len = state["answer_len_words"]
    ans_conf = state["answer_confidence_proxy"]
    bridge_cues = state["has_bridge_cues"]
    score_gap = state["retrieval_score_gap"]

    # Rule 1: abstained one-shot → escalate to full pipeline
    if ans_conf == 0.0:
        return _decision(TOP2_REPAIR, "abstain_escalate", state)

    # Rule 2: date / count / yes_no → STOP (empirically 0% bridge_wins)
    if qtype in ("date", "yes_no", "count"):
        return _decision(STOP, f"type_{qtype}_never_benefits", state)

    # Rule 3: short confident entity answer → STOP
    # bridge_wins mean answer_len=4.2; stop_perfect mean=2.3
    if ans_len <= 2 and ans_conf >= 0.6 and not bridge_cues:
        return _decision(STOP, "short_confident_entity", state)

    # Rule 4: long hedging answer + bridge cues → TOP2_PRUNE
    # combining the two strongest separators
    if ans_len >= 5 and bridge_cues:
        return _decision(TOP2_PRUNE, "long_answer_with_cues", state)

    # Rule 5: long hedging answer, scattered retrieval → TOP2_PRUNE
    if ans_len >= 5 and score_gap < 0.10:
        return _decision(TOP2_PRUNE, "long_answer_scattered", state)

    # Rule 6: medium answer with bridge cues → TOP1_BRIDGE
    if bridge_cues and ans_len >= 3:
        return _decision(TOP1, "bridge_cues_medium", state)

    # Rule 7: medium-length answer without cues → TOP1_BRIDGE (cheap check)
    if 3 <= ans_len <= 5 and ans_conf < 0.7:
        return _decision(TOP1, "medium_low_conf", state)

    # Default: trust one-shot
    return _decision(STOP, "default_trust_one_shot", state)


# =============================================================================
# LLM router (final method inner layer)
# =============================================================================

_ROUTER_SYSTEM = """You are a routing policy for a multi-hop QA controller. \
Your job is to read a JSON state describing a question, a one-shot answer, \
and simple retrieval statistics, then output EXACTLY ONE discrete action.

You MUST NOT try to answer the question. Your only job is to pick the \
cheapest route that is sufficient to get a correct final answer."""

_ROUTER_USER_TEMPLATE = """Given the following state, identify whether this question is a BRIDGE-GAP case and pick the cheapest sufficient route.

State:
{state_json}

Your task: decide whether the one-shot answer is already reliable, or whether a second retrieval hop through a bridge entity is needed to actually answer the question.

Positive signals that this IS a bridge-gap case (escalate):
- The question contains bridge cues (has_bridge_cues=true): "who directed the movie that...", "the spouse of the author of...", "the country where X was born". These are compositional questions where the one-shot retriever rarely has both endpoints in one chunk.
- The one-shot answer is long and hedging (answer_len_words ≥ 5), which usually means the model is paraphrasing retrieved context without having the actual target entity.
- answer_confidence_proxy is low (< 0.6), or the retrieval is flat (retrieval_score_gap < 0.10), indicating no dominant chunk.
- The one-shot answer looks like an intermediate entity (the bridge itself) rather than the final target the question asks for.

Signals that the one-shot answer is ALREADY sufficient (STOP is enough):
- A short, confident entity/number/date answer that directly matches the question type.
- question_type in {{date, yes_no, count}}: these are almost never resolved by bridging — pick STOP.

Routes — pick the cheapest one that you believe is sufficient:
- STOP              — trust the one-shot answer. Use this only if you actively believe it is already correct, not just because you are uncertain.
- TOP1_BRIDGE       — propose one bridge entity and do one branch retrieval. Use when you can name a likely bridge entity and have medium confidence in it.
- TOP2_BRIDGE       — two parallel bridge candidates, no verifier. Use when two plausible bridges compete and you have no cheap way to choose.
- TOP2_PRUNE        — top-2 bridges + verifier pruning. Use when there are multiple plausible bridges AND some are likely distractors.
- TOP2_PRUNE_REPAIR — top-2 bridges + verifier + local repair. Use when the one-shot answer looks unreliable AND the correct bridge itself may be wrong.

Default reasoning: if any positive bridge-gap signal is present, prefer escalation. Do NOT default to STOP just because routing is cheaper — STOP on a bridge-gap question gives a wrong answer, which is the worst outcome.

Output ONE JSON object of the form:
  {{"route": "<ROUTE_NAME>", "reason": "<one short sentence naming the signal you used>"}}
No other text. No code fences. No explanation."""


class LLMRouter:
    """LLM policy over structured state, with safety gate wrapping.

    Args:
        llm: shared LLMClient (to accumulate tokens with the pipeline)
        use_safety_gate: apply hard rules before/after LLM call
        enable_short_confident_stop: extra ablation gate (short+confident→STOP)
        budget_remaining_units: if set, disallow routes costlier than this
    """

    def __init__(self, llm: LLMClient,
                 use_safety_gate: bool = True,
                 enable_short_confident_stop: bool = False):
        self.llm = llm
        self.use_safety_gate = use_safety_gate
        self.enable_short_confident_stop = enable_short_confident_stop

    # ------ Safety gate ------
    def _hard_stop_rule(self, state: Dict[str, Any]) -> Optional[str]:
        """Return a forced route name if a hard rule applies, else None.

        Surgical minimal gate (2026-04-14): only abstain-escalate and yes_no STOP
        remain. The earlier `date → STOP` rule tuned well on MuSiQue but
        over-suppressed 2Wiki compositional date-comparison questions, and the
        narrowed `date && !has_bridge_cues` fix then regressed MuSiQue. Both
        versions were dataset-specific hand-tuning, so we remove the rule
        entirely and let the LLM router decide on date questions.
        """
        qtype = state["question_type"]
        if state["answer_confidence_proxy"] == 0.0:
            return TOP2_REPAIR
        if qtype == "yes_no":
            return STOP
        if self.enable_short_confident_stop:
            if (state["answer_len_words"] <= 2
                    and state["answer_confidence_proxy"] >= 0.7
                    and not state["has_bridge_cues"]):
                return STOP
        return None

    def _budget_downgrade(self, route: str, budget_units: Optional[int]) -> str:
        """Downgrade the route until it fits the remaining budget."""
        if budget_units is None:
            return route
        while _ROUTE_COST[route] > budget_units and route != STOP:
            route = _DOWNGRADE[route]
        return route

    # ------ LLM call ------
    def _call_llm(self, state: Dict[str, Any]) -> tuple:
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        prompt = _ROUTER_USER_TEMPLATE.format(state_json=state_json)
        # gpt-oss-120b is a reasoning model: hidden reasoning consumes budget
        # before visible content is emitted. 300 tokens is too tight and causes
        # silent empty responses. 2000 leaves headroom while still being cheap
        # (one router call per question).
        raw, tokens = self.llm.call(prompt, max_tokens=2000,
                                    system=_ROUTER_SYSTEM, temperature=0.0)
        route, reason = self._parse(raw, state)
        return route, reason, tokens, raw

    @staticmethod
    def _parse(raw: str, state: Dict[str, Any]) -> tuple:
        """Extract (route, reason) from the LLM response.

        On empty or unparseable output, fall back to rule_router(state) instead
        of silently defaulting to STOP. A STOP default biases the router toward
        the cheapest route precisely when the LLM failed — which is exactly the
        wrong inductive bias.
        """
        if raw:
            # Try JSON first (match outermost {...}, allow nested content)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    route = str(obj.get("route", "")).strip().upper()
                    reason = str(obj.get("reason", "")).strip()
                    if route in _ALL_ROUTES:
                        return route, reason or "llm_decision"
                except json.JSONDecodeError:
                    pass
            # Fallback: look for route name anywhere in text
            for r in _ALL_ROUTES:
                if r in raw.upper():
                    return r, "llm_substring_parse"
        # Empty/unparseable → delegate to rule_router (not STOP)
        rule_decision = rule_router(state)
        return rule_decision["route"], f"llm_fallback_to_rule({rule_decision['reason']})"

    # ------ Main entry ------
    def route(self, state: Dict[str, Any],
              budget_units: Optional[int] = None) -> Dict[str, Any]:
        """Return a full routing decision dict.

        Safety gate ordering:
          1. Hard rules (pre-LLM) — may short-circuit the LLM call entirely
          2. LLM call
          3. Budget downgrade (post-LLM)
        """
        tokens = 0
        raw_response = None
        # 1. Pre-LLM hard rule
        if self.use_safety_gate:
            forced = self._hard_stop_rule(state)
            if forced is not None:
                forced = self._budget_downgrade(forced, budget_units)
                return _decision(forced, "safety_gate_hard_rule", state,
                                 tokens=0, source="gate")

        # 2. LLM call
        route, reason, tokens, raw_response = self._call_llm(state)

        # 3. Post-LLM budget downgrade
        if self.use_safety_gate:
            downgraded = self._budget_downgrade(route, budget_units)
            if downgraded != route:
                reason = f"{reason}+budget_downgrade({route}→{downgraded})"
                route = downgraded

        return _decision(route, reason, state,
                         tokens=tokens, source="llm", raw=raw_response)


# =============================================================================
# Selective LLM router (3-route: STOP / TOP2_PRUNE / ABSTAIN)
# =============================================================================
#
# Motivation (N=300 route-wise analysis, 2026-04-15):
#   The 5-route llm_routed_gate_abv main method significantly underperformed
#   both abv_top2_prune and KiRAG on MuSiQue (ΔF1≈-0.035, p<0.05). Route-wise
#   breakdown showed the single failure mode: TOP1_BRIDGE was chosen on 42%
#   of MuSiQue / 20% of 2Wiki samples and achieved F1≈0.50, dramatically
#   below both the STOP bucket (F1≈0.62-0.81) and the TOP2_BRIDGE/TOP2_PRUNE
#   buckets when they were selected (F1≈0.85-1.00). These TOP1 samples align
#   with the oracle analysis's `stop_tied_failure` bucket — unrecoverable
#   questions where no bridge helps, that the router over-eagerly escalated.
#
# The Selective router recasts the routing problem as recoverability-aware
# selective control:
#   STOP       — one-shot answer is reliable, commit to it (no abstain)
#   TOP2_PRUNE — compositional bridge-gap, escalate to 2 parallel bridges
#                + verifier pruning (the only escalation that actually won
#                in the N=300 route-wise data)
#   ABSTAIN    — unrecoverable / low-confidence; pass through one-shot but
#                flag as abstain so selective-prediction metrics (AURC,
#                risk@coverage, coverage) credit the recoverability judgment
#
# TOP1_BRIDGE is removed from the main action space because (a) it was the
# single biggest loss bucket in the 5-route main method and (b) when 2Wiki
# route-wise showed bridging was working, it was TOP2_BRIDGE / TOP2_PRUNE
# doing the work, not TOP1.

_SELECTIVE_SYSTEM = """You are a routing policy for a multi-hop QA controller. \
Your job is to read a JSON state describing a question, a one-shot answer, \
and simple retrieval statistics, then output EXACTLY ONE discrete action \
from a three-action space.

You MUST NOT try to answer the question. Your job is recoverability-aware \
selective control: decide whether this question is already answered \
correctly, needs a targeted bridge-gap escalation, or is unrecoverable."""

_SELECTIVE_USER_TEMPLATE = """Given the following state, pick exactly one of three routes.

State:
{state_json}

The three routes are:

- STOP — The one-shot answer is already reliable. Choose this when the answer is a short, confident, directly-matching entity/date/number, and the question type is simple (single-hop factual, yes_no, date).

- TOP2_PRUNE — The question is a genuine compositional bridge-gap case that a targeted 2-bridge expansion can fix. Choose this ONLY when ALL of the following hold:
    * has_bridge_cues is true OR the question clearly names two entities to compose over
    * AND the one-shot answer either names a bridge entity (not the final target) or paraphrases without the target
    * AND you have medium-or-better confidence that a correct bridge expansion would recover the true answer
  Do NOT pick TOP2_PRUNE just because has_bridge_cues is true — many such samples are unrecoverable and bridging only wastes budget.

- ABSTAIN — The question is likely unrecoverable from the current retrieval. Choose this when ANY of the following hold:
    * The one-shot answer is "I don't know" or empty AND no clear bridge entity is available
    * The retrieval is flat (retrieval_score_gap < 0.05) AND the answer is long and hedging
    * The question asks for a specific attribute (birthplace, spouse, death date) but even the bridge entity is not confidently identifiable from the state
  ABSTAIN does NOT mean "I'm uncertain" — it means "no amount of bridging will fix this, prefer to pass through the one-shot answer and flag low confidence". It is a positive, specific recoverability judgment.

Key reasoning guidance:
- Prefer STOP when in doubt on easy-looking questions. STOP is cheap and correct on the vast majority of samples.
- Prefer ABSTAIN over TOP2_PRUNE when the one-shot answer is clearly bad AND you cannot name a specific bridge entity that would recover it. Escalating an unrecoverable sample to TOP2_PRUNE is the most expensive mistake — it costs tokens and rarely helps.
- Only pick TOP2_PRUNE when you can describe, in one sentence, what the bridge entity likely is and why it would close the compositional gap.

Output ONE JSON object of the form:
  {{"route": "<STOP|TOP2_PRUNE|ABSTAIN>", "reason": "<one short sentence naming the signal you used>"}}
No other text. No code fences. No explanation."""


class SelectiveLLMRouter:
    """3-route LLM router: {STOP, TOP2_PRUNE, ABSTAIN}.

    Keeps the same hard-rule safety gate structure as LLMRouter but over a
    narrower action space. The only hard rules retained are:
        abstain_confidence_zero → ABSTAIN  (was TOP2_REPAIR — flipped per
            N=300 finding that the REPAIR path wastes budget on samples the
            one-shot already gave up on)
        yes_no → STOP                      (cross-dataset stable)
    No date rule — the N=300 ablation showed it was dataset-specific.
    """

    def __init__(self, llm: LLMClient, use_safety_gate: bool = True):
        self.llm = llm
        self.use_safety_gate = use_safety_gate

    def _hard_rule(self, state: Dict[str, Any]) -> Optional[str]:
        if state["answer_confidence_proxy"] == 0.0:
            return ABSTAIN
        if state["question_type"] == "yes_no":
            return STOP
        return None

    def _call_llm(self, state: Dict[str, Any]) -> tuple:
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        prompt = _SELECTIVE_USER_TEMPLATE.format(state_json=state_json)
        raw, tokens = self.llm.call(prompt, max_tokens=2000,
                                    system=_SELECTIVE_SYSTEM, temperature=0.0)
        route, reason = self._parse(raw, state)
        return route, reason, tokens, raw

    @staticmethod
    def _parse(raw: str, state: Dict[str, Any]) -> tuple:
        if raw:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    route = str(obj.get("route", "")).strip().upper()
                    reason = str(obj.get("reason", "")).strip()
                    if route in _SELECTIVE_ROUTES:
                        return route, reason or "selective_llm_decision"
                except json.JSONDecodeError:
                    pass
            for r in _SELECTIVE_ROUTES:
                if r in raw.upper():
                    return r, "selective_substring_parse"
        # Unparseable → fall back to STOP (safest in the 3-route space:
        # ABSTAIN would silently inflate abstain rate on LLM errors, and
        # TOP2_PRUNE would waste budget. STOP is the recoverable default.)
        return STOP, "selective_llm_fallback_stop"

    def route(self, state: Dict[str, Any],
              budget_units: Optional[int] = None) -> Dict[str, Any]:
        if self.use_safety_gate:
            forced = self._hard_rule(state)
            if forced is not None:
                return _decision(forced, "selective_hard_rule", state,
                                 tokens=0, source="gate")
        route, reason, tokens, raw = self._call_llm(state)
        return _decision(route, reason, state,
                         tokens=tokens, source="selective_llm", raw=raw)


# =============================================================================
# Classifier-based router (2-route: STOP / TOP2_PRUNE)
# =============================================================================
#
# Motivation (N=300 analysis, 2026-04-16):
#   The zero-shot LLM routers (5-route and 3-route selective) both fail:
#   the LLM cannot reliably distinguish "bridgeable" from "unrecoverable"
#   questions. The Selective 3-route experiment showed ABSTAIN over-triggered
#   (40% MuSiQue) and the escalation to TOP2_PRUNE was under-used (5-8%).
#
#   This router replaces the LLM with a lightweight supervised classifier
#   trained on oracle labels (bridge_f1 > stop_f1). Two actions only:
#     STOP       — one-shot answer is already best or bridge won't help
#     TOP2_PRUNE — bridge retrieval has a good chance of improving the answer
#
#   No ABSTAIN (it was proven harmful in full-coverage evaluation).
#   No TOP1_BRIDGE (proven dominated in N=300 route-wise analysis).
#   No REPAIR as primary action (reserve for post-hoc diagnostic).

_ESCALATION_ROUTES = (STOP, TOP2_PRUNE)

QTYPE_MAP = {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}


class ClassifierRouter:
    """Recoverability-aware selective escalation: 2-route {STOP, TOP2_PRUNE}.

    Uses a pre-trained GBM/logistic classifier to decide whether the
    question is BRIDGEABLE or STOP-better, based on answer-side and
    retrieval-side features computed at routing time.

    Args:
        model_path: path to pickled classifier (output of
            train_recoverability_classifier.py)
        threshold: probability threshold for escalation (default from model)
    """

    def __init__(self, model_path: str, threshold: float = None):
        import pickle
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        self._clf = blob["model"]
        self._threshold = threshold or blob.get("threshold", 0.2)
        self._model_name = blob.get("model_name", "unknown")

    def _extract_features(self, state: Dict[str, Any]):
        """Extract the same 6 features used during training."""
        import numpy as np
        return np.array([[
            state["answer_confidence_proxy"],
            state["answer_len_words"],
            1.0 if state["has_bridge_cues"] else 0.0,
            state["retrieval_score_gap"],
            state["retrieval_score_top1"],
            QTYPE_MAP.get(state.get("question_type", "other"), 4),
        ]])

    def route(self, state: Dict[str, Any],
              budget_units: Optional[int] = None) -> Dict[str, Any]:
        # Hard rule: yes_no → STOP (cross-dataset stable)
        if state["question_type"] == "yes_no":
            return _decision(STOP, "clf_hard_yes_no", state,
                             tokens=0, source="clf_gate")

        X = self._extract_features(state)
        proba = float(self._clf.predict_proba(X)[0, 1])
        route = TOP2_PRUNE if proba >= self._threshold else STOP
        reason = f"clf_{self._model_name}_p={proba:.3f}_thr={self._threshold}"

        return _decision(route, reason, state,
                         tokens=0, source="classifier")


# =============================================================================
# Decision assembly
# =============================================================================

def _decision(route: str, reason: str, state: Dict[str, Any],
              tokens: int = 0, source: str = "rule",
              raw: str = None) -> Dict[str, Any]:
    cfg = _ROUTE_CONFIG[route]
    return {
        "route": route,
        "reason": reason,
        "source": source,
        "router_tokens": tokens,
        "raw_response": raw,
        "features": {
            "question_type": state["question_type"],
            "answer_confidence_proxy": state["answer_confidence_proxy"],
            "answer_len_words": state["answer_len_words"],
            "has_bridge_cues": state["has_bridge_cues"],
            "retrieval_score_gap": state["retrieval_score_gap"],
            "retrieval_score_top1": state["retrieval_score_top1"],
        },
        **cfg,
    }
