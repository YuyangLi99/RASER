"""ABV-Bridge Pipeline: Adaptive Branch-and-Verify Bridge Agent.

Flow:
  1. One-shot dense retrieval + one-shot answer
  2. Trigger gate: STOP or BRIDGE?
  3. If BRIDGE: top-k bridge proposals (structured JSON)
  4. Branch-conditioned retrieval per proposal
  5. Branch verifier: keep / drop / repair
  6. Optional one-step local repair
  7. Final synthesis from curated evidence + authoritative bridges

Conditional variant (ConditionalABVBridgePipeline):
  Replaces the binary trigger gate with a policy gate that routes each
  question to the cheapest-sufficient strategy: STOP / TOP1 / TOP2 /
  TOP2_PRUNE / TOP2_PRUNE_REPAIR.
"""

from typing import Dict, List, Any

from src.tools.text_tools import TextRetriever
from src.eval.answer_normalizer import normalize_prediction

from .llm_client import LLMClient
from .trigger_gate import trigger_gate
from .policy_gate import policy_gate, STOP
from .router import (
    LLMRouter, SelectiveLLMRouter, ClassifierRouter,
    rule_router, build_router_state,
    STOP as R_STOP, ABSTAIN as R_ABSTAIN,
)
from .bridge_proposer import BridgeProposer
from .branch_retriever import BranchRetriever
from .branch_verifier import BranchVerifier
from .local_repair import LocalRepair
from .final_synthesizer import FinalSynthesizer


class ABVBridgePipeline:
    """Adaptive Branch-and-Verify Bridge Agent for Multi-Hop QA.

    Args:
        processed_dir: path to processed data directory
        retriever_mode: "bm25" | "dense" | "hybrid"
        encoder: pre-loaded encoder for dense retrieval
        encoder_name: e.g. "nomic-v1.5"
        top_k_retrieve: chunks per retrieval call
        top_k_bridges: max bridge proposals
        max_repair_attempts: max local repair steps (0 or 1)
    """

    def __init__(self, processed_dir: str,
                 retriever_mode: str = "dense",
                 encoder=None, encoder_name: str = "nomic-v1.5",
                 top_k_retrieve: int = 10,
                 top_k_bridges: int = 2,
                 max_repair_attempts: int = 1,
                 disable_trigger: bool = False,
                 disable_verifier: bool = False,
                 disable_repair: bool = False,
                 llm_model: str = None):
        self.text_retriever = TextRetriever(
            processed_dir, mode=retriever_mode,
            encoder=encoder, encoder_name=encoder_name,
        )
        self.llm = LLMClient(model=llm_model)
        self.top_k = top_k_retrieve
        self.top_k_bridges = top_k_bridges
        self.max_repair = max_repair_attempts
        self.disable_trigger = disable_trigger
        self.disable_verifier = disable_verifier
        self.disable_repair = disable_repair

        # Sub-modules share one LLM client for token accounting
        self.proposer = BridgeProposer(llm=self.llm, top_k=top_k_bridges)
        self.branch_retriever = BranchRetriever(self.text_retriever, top_k=top_k_retrieve)
        self.verifier = BranchVerifier()
        self.repair = LocalRepair(llm=self.llm)
        self.synthesizer = FinalSynthesizer(llm=self.llm)

    def _one_shot_answer(self, question: str, chunks: List[Dict]) -> tuple:
        """Quick one-shot answer from retrieved chunks."""
        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in chunks[:10]
        )
        prompt = f"""Answer the following question concisely based on the given context.
If you cannot find the answer, reply "I don't know".

Context:
{ctx}

Question: {question}

Answer (be concise, just the answer):"""
        return self.llm.call(prompt, max_tokens=800)

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        """Run the full ABV-Bridge pipeline on one question.

        Returns a trace dict compatible with evaluate_batch().
        """
        trace = {
            "question_id": question_id,
            "question": question,
            "action_trace": [],
        }
        total_tokens = 0
        retrieval_calls = 0

        # === Step 1: One-shot retrieval + answer ===
        one_shot_chunks = self.text_retriever.retrieve(
            question, top_k=self.top_k, question_id=question_id
        )
        retrieval_calls += 1
        one_shot_ids = {c["chunk_id"] for c in one_shot_chunks}

        one_shot_answer, os_tok = self._one_shot_answer(question, one_shot_chunks)
        total_tokens += os_tok

        trace["action_trace"].append({
            "step": 1,
            "action": "one_shot",
            "answer": one_shot_answer,
            "tokens": os_tok,
        })

        # === Step 2: Trigger gate ===
        if self.disable_trigger:
            # Always trigger (ablation: skip gate, always bridge)
            gate = {"trigger": True, "reason": "trigger_disabled",
                    "hop_estimate": 0, "evidence_concentration": 0.0}
        else:
            gate = trigger_gate(question, one_shot_answer, one_shot_chunks, os_tok)
        trace["action_trace"].append({
            "step": 2,
            "action": "trigger_gate",
            "trigger": gate["trigger"],
            "reason": gate["reason"],
            "hop_estimate": gate.get("hop_estimate", 0),
        })

        if not gate["trigger"]:
            # Trust one-shot answer
            final_answer = one_shot_answer
            trace["action_trace"].append({
                "step": 3,
                "action": "STOP",
                "note": "trigger gate says confident",
            })
            return self._finalize(trace, final_answer, one_shot_chunks, [],
                                  [], total_tokens, retrieval_calls)

        # === Step 3: Bridge proposals ===
        proposals, prop_tok = self.proposer.propose(
            question, one_shot_chunks, current_answer=one_shot_answer
        )
        total_tokens += prop_tok

        trace["action_trace"].append({
            "step": 3,
            "action": "PROPOSE_BRIDGES",
            "n_proposals": len(proposals),
            "proposals": proposals,
            "tokens": prop_tok,
        })

        if not proposals:
            # No bridges found, fall back to one-shot
            return self._finalize(trace, one_shot_answer, one_shot_chunks, [],
                                  [], total_tokens, retrieval_calls)

        # === Step 4: Branch-conditioned retrieval ===
        branches = self.branch_retriever.retrieve_branches(
            question, question_id, proposals
        )
        retrieval_calls += len(branches)

        trace["action_trace"].append({
            "step": 4,
            "action": "RETRIEVE_BRANCHES",
            "n_branches": len(branches),
        })

        # === Step 5: Branch verification and pruning ===
        if self.disable_verifier:
            # Ablation: keep all branches, no scoring
            scored_branches = [{**b, "novelty": 0, "support": 0,
                                "info_gain": 0, "score": 1.0, "action": "keep"}
                               for b in branches]
        else:
            scored_branches = self.verifier.verify_branches(branches, one_shot_ids)

        kept = [b for b in scored_branches if b["action"] == "keep"]
        repair_candidates = [b for b in scored_branches if b["action"] == "repair"]
        dropped = [b for b in scored_branches if b["action"] == "drop"]

        trace["action_trace"].append({
            "step": 5,
            "action": "PRUNE_OR_REPAIR",
            "kept": len(kept),
            "repair": len(repair_candidates),
            "dropped": len(dropped),
            "branch_scores": [
                {"entity": b["proposal"]["bridge_entity"],
                 "score": round(b.get("score", 0), 3),
                 "action": b["action"],
                 "novelty": round(b.get("novelty", 0), 3),
                 "support": round(b.get("support", 0), 3)}
                for b in scored_branches
            ],
        })

        # === Step 6: Local repair (at most 1 attempt) ===
        repaired_branches = []
        if not kept and repair_candidates and self.max_repair > 0 and not self.disable_repair:
            # Try repairing the best repair candidate
            best_repair = repair_candidates[0]
            repaired_prop, repair_tok = self.repair.repair(
                question, best_repair["proposal"], one_shot_chunks
            )
            total_tokens += repair_tok

            if repaired_prop:
                # Re-retrieve with repaired entity
                repaired_branch_list = self.branch_retriever.retrieve_branches(
                    question, question_id, [repaired_prop]
                )
                retrieval_calls += 1
                if repaired_branch_list:
                    repaired_branches = repaired_branch_list
                    trace["action_trace"].append({
                        "step": 6,
                        "action": "LOCAL_REPAIR",
                        "old_entity": best_repair["proposal"]["bridge_entity"],
                        "new_entity": repaired_prop["bridge_entity"],
                        "tokens": repair_tok,
                    })

        # === Step 7: Final synthesis ===
        # Collect surviving branch chunks
        surviving_branches = kept or repaired_branches
        branch_chunks = []
        bridge_entities = []
        for b in surviving_branches:
            branch_chunks.extend(b["chunks"])
            bridge_entities.append(b["proposal"]["bridge_entity"])

        # If no branches survived at all, use one-shot evidence only
        if not surviving_branches:
            final_answer = one_shot_answer
            trace["action_trace"].append({
                "step": 7,
                "action": "ANSWER",
                "note": "no surviving branches, using one-shot answer",
            })
        else:
            final_answer, synth_tok = self.synthesizer.synthesize(
                question, one_shot_chunks, branch_chunks, bridge_entities
            )
            total_tokens += synth_tok
            trace["action_trace"].append({
                "step": 7,
                "action": "ANSWER",
                "bridges_used": bridge_entities,
                "tokens": synth_tok,
            })

        return self._finalize(trace, final_answer, one_shot_chunks,
                              branch_chunks, bridge_entities,
                              total_tokens, retrieval_calls)

    def _finalize(self, trace: Dict, answer: str,
                  one_shot_chunks: List[Dict],
                  branch_chunks: List[Dict],
                  bridges: List[str],
                  total_tokens: int,
                  retrieval_calls: int) -> Dict[str, Any]:
        """Build the final result dict."""
        normalized = normalize_prediction(answer, trace["question"]) if answer else None
        is_abstain = (not answer or answer.lower() in ("i don't know", ""))

        # Merge evidence for trace
        seen = set()
        all_evidence = []
        for c in (branch_chunks + one_shot_chunks):
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                all_evidence.append(c)

        trace.update({
            "status": "abstain" if is_abstain else "answer",
            "answer": normalized,
            "answer_raw": answer,
            "text_evidence": [
                {"evidence_id": c["chunk_id"], "content": c["text"],
                 "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                 "metadata": {"title": c.get("title", "")}}
                for c in all_evidence[:10]
            ],
            "graph_evidence": [],
            "bridges": bridges,
            "budget_used": {
                "steps_used": len(trace["action_trace"]),
                "tool_calls_used": retrieval_calls,
                "tokens_used": total_tokens,
                "verifications_used": 0,
            },
            "planner_tokens": total_tokens,
            "verifier_tokens": 0,
        })
        return trace


class ConditionalABVBridgePipeline:
    """Conditional ABV-Bridge: routes each question to the cheapest strategy.

    Uses a policy gate instead of a binary trigger to decide:
    STOP / TOP1_BRIDGE / TOP2_BRIDGE / TOP2_PRUNE / TOP2_PRUNE_REPAIR.
    """

    def __init__(self, processed_dir: str,
                 retriever_mode: str = "dense",
                 encoder=None, encoder_name: str = "nomic-v1.5",
                 top_k_retrieve: int = 10,
                 llm_model: str = None):
        self.text_retriever = TextRetriever(
            processed_dir, mode=retriever_mode,
            encoder=encoder, encoder_name=encoder_name,
        )
        self.llm = LLMClient(model=llm_model)
        self.top_k = top_k_retrieve

        self.proposer = BridgeProposer(llm=self.llm, top_k=2)
        self.branch_retriever = BranchRetriever(self.text_retriever, top_k=top_k_retrieve)
        self.verifier = BranchVerifier()
        self.repair = LocalRepair(llm=self.llm)
        self.synthesizer = FinalSynthesizer(llm=self.llm)

    def _one_shot_answer(self, question: str, chunks: List[Dict]) -> tuple:
        ctx = "\n\n".join(
            f"[{c.get('title', '')}]: {c.get('text', '')}"
            for c in chunks[:10]
        )
        prompt = f"""Answer the following question concisely based on the given context.
If you cannot find the answer, reply "I don't know".

Context:
{ctx}

Question: {question}

Answer (be concise, just the answer):"""
        return self.llm.call(prompt, max_tokens=800)

    def _decide_route(self, question: str, one_shot_answer: str,
                      one_shot_chunks: List[Dict]) -> Dict[str, Any]:
        """Routing decision (override in subclasses to swap the policy)."""
        return policy_gate(question, one_shot_answer, one_shot_chunks)

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        trace = {
            "question_id": question_id,
            "question": question,
            "action_trace": [],
        }
        total_tokens = 0
        retrieval_calls = 0

        # === Step 1: One-shot retrieval + answer ===
        one_shot_chunks = self.text_retriever.retrieve(
            question, top_k=self.top_k, question_id=question_id
        )
        retrieval_calls += 1
        one_shot_ids = {c["chunk_id"] for c in one_shot_chunks}

        one_shot_answer, os_tok = self._one_shot_answer(question, one_shot_chunks)
        total_tokens += os_tok

        trace["action_trace"].append({
            "step": 1, "action": "one_shot",
            "answer": one_shot_answer, "tokens": os_tok,
        })

        # === Step 2: Routing decision (overridable by subclasses) ===
        route_decision = self._decide_route(question, one_shot_answer, one_shot_chunks)
        total_tokens += route_decision.get("router_tokens", 0)
        route = route_decision["route"]
        top_k_bridges = route_decision["top_k_bridges"]
        use_verifier = route_decision["use_verifier"]
        use_repair = route_decision["use_repair"]

        trace["action_trace"].append({
            "step": 2, "action": "policy_gate",
            "route": route,
            "reason": route_decision["reason"],
            "features": route_decision["features"],
        })

        if route == STOP:
            trace["action_trace"].append({
                "step": 3, "action": "STOP",
                "note": "policy gate: one-shot sufficient",
            })
            return self._finalize(trace, one_shot_answer, one_shot_chunks,
                                  [], [], total_tokens, retrieval_calls)

        if route == R_ABSTAIN:
            # Selective router: the sample is judged unrecoverable. Pass the
            # one-shot answer through as the final answer (it may still be
            # correct by luck) but force the trace status to "abstain" so
            # selective-prediction metrics (AURC, coverage, risk@k) credit
            # the recoverability judgment even when the one-shot happened to
            # score a non-zero F1.
            trace["action_trace"].append({
                "step": 3, "action": "ABSTAIN",
                "note": "selective router: unrecoverable, passing through one-shot",
            })
            final = self._finalize(trace, one_shot_answer, one_shot_chunks,
                                   [], [], total_tokens, retrieval_calls)
            final["status"] = "abstain"
            return final

        # === Step 3: Bridge proposals (capped by route) ===
        self.proposer.top_k = top_k_bridges
        proposals, prop_tok = self.proposer.propose(
            question, one_shot_chunks, current_answer=one_shot_answer
        )
        total_tokens += prop_tok

        trace["action_trace"].append({
            "step": 3, "action": "PROPOSE_BRIDGES",
            "n_proposals": len(proposals), "tokens": prop_tok,
            "proposals": proposals,
        })

        if not proposals:
            return self._finalize(trace, one_shot_answer, one_shot_chunks,
                                  [], [], total_tokens, retrieval_calls)

        # === Step 4: Branch-conditioned retrieval ===
        branches = self.branch_retriever.retrieve_branches(
            question, question_id, proposals
        )
        retrieval_calls += len(branches)

        trace["action_trace"].append({
            "step": 4, "action": "RETRIEVE_BRANCHES",
            "n_branches": len(branches),
        })

        # === Step 5: Branch verification (if route says so) ===
        if use_verifier:
            scored_branches = self.verifier.verify_branches(branches, one_shot_ids)
        else:
            scored_branches = [{**b, "novelty": 0, "support": 0,
                                "info_gain": 0, "score": 1.0, "action": "keep"}
                               for b in branches]

        kept = [b for b in scored_branches if b["action"] == "keep"]
        repair_candidates = [b for b in scored_branches if b["action"] == "repair"]
        dropped = [b for b in scored_branches if b["action"] == "drop"]

        trace["action_trace"].append({
            "step": 5, "action": "PRUNE_OR_REPAIR",
            "kept": len(kept), "repair": len(repair_candidates),
            "dropped": len(dropped),
            "branch_scores": [
                {"entity": b["proposal"]["bridge_entity"],
                 "score": round(b.get("score", 0), 3),
                 "action": b["action"]}
                for b in scored_branches
            ],
        })

        # === Step 6: Local repair (only if route permits AND no kept branches) ===
        repaired_branches = []
        if use_repair and not kept and repair_candidates:
            best_repair = repair_candidates[0]
            repaired_prop, repair_tok = self.repair.repair(
                question, best_repair["proposal"], one_shot_chunks
            )
            total_tokens += repair_tok
            if repaired_prop:
                repaired_branch_list = self.branch_retriever.retrieve_branches(
                    question, question_id, [repaired_prop]
                )
                retrieval_calls += 1
                if repaired_branch_list:
                    repaired_branches = repaired_branch_list
                    trace["action_trace"].append({
                        "step": 6, "action": "LOCAL_REPAIR",
                        "old_entity": best_repair["proposal"]["bridge_entity"],
                        "new_entity": repaired_prop["bridge_entity"],
                        "tokens": repair_tok,
                    })

        # === Step 7: Final synthesis ===
        surviving_branches = kept or repaired_branches
        branch_chunks = []
        bridge_entities = []
        for b in surviving_branches:
            branch_chunks.extend(b["chunks"])
            bridge_entities.append(b["proposal"]["bridge_entity"])

        if not surviving_branches:
            final_answer = one_shot_answer
            trace["action_trace"].append({
                "step": 7, "action": "ANSWER",
                "note": "no surviving branches, using one-shot",
            })
        else:
            final_answer, synth_tok = self.synthesizer.synthesize(
                question, one_shot_chunks, branch_chunks, bridge_entities
            )
            total_tokens += synth_tok
            trace["action_trace"].append({
                "step": 7, "action": "ANSWER",
                "bridges_used": bridge_entities, "tokens": synth_tok,
            })

        return self._finalize(trace, final_answer, one_shot_chunks,
                              branch_chunks, bridge_entities,
                              total_tokens, retrieval_calls)

    def _finalize(self, trace: Dict, answer: str,
                  one_shot_chunks: List[Dict],
                  branch_chunks: List[Dict],
                  bridges: List[str],
                  total_tokens: int,
                  retrieval_calls: int) -> Dict[str, Any]:
        normalized = normalize_prediction(answer, trace["question"]) if answer else None
        is_abstain = (not answer or answer.lower() in ("i don't know", ""))

        seen = set()
        all_evidence = []
        for c in (branch_chunks + one_shot_chunks):
            cid = c.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                all_evidence.append(c)

        trace.update({
            "status": "abstain" if is_abstain else "answer",
            "answer": normalized,
            "answer_raw": answer,
            "text_evidence": [
                {"evidence_id": c["chunk_id"], "content": c["text"],
                 "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                 "metadata": {"title": c.get("title", "")}}
                for c in all_evidence[:10]
            ],
            "graph_evidence": [],
            "bridges": bridges,
            "budget_used": {
                "steps_used": len(trace["action_trace"]),
                "tool_calls_used": retrieval_calls,
                "tokens_used": total_tokens,
                "verifications_used": 0,
            },
            "planner_tokens": total_tokens,
            "verifier_tokens": 0,
        })
        return trace


class LLMRoutedABVBridgePipeline(ConditionalABVBridgePipeline):
    """Final method: Conditional ABV-Bridge with LLM routing + safety gate.

    Variants (selected via `router_mode`):
        rule_router  — hand-written rules (diagnostic baseline, no LLM call)
        llm_router   — LLM router over structured state, no safety gate
        llm_gate     — LLM router + hard rule gate (main method)
        llm_gate_sc  — LLM router + hard rule gate + short_confident_stop (ablation)

    The "safety gate" enforces strong empirical rules (date/yes_no → STOP,
    budget downgrade) around the LLM router. The ablation variant adds the
    weaker "short confident answer → STOP" rule.
    """

    def __init__(self, processed_dir: str,
                 retriever_mode: str = "dense",
                 encoder=None, encoder_name: str = "nomic-v1.5",
                 top_k_retrieve: int = 10,
                 router_mode: str = "llm_gate",
                 llm_model: str = None):
        super().__init__(processed_dir=processed_dir,
                         retriever_mode=retriever_mode,
                         encoder=encoder, encoder_name=encoder_name,
                         top_k_retrieve=top_k_retrieve,
                         llm_model=llm_model)
        self.router_mode = router_mode
        if router_mode == "rule_router":
            self._llm_router = None
        elif router_mode == "llm_router":
            self._llm_router = LLMRouter(self.llm, use_safety_gate=False,
                                         enable_short_confident_stop=False)
        elif router_mode == "llm_gate":
            self._llm_router = LLMRouter(self.llm, use_safety_gate=True,
                                         enable_short_confident_stop=False)
        elif router_mode == "llm_gate_sc":
            self._llm_router = LLMRouter(self.llm, use_safety_gate=True,
                                         enable_short_confident_stop=True)
        elif router_mode == "selective":
            self._llm_router = SelectiveLLMRouter(self.llm, use_safety_gate=True)
        elif router_mode.startswith("classifier:"):
            # Format: "classifier:/path/to/model.pkl" or "classifier:/path:threshold"
            parts = router_mode.split(":", 2)
            model_path = parts[1]
            threshold = float(parts[2]) if len(parts) > 2 else None
            self._llm_router = ClassifierRouter(model_path, threshold=threshold)
        else:
            raise ValueError(f"Unknown router_mode: {router_mode}")

    def _decide_route(self, question: str, one_shot_answer: str,
                      one_shot_chunks: List[Dict]) -> Dict[str, Any]:
        state = build_router_state(question, one_shot_answer, one_shot_chunks)
        if self.router_mode == "rule_router":
            return rule_router(state)
        return self._llm_router.route(state)
