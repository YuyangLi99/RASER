"""Baseline systems for comparison with RASER.

Baselines:
1. naive_bm25: BM25 retrieve top-k, feed to LLM, answer directly (no agent loop)
2. no_harness: RASER agent but action harness always approves (no gating)
3. no_verifier: RASER agent but strong_verify is disabled
4. graph_only: RASER agent but text_targeted_retrieve is disabled
5. text_only: RASER agent but graph tools are disabled
"""

import json
import os
import time
from typing import Dict, List, Any

from openai import OpenAI
from src.tools.text_tools import TextRetriever
from src.tools.graph_tools import GraphRetriever
from src.tools.llm_utils import call_chat
from src.eval.answer_normalizer import normalize_prediction


class NaiveBM25RAG:
    """Baseline: text-only retrieve top-k chunks, feed to LLM, answer in one shot.

    The class is named "BM25" for backwards compatibility but now supports any retrieval mode
    via retriever_mode={"bm25","dense","hybrid"}.
    """

    def __init__(self, processed_dir: str, top_k: int = 10,
                 api_key: str = None, model: str = "gpt-oss-120b",
                 retriever_mode: str = "bm25", encoder_name: str = None, encoder=None):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.top_k = top_k
        self.retriever_mode = retriever_mode
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("LLM_BASE_URL")
                      or ""),
        )
        self.model = model
        self.total_tokens = 0

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        chunks = self.text_retriever.retrieve(question, top_k=self.top_k, question_id=question_id)
        context = "\n\n".join(f"[{c['title']}]: {c['text']}" for c in chunks)

        prompt = f"""Answer the following question concisely based on the given context.
If you cannot find the answer, reply "I don't know".

Context:
{context}

Question: {question}

Answer (be concise, just the answer):"""

        answer, tokens = call_chat(self.client, self.model,
                                   [{"role": "user", "content": prompt}],
                                   max_tokens=800, temperature=0.0)
        self.total_tokens += tokens

        return {
            "question_id": question_id,
            "question": question,
            "status": "answer" if answer and answer.lower() != "i don't know" else "abstain",
            "answer": normalize_prediction(answer, question) if answer else None,
            "answer_raw": answer,
            "text_evidence": [{"evidence_id": c["chunk_id"], "content": c["text"],
                               "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                               "metadata": {"title": c["title"]}}
                              for c in chunks[:5]],
            "graph_evidence": [],
            "action_trace": [{"step": 1, "final_action": "naive_retrieve_and_answer",
                              "note": f"BM25 top-{self.top_k}"}],
            "budget_used": {"steps_used": 1, "tool_calls_used": 1,
                            "tokens_used": tokens, "verifications_used": 0},
            "planner_tokens": tokens,
            "verifier_tokens": 0,
        }


class NaiveGraphTextRAG:
    """Baseline: graph + text retrieve, feed to LLM, answer in one shot.

    Uses retriever_mode={"bm25","dense","hybrid"} for both text and graph retrievers.
    """

    def __init__(self, processed_dir: str, top_k: int = 5,
                 api_key: str = None, model: str = "gpt-oss-120b",
                 retriever_mode: str = "bm25", encoder_name: str = None, encoder=None):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.graph_retriever = GraphRetriever(processed_dir, mode=retriever_mode,
                                              encoder=encoder, encoder_name=encoder_name)
        self.top_k = top_k
        self.retriever_mode = retriever_mode
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("LLM_BASE_URL")
                      or ""),
        )
        self.model = model
        self.total_tokens = 0

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        text_chunks = self.text_retriever.retrieve(question, top_k=self.top_k, question_id=question_id)
        graph_nodes = self.graph_retriever.expand_cheap(question, question_id, top_k=self.top_k)
        triples = self.graph_retriever.get_subgraph_triples(
            question_id, [n["node"]["node_id"] for n in graph_nodes])

        text_ctx = "\n".join(f"- [{c['title']}]: {c['text'][:200]}" for c in text_chunks)
        graph_ctx = "\n".join(f"- {n['linearized']}" for n in graph_nodes[:5])
        triple_ctx = "\n".join(f"- {t}" for t in triples[:10])

        prompt = f"""Answer the following question concisely based on graph and text evidence.
If you cannot find the answer, reply "I don't know".

## Graph Evidence
{graph_ctx}

## Graph Triples
{triple_ctx}

## Text Evidence
{text_ctx}

Question: {question}

Answer (be concise, just the answer):"""

        answer, tokens = call_chat(self.client, self.model,
                                   [{"role": "user", "content": prompt}],
                                   max_tokens=800, temperature=0.0)
        self.total_tokens += tokens

        return {
            "question_id": question_id,
            "question": question,
            "status": "answer" if answer and answer.lower() != "i don't know" else "abstain",
            "answer": normalize_prediction(answer, question) if answer else None,
            "answer_raw": answer,
            "text_evidence": [{"evidence_id": c["chunk_id"], "content": c["text"],
                               "score": c["bm25_score"], "metadata": {"title": c["title"]}}
                              for c in text_chunks[:5]],
            "graph_evidence": [{"evidence_id": n["node"]["node_id"], "content": n["linearized"],
                                "score": n.get("bm25_score", 0), "metadata": {}}
                               for n in graph_nodes[:5]],
            "action_trace": [{"step": 1, "final_action": "naive_graph_text_retrieve",
                              "note": f"BM25 top-{self.top_k} graph+text"}],
            "budget_used": {"steps_used": 1, "tool_calls_used": 2,
                            "tokens_used": tokens, "verifications_used": 0},
            "planner_tokens": tokens,
            "verifier_tokens": 0,
        }


class BridgeConditionedRAG:
    """Proposed method: bridge-conditioned multi-round retrieval.

    Round 1: retrieve top-k for original question.
    Bridge extraction (LLM): "What is the answer to the FIRST sub-question?"
        — extracts a literal answer string from round-1 chunks.
    Round 2..N: augmented_query = original_question + " " + concatenated_bridge_answers.
        Retrieve again. Stop early if Jaccard(round_k_chunks, round_{k-1}_chunks) > stop_threshold.
    Synthesis (LLM): final answer over union of all rounds' chunks.

    LLM calls per question: 1 (bridge extract) per non-final round + 1 (synthesize) = 2..N+1.

    The class is the empirical realization of the oracle bridge experiment: we showed that
    appending the gold answer string lifts MuSiQue 4hop later-recall from 0.63 to 0.88.
    Here we replace the oracle with an LLM-extracted bridge to test how close we can get.
    """

    def __init__(self, processed_dir: str, top_k: int = 10,
                 max_rounds: int = 2, stop_jaccard: float = 0.7,
                 api_key: str = None, model: str = "gpt-oss-120b",
                 retriever_mode: str = "dense", encoder_name: str = "nomic-v1.5",
                 encoder=None):
        self.text_retriever = TextRetriever(processed_dir, mode=retriever_mode,
                                            encoder=encoder, encoder_name=encoder_name)
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.stop_jaccard = stop_jaccard
        self.retriever_mode = retriever_mode
        self.encoder_name = encoder_name
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("LLM_API_KEY")
                     or ""),
            base_url=(_os.environ.get("LLM_BASE_URL")
                      or ""),
        )
        self.model = model
        self.total_tokens = 0

    def _llm_call(self, prompt: str, max_tokens: int = 800, prefer_short: bool = False) -> tuple:
        content, tokens = call_chat(self.client, self.model,
                                    [{"role": "user", "content": prompt}],
                                    max_tokens=max_tokens, temperature=0.0)
        if prefer_short and content:
            first_line = content.splitlines()[0].strip()
            if len(first_line) <= 80:
                content = first_line
        return content, tokens

    def _extract_bridge(self, question: str, chunks: List[Dict], prior_bridges: List[str]) -> tuple:
        """LLM call: extract the answer to the next sub-question from the current chunks."""
        ctx = "\n\n".join(f"[{c['title']}]: {c['text']}" for c in chunks[:8])
        prior = ""
        if prior_bridges:
            prior = "\n\nFacts already found:\n" + "\n".join(f"- {b}" for b in prior_bridges)
        prompt = f"""You are decomposing a multi-hop question. Read the passages and extract ONE intermediate fact that helps answer the question. The fact MUST be a short answer string (entity name, date, place — at most 5 words). NOT a sentence. NOT an explanation. NO reasoning.

Examples of valid replies: "Tracy McConnell" / "1973" / "Mississippi River" / "DONE"
Examples of INVALID replies: "The answer is X because..." / multi-line essays.

Passages:
{ctx}{prior}

Question: {question}

Reply with EXACTLY one short fact (≤5 words) on a single line, or "DONE" if the question can be answered without more retrieval.
Fact:"""
        return self._llm_call(prompt, max_tokens=1500, prefer_short=True)

    def _synthesize(self, question: str, all_chunks: List[Dict], bridges: List[str] = None) -> tuple:
        ctx = "\n\n".join(f"[{c['title']}]: {c['text']}" for c in all_chunks)
        bridge_block = ""
        if bridges:
            uniq = []
            for b in bridges:
                if b and b not in uniq:
                    uniq.append(b)
            if uniq:
                bridge_block = "\n\nIntermediate facts already established by decomposition (treat these as authoritative bridge entities):\n" + "\n".join(f"- {b}" for b in uniq) + "\n"
        prompt = f"""Answer the following question concisely based on the given context.
If you cannot find the answer, reply "I don't know".
{bridge_block}
Context:
{ctx}

Question: {question}

Answer (be concise, just the answer):"""
        return self._llm_call(prompt, max_tokens=800)

    def run(self, question_id: str, question: str, **kwargs) -> Dict[str, Any]:
        rounds_evidence = []
        bridges = []
        round_chunk_ids = []
        total_tokens = 0

        # Round 1
        chunks = self.text_retriever.retrieve(question, top_k=self.top_k, question_id=question_id)
        rounds_evidence.append(chunks)
        round_chunk_ids.append({c["chunk_id"] for c in chunks})

        # Subsequent rounds: extract bridge, augment, re-retrieve
        for r in range(1, self.max_rounds):
            bridge, btok = self._extract_bridge(question, chunks, bridges)
            total_tokens += btok
            if not bridge or bridge.upper().startswith("DONE"):
                break
            bridges.append(bridge)
            aug_query = question + " " + " ".join(bridges)
            new_chunks = self.text_retriever.retrieve(aug_query, top_k=self.top_k, question_id=question_id)
            new_ids = {c["chunk_id"] for c in new_chunks}
            # Information-gain stop: if too much overlap with previous round, stop
            overlap = len(new_ids & round_chunk_ids[-1]) / max(1, len(new_ids | round_chunk_ids[-1]))
            rounds_evidence.append(new_chunks)
            round_chunk_ids.append(new_ids)
            chunks = new_chunks
            if overlap > self.stop_jaccard:
                break

        # Build union (preserve order, dedupe by chunk_id, cap to budget)
        seen = set()
        union = []
        for round_chunks in rounds_evidence:
            for c in round_chunks:
                if c["chunk_id"] in seen:
                    continue
                seen.add(c["chunk_id"])
                union.append(c)
        # Cap to top_k * max_rounds for prompt budget
        union = union[: self.top_k * self.max_rounds]

        # Synthesize (pass bridges so the LLM treats them as authoritative)
        answer, stok = self._synthesize(question, union, bridges=bridges)
        total_tokens += stok
        self.total_tokens += total_tokens

        return {
            "question_id": question_id,
            "question": question,
            "status": "answer" if answer and answer.lower() != "i don't know" else "abstain",
            "answer": normalize_prediction(answer, question) if answer else None,
            "answer_raw": answer,
            "text_evidence": [{"evidence_id": c["chunk_id"], "content": c["text"],
                               "score": c.get("dense_score", c.get("bm25_score", 0.0)),
                               "metadata": {"title": c["title"]}}
                              for c in union[:8]],
            "graph_evidence": [],
            "action_trace": [{"step": 1, "final_action": "bridge_conditioned_retrieve",
                              "note": f"rounds={len(rounds_evidence)} bridges={bridges}"}],
            "budget_used": {"steps_used": len(rounds_evidence), "tool_calls_used": len(rounds_evidence),
                            "tokens_used": total_tokens, "verifications_used": 0},
            "planner_tokens": total_tokens,
            "verifier_tokens": 0,
            "bridges": bridges,
            "n_rounds": len(rounds_evidence),
        }


def _compute_abv_diagnostics(results: List[Dict]) -> Dict[str, Any]:
    """Extract ABV-Bridge diagnostic metrics from traces."""
    n = len(results)
    if n == 0:
        return {}
    triggered = 0
    n_proposals_total = 0
    n_kept = 0
    n_repaired = 0
    n_dropped = 0
    n_repair_attempted = 0
    n_with_bridges = 0
    trigger_reasons = {}
    route_distribution = {}

    for r in results:
        trace = r.get("action_trace", [])
        for step in trace:
            action = step.get("action", "")
            if action == "trigger_gate":
                if step.get("trigger"):
                    triggered += 1
                reason = step.get("reason", "")
                trigger_reasons[reason] = trigger_reasons.get(reason, 0) + 1
            elif action == "policy_gate":
                route = step.get("route", "unknown")
                route_distribution[route] = route_distribution.get(route, 0) + 1
                if route != "STOP":
                    triggered += 1
                reason = step.get("reason", "")
                trigger_reasons[reason] = trigger_reasons.get(reason, 0) + 1
            elif action == "PROPOSE_BRIDGES":
                n_proposals_total += step.get("n_proposals", 0)
            elif action == "PRUNE_OR_REPAIR":
                n_kept += step.get("kept", 0)
                n_repaired += step.get("repair", 0)
                n_dropped += step.get("dropped", 0)
            elif action == "LOCAL_REPAIR":
                n_repair_attempted += 1
            elif action == "ANSWER":
                if step.get("bridges_used"):
                    n_with_bridges += 1

    diag = {
        "trigger_rate": triggered / n,
        "trigger_reasons": trigger_reasons,
        "avg_proposals": n_proposals_total / max(1, triggered),
        "total_kept": n_kept,
        "total_repair_candidates": n_repaired,
        "total_dropped": n_dropped,
        "repair_attempted": n_repair_attempted,
        "answers_with_bridges": n_with_bridges,
        "bridge_usage_rate": n_with_bridges / n,
    }
    if route_distribution:
        diag["route_distribution"] = route_distribution
    return diag


def run_baseline(baseline_name: str, processed_dir: str, n: int = None,
                 output_dir: str = "outputs/traces",
                 retriever_mode: str = "bm25", encoder_name: str = None,
                 encoder=None, label_suffix: str = "",
                 max_rounds: int = 2, stop_jaccard: float = 0.7,
                 filter_question_type: str = None,
                 filter_qids: set = None,
                 disable_trigger: bool = False,
                 disable_verifier: bool = False,
                 disable_repair: bool = False,
                 llm_model: str = None) -> Dict:
    """Run a baseline on all examples and save results.

    Args:
        baseline_name: "naive_bm25" (text-only) or "naive_graph_text" (text + graph)
        retriever_mode: "bm25" | "dense" | "hybrid"
        encoder_name: e.g. "nomic-v1.5", "gte-qwen2-7b"
        encoder: pre-loaded encoder (avoid re-loading 7B model across runs)
        label_suffix: appended to variant name in saved files (e.g. "_dense_qwen")
    """
    from src.eval.metrics import evaluate_batch

    graphed_file = os.path.join(processed_dir, "graphed.jsonl")
    examples = []
    with open(graphed_file) as f:
        for line in f:
            ex = json.loads(line)
            if filter_question_type and ex.get("question_type") != filter_question_type:
                continue
            if filter_qids is not None and ex.get("question_id") not in filter_qids:
                continue
            examples.append(ex)
            if n and len(examples) >= n:
                break

    dataset_name = examples[0].get("dataset_name", "unknown") if examples else "unknown"
    variant_label = baseline_name + label_suffix

    if baseline_name == "naive_bm25":
        _kw = {"model": llm_model} if llm_model else {}
        system = NaiveBM25RAG(processed_dir, retriever_mode=retriever_mode,
                              encoder_name=encoder_name, encoder=encoder, **_kw)
    elif baseline_name == "naive_graph_text":
        system = NaiveGraphTextRAG(processed_dir, retriever_mode=retriever_mode,
                                   encoder_name=encoder_name, encoder=encoder)
    elif baseline_name == "bridge_conditioned":
        system = BridgeConditionedRAG(processed_dir, retriever_mode=retriever_mode,
                                      encoder_name=encoder_name, encoder=encoder,
                                      max_rounds=max_rounds, stop_jaccard=stop_jaccard)
    elif baseline_name == "abv_bridge":
        from src.methods.abv_bridge import ABVBridgePipeline
        system = ABVBridgePipeline(processed_dir, retriever_mode=retriever_mode,
                                   encoder=encoder, encoder_name=encoder_name,
                                   top_k_bridges=max_rounds,
                                   disable_trigger=disable_trigger,
                                   disable_verifier=disable_verifier,
                                   disable_repair=disable_repair,
                                   llm_model=llm_model)
    elif baseline_name == "conditional_abv":
        from src.methods.abv_bridge import ConditionalABVBridgePipeline
        system = ConditionalABVBridgePipeline(processed_dir, retriever_mode=retriever_mode,
                                              encoder=encoder, encoder_name=encoder_name,
                                              llm_model=llm_model)
    elif baseline_name in ("rule_routed_abv", "llm_routed_abv",
                           "llm_routed_gate_abv", "llm_routed_gate_sc_abv",
                           "selective_abv"):
        from src.methods.abv_bridge import LLMRoutedABVBridgePipeline
        _ROUTER_MODE = {
            "rule_routed_abv":       "rule_router",
            "llm_routed_abv":        "llm_router",
            "llm_routed_gate_abv":   "llm_gate",
            "llm_routed_gate_sc_abv":"llm_gate_sc",
            "selective_abv":         "selective",
        }[baseline_name]
        system = LLMRoutedABVBridgePipeline(
            processed_dir, retriever_mode=retriever_mode,
            encoder=encoder, encoder_name=encoder_name,
            router_mode=_ROUTER_MODE,
            llm_model=llm_model,
        )
    elif baseline_name == "escalation_abv":
        from src.methods.abv_bridge import LLMRoutedABVBridgePipeline
        import os as _os
        _clf_dir = _os.environ.get("ESCALATION_CLF_DIR", "outputs/models")
        _clf_thr = float(_os.environ.get("ESCALATION_THR", "0.20"))
        _clf_path = _os.path.join(_clf_dir, "recoverability_clf_combined.pkl")
        system = LLMRoutedABVBridgePipeline(
            processed_dir, retriever_mode=retriever_mode,
            encoder=encoder, encoder_name=encoder_name,
            router_mode=f"classifier:{_clf_path}:{_clf_thr}",
            llm_model=llm_model,
        )
    elif baseline_name == "chain_rag":
        from src.methods.chain_rag import ChainRAG
        _kw = {"model": llm_model} if llm_model else {}
        system = ChainRAG(processed_dir, retriever_mode=retriever_mode,
                          encoder_name=encoder_name, encoder=encoder, **_kw)
    elif baseline_name == "chain_rag_faithful":
        from src.methods.chain_rag_faithful import ChainRAGFaithful
        _kw = {"model": llm_model} if llm_model else {}
        system = ChainRAGFaithful(processed_dir, retriever_mode=retriever_mode,
                                   encoder_name=encoder_name, encoder=encoder, **_kw)
    elif baseline_name == "prism_rag":
        from src.methods.prism_rag import PrismRAG
        _kw = {"model": llm_model} if llm_model else {}
        system = PrismRAG(processed_dir, retriever_mode=retriever_mode,
                          encoder_name=encoder_name, encoder=encoder,
                          max_iterations=max_rounds, **_kw)
    elif baseline_name == "kirag":
        from src.methods.kirag import KiRAG
        _kw = {"model": llm_model} if llm_model else {}
        system = KiRAG(processed_dir, retriever_mode=retriever_mode,
                       encoder_name=encoder_name, encoder=encoder,
                       max_iterations=max_rounds, **_kw)
    elif baseline_name == "kirag_faithful":
        from src.methods.kirag_faithful import KiRAGFaithful
        _kw = {"model": llm_model} if llm_model else {}
        system = KiRAGFaithful(processed_dir, retriever_mode=retriever_mode,
                                encoder_name=encoder_name, encoder=encoder,
                                max_iterations=max_rounds, **_kw)
    elif baseline_name == "adaptive_rag":
        from src.methods.adaptive_rag import AdaptiveRAG
        _kw = {"model": llm_model} if llm_model else {}
        system = AdaptiveRAG(processed_dir, retriever_mode=retriever_mode,
                             encoder_name=encoder_name, encoder=encoder,
                             max_ircot_iter=max_rounds, **_kw)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    print(f"\nRunning baseline {variant_label} on {len(examples)} {dataset_name} examples...")
    results = []
    total_time = 0
    for i, ex in enumerate(examples):
        start = time.time()
        result = system.run(question_id=ex["question_id"], question=ex["question"])
        elapsed = time.time() - start
        total_time += elapsed

        result["gold_answer"] = ex.get("answer", "")
        result["gold_supporting_evidence"] = ex.get("supporting_evidence", [])
        result["latency"] = elapsed
        results.append(result)

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            print(f"  [{i+1}/{len(examples)}] avg_latency={total_time/(i+1):.1f}s")

    report = evaluate_batch(results)
    report["variant"] = variant_label
    report["retriever_mode"] = retriever_mode
    report["encoder"] = encoder_name or ""
    report["dataset"] = dataset_name
    report["total_time"] = total_time
    report["avg_latency"] = total_time / len(results) if results else 0

    # ABV-Bridge diagnostic metrics
    if baseline_name in ("abv_bridge", "conditional_abv",
                         "rule_routed_abv", "llm_routed_abv",
                         "llm_routed_gate_abv", "llm_routed_gate_sc_abv",
                         "selective_abv", "escalation_abv"):
        report["abv_diagnostics"] = _compute_abv_diagnostics(results)

    os.makedirs(output_dir, exist_ok=True)
    traces_file = os.path.join(output_dir, f"{variant_label}_{dataset_name}_traces.jsonl")
    with open(traces_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    report_file = os.path.join(output_dir, f"{variant_label}_{dataset_name}_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Baseline Report: {variant_label} on {dataset_name}")
    print(f"{'='*60}")
    print(f"  Answer EM:     {report['ans_em']:.4f}")
    print(f"  Answer F1:     {report['ans_f1']:.4f}")
    print(f"  SP F1:         {report['sp_f1']:.4f}")
    print(f"  Avg Tokens:    {report['avg_tokens']:.0f}")
    print(f"  Avg Latency:   {report['avg_latency']:.1f}s")
    if "abv_diagnostics" in report:
        diag = report["abv_diagnostics"]
        print(f"  --- ABV Diagnostics ---")
        print(f"  Trigger Rate:  {diag['trigger_rate']:.2%}")
        print(f"  Trigger Reasons: {diag['trigger_reasons']}")
        print(f"  Avg Proposals: {diag['avg_proposals']:.1f}")
        print(f"  Branches: kept={diag['total_kept']} repair={diag['total_repair_candidates']} drop={diag['total_dropped']}")
        print(f"  Repair Attempted: {diag['repair_attempted']}")
        print(f"  Bridge Usage:  {diag['bridge_usage_rate']:.2%}")
        if "route_distribution" in diag:
            print(f"  Route Distribution: {diag['route_distribution']}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        choices=["naive_bm25", "naive_graph_text", "bridge_conditioned",
                                 "abv_bridge", "conditional_abv",
                                 "rule_routed_abv", "llm_routed_abv",
                                 "llm_routed_gate_abv", "llm_routed_gate_sc_abv",
                                 "selective_abv", "escalation_abv",
                                 "chain_rag", "chain_rag_faithful",
                                 "prism_rag", "kirag", "kirag_faithful",
                                 "adaptive_rag"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/traces")
    parser.add_argument("--retriever-mode", default="bm25", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--label-suffix", default="")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--stop-jaccard", type=float, default=0.7)
    parser.add_argument("--filter-question-type", default=None,
                        help="Only run examples whose question_type matches (e.g. 4hop)")
    parser.add_argument("--qid-file", default=None,
                        help="Path to a text file with one question_id per line; only "
                             "examples whose qid is in this set are evaluated. The "
                             "retrieval corpus still uses full graphed.jsonl (so the "
                             "precomputed dense index stays valid).")
    parser.add_argument("--disable-trigger", action="store_true",
                        help="ABV ablation: always trigger bridge (skip gate)")
    parser.add_argument("--disable-verifier", action="store_true",
                        help="ABV ablation: keep all branches (skip verifier)")
    parser.add_argument("--disable-repair", action="store_true",
                        help="ABV ablation: no local repair step")
    parser.add_argument("--llm-model", default=None,
                        help="Override LLM model (e.g. qwen3.5-397b, "
                             "minimax-m2.5-229b). Default: gpt-oss-120b")
    parser.add_argument("--base-url", default=None,
                        help="Override OpenAI-compat base URL (e.g. "
                             "http://haicn1704.localdomain:8002/v1 for self-hosted vLLM). "
                             "Default: KIT toolbox.")
    parser.add_argument("--api-key", default=None,
                        help="Override OpenAI-compat API key. "
                             "Default: KIT key embedded in baselines.")
    args = parser.parse_args()

    # Propagate base_url / api_key to the OpenAI clients constructed inside the
    # baselines via env vars (read by every OpenAI(...) call site as fallback).
    import os as _os
    if args.base_url:
        _os.environ["LLM_BASE_URL"] = args.base_url
    if args.api_key:
        _os.environ["LLM_API_KEY"] = args.api_key

    encoder = None
    if args.retriever_mode in ("dense", "hybrid"):
        from src.tools.encoders import get_encoder
        encoder = get_encoder("nomic", device="cuda")
        if args.encoder_name is None:
            args.encoder_name = encoder.name

    qid_set = None
    if args.qid_file:
        with open(args.qid_file) as f:
            qid_set = {l.strip() for l in f if l.strip()}
        print(f"[qid-file] filtering to {len(qid_set)} qids from {args.qid_file}")

    run_baseline(args.baseline, args.data_dir, args.n, args.output_dir,
                 retriever_mode=args.retriever_mode, encoder_name=args.encoder_name,
                 encoder=encoder, label_suffix=args.label_suffix,
                 max_rounds=args.max_rounds, stop_jaccard=args.stop_jaccard,
                 filter_question_type=args.filter_question_type,
                 filter_qids=qid_set,
                 disable_trigger=args.disable_trigger,
                 disable_verifier=args.disable_verifier,
                 disable_repair=args.disable_repair,
                 llm_model=args.llm_model)
