"""3-route feasibility study (DRY-RUN, no LLM calls).

Reads existing traces for STOP / PRUNE / ITER_TRIPLE (KiRAG-style) /
DECOMP (ChainRAG-style / Self-Ask-style) across 3 readers x 3 datasets.

Computes:
  1. Per-question (F1, tokens) for all 4 methods
  2. Oracle ceilings for various route subsets and cost margins
  3. Held-out 3-class GBM (5-fold CV) router F1 + tokens + route distribution
  4. Adoption-criteria check

NOTE on naming: per the user's review, what we previously called
"simplified KiRAG" / "simplified ChainRAG" are more accurately described
as IRCoT-style / Self-Ask-style baselines. We use generic labels
ITER_TRIPLE and DECOMP throughout this analysis.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.methods.abv_bridge.trigger_gate import has_bridge_cues
from src.eval.answer_normalizer import classify_question_type


# ── Feature extraction (mirrors deployed RASER's build_router_state) ─────
def _confidence(answer: str, qtype: str) -> float:
    if not answer or answer.strip().lower() in ("i don't know", ""):
        return 0.0
    if qtype == "yes_no" and answer.strip().lower() in ("yes", "no"):
        return 0.9
    if qtype in ("date", "count"):
        return 0.8
    w = answer.strip().split()
    if len(w) <= 2: return 0.6
    if len(w) >= 10: return 0.3
    return 0.7


QTYPE_MAP = {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}


def features_from_stop_rec(rec):
    q = rec.get("question", "")
    a = rec.get("answer") or rec.get("answer_raw") or ""
    qt = classify_question_type(q)
    chunks = rec.get("text_evidence") or []
    s = [float(c.get("score") or 0) for c in chunks[:10]]
    t1 = s[0] if s else 0
    t5 = s[4] if len(s) >= 5 else 0
    return [
        round(_confidence(a, qt), 3),
        len(a.split()),
        1.0 if has_bridge_cues(q) else 0.0,
        round(t1 - t5, 4),
        round(t1, 4),
        QTYPE_MAP.get(qt, 4),
    ]


# ── Data loaders ─────────────────────────────────────────────────────────
READER_TRACES = {
    "GPT-OSS-120B": {
        "MuSiQue":  ("outputs/traces_holdout/musique/naive_bm25_dense_nomic_holdout_musique_traces.jsonl",
                     "outputs/traces_holdout/musique/abv_bridge_dense_nomic_holdout_musique_traces.jsonl",
                     "outputs/traces_holdout/musique/kirag_dense_nomic_holdout_musique_traces.jsonl",
                     "outputs/traces_holdout/musique/chain_rag_dense_nomic_holdout_musique_traces.jsonl"),
        "2Wiki":    ("outputs/traces_holdout/2wikimultihopqa/naive_bm25_dense_nomic_holdout_2wikimultihopqa_traces.jsonl",
                     "outputs/traces_holdout/2wikimultihopqa/abv_bridge_dense_nomic_holdout_2wikimultihopqa_traces.jsonl",
                     "outputs/traces_holdout/2wikimultihopqa/kirag_dense_nomic_holdout_2wikimultihopqa_traces.jsonl",
                     "outputs/traces_holdout/2wikimultihopqa/chain_rag_dense_nomic_holdout_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/traces_holdout/hotpotqa/naive_bm25_dense_nomic_holdout_hotpotqa_traces.jsonl",
                     "outputs/traces_holdout/hotpotqa/abv_bridge_dense_nomic_holdout_hotpotqa_traces.jsonl",
                     "outputs/traces_holdout/hotpotqa/kirag_dense_nomic_holdout_hotpotqa_traces.jsonl",
                     "outputs/traces_holdout/hotpotqa/chain_rag_dense_nomic_holdout_hotpotqa_traces.jsonl"),
    },
    "Llama-3-8B": {
        "MuSiQue":  ("outputs/sweep_llama3/musique/naive_bm25_llama3_musique_traces.jsonl",
                     "outputs/sweep_llama3/musique/abv_bridge_llama3_musique_traces.jsonl",
                     "outputs/sweep_llama3/musique/kirag_llama3_musique_traces.jsonl",
                     "outputs/sweep_llama3/musique/chain_rag_llama3_musique_traces.jsonl"),
        "2Wiki":    ("outputs/sweep_llama3/2wiki/naive_bm25_llama3_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama3/2wiki/abv_bridge_llama3_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama3/2wiki/kirag_llama3_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama3/2wiki/chain_rag_llama3_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/sweep_llama3/hotpotqa/naive_bm25_llama3_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama3/hotpotqa/abv_bridge_llama3_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama3/hotpotqa/kirag_llama3_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama3/hotpotqa/chain_rag_llama3_hotpotqa_traces.jsonl"),
    },
    "Phi-4-mini": {
        "MuSiQue":  ("outputs/sweep_phi/musique/naive_bm25_phi_musique_traces.jsonl",
                     "outputs/sweep_phi/musique/abv_bridge_phi_musique_traces.jsonl",
                     "outputs/sweep_phi/musique/kirag_phi_musique_traces.jsonl",
                     "outputs/sweep_phi/musique/chain_rag_phi_musique_traces.jsonl"),
        "2Wiki":    ("outputs/sweep_phi/2wiki/naive_bm25_phi_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_phi/2wiki/abv_bridge_phi_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_phi/2wiki/kirag_phi_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_phi/2wiki/chain_rag_phi_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/sweep_phi/hotpotqa/naive_bm25_phi_hotpotqa_traces.jsonl",
                     "outputs/sweep_phi/hotpotqa/abv_bridge_phi_hotpotqa_traces.jsonl",
                     "outputs/sweep_phi/hotpotqa/kirag_phi_hotpotqa_traces.jsonl",
                     "outputs/sweep_phi/hotpotqa/chain_rag_phi_hotpotqa_traces.jsonl"),
    },
    "Llama-3.1-8B": {
        "MuSiQue":  ("outputs/sweep_llama31/musique/naive_bm25_llama31_musique_traces.jsonl",
                     "outputs/sweep_llama31/musique/abv_bridge_llama31_musique_traces.jsonl",
                     "outputs/sweep_llama31/musique/kirag_llama31_musique_traces.jsonl",
                     "outputs/sweep_llama31/musique/chain_rag_llama31_musique_traces.jsonl"),
        "2Wiki":    ("outputs/sweep_llama31/2wiki/naive_bm25_llama31_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama31/2wiki/abv_bridge_llama31_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama31/2wiki/kirag_llama31_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_llama31/2wiki/chain_rag_llama31_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/sweep_llama31/hotpotqa/naive_bm25_llama31_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama31/hotpotqa/abv_bridge_llama31_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama31/hotpotqa/kirag_llama31_hotpotqa_traces.jsonl",
                     "outputs/sweep_llama31/hotpotqa/chain_rag_llama31_hotpotqa_traces.jsonl"),
    },
    "Gemma-3-31B": {
        "MuSiQue":  ("outputs/sweep_gemma31b/musique/naive_bm25_gemma31b_musique_traces.jsonl",
                     "outputs/sweep_gemma31b/musique/abv_bridge_gemma31b_musique_traces.jsonl",
                     "outputs/sweep_gemma31b/musique/kirag_gemma31b_musique_traces.jsonl",
                     "outputs/sweep_gemma31b/musique/chain_rag_gemma31b_musique_traces.jsonl"),
        "2Wiki":    ("outputs/sweep_gemma31b/2wiki/naive_bm25_gemma31b_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_gemma31b/2wiki/abv_bridge_gemma31b_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_gemma31b/2wiki/kirag_gemma31b_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_gemma31b/2wiki/chain_rag_gemma31b_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/sweep_gemma31b/hotpotqa/naive_bm25_gemma31b_hotpotqa_traces.jsonl",
                     "outputs/sweep_gemma31b/hotpotqa/abv_bridge_gemma31b_hotpotqa_traces.jsonl",
                     "outputs/sweep_gemma31b/hotpotqa/kirag_gemma31b_hotpotqa_traces.jsonl",
                     "outputs/sweep_gemma31b/hotpotqa/chain_rag_gemma31b_hotpotqa_traces.jsonl"),
    },
    "Mistral-Small-4-119B": {
        "MuSiQue":  ("outputs/sweep_mistral_small/musique/naive_bm25_mistral_small_musique_traces.jsonl",
                     "outputs/sweep_mistral_small/musique/abv_bridge_mistral_small_musique_traces.jsonl",
                     "outputs/sweep_mistral_small/musique/kirag_mistral_small_musique_traces.jsonl",
                     "outputs/sweep_mistral_small/musique/chain_rag_mistral_small_musique_traces.jsonl"),
        "2Wiki":    ("outputs/sweep_mistral_small/2wiki/naive_bm25_mistral_small_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_mistral_small/2wiki/abv_bridge_mistral_small_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_mistral_small/2wiki/kirag_mistral_small_2wikimultihopqa_traces.jsonl",
                     "outputs/sweep_mistral_small/2wiki/chain_rag_mistral_small_2wikimultihopqa_traces.jsonl"),
        "HotpotQA": ("outputs/sweep_mistral_small/hotpotqa/naive_bm25_mistral_small_hotpotqa_traces.jsonl",
                     "outputs/sweep_mistral_small/hotpotqa/abv_bridge_mistral_small_hotpotqa_traces.jsonl",
                     "outputs/sweep_mistral_small/hotpotqa/kirag_mistral_small_hotpotqa_traces.jsonl",
                     "outputs/sweep_mistral_small/hotpotqa/chain_rag_mistral_small_hotpotqa_traces.jsonl"),
    },
}


def _load_jsonl(p):
    p = ROOT / p
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def build_table(reader: str):
    """Return per-question dict: qid -> {dataset, X, F1_S, F1_P, F1_I, F1_D, T_S, T_P, T_I, T_D}."""
    table = {}
    sample_n = {}
    for ds, (sp, pp, ip, dp) in READER_TRACES[reader].items():
        s_recs = _load_jsonl(sp)
        p_recs = _load_jsonl(pp)
        i_recs = _load_jsonl(ip)
        d_recs = _load_jsonl(dp)
        if any(x is None for x in (s_recs, p_recs, i_recs, d_recs)):
            print(f"  WARN: missing trace for {reader} / {ds}")
            continue
        s_map = {r["question_id"]: r for r in s_recs}
        p_map = {r["question_id"]: r for r in p_recs}
        i_map = {r["question_id"]: r for r in i_recs}
        d_map = {r["question_id"]: r for r in d_recs}
        common = set(s_map) & set(p_map) & set(i_map) & set(d_map)
        sample_n[ds] = len(common)
        for q in common:
            sr = s_map[q]
            try:
                X = features_from_stop_rec(sr)
            except Exception:
                continue
            table[(ds, q)] = {
                "dataset": ds,
                "X": X,
                "F1": {
                    "STOP":  float(sr.get("ans_f1") or 0),
                    "PRUNE": float(p_map[q].get("ans_f1") or 0),
                    "ITER":  float(i_map[q].get("ans_f1") or 0),
                    "DECOMP": float(d_map[q].get("ans_f1") or 0),
                },
                "TOK": {
                    "STOP":  int(sr.get("planner_tokens") or 0),
                    "PRUNE": int(p_map[q].get("planner_tokens") or 0),
                    "ITER":  int(i_map[q].get("planner_tokens") or 0),
                    "DECOMP": int(d_map[q].get("planner_tokens") or 0),
                },
            }
    return table, sample_n


# ── Oracle computation ──────────────────────────────────────────────────
def oracle_f1(table, allowed_routes, delta=0.0, cost_order=None):
    """For each q, pick best route in allowed_routes (with delta margin if cost_order given)."""
    f1s = []; toks = []; routes = []
    for q, d in table.items():
        F1, TOK = d["F1"], d["TOK"]
        if cost_order is None:
            # pure F1 oracle
            best_route = max(allowed_routes, key=lambda r: F1[r])
        else:
            # cost-sensitive: pick more expensive route only if F1 improves >= delta
            sorted_routes = [r for r in cost_order if r in allowed_routes]
            best_route = sorted_routes[0]
            best_f1 = F1[best_route]
            for r in sorted_routes[1:]:
                if F1[r] >= best_f1 + delta:
                    best_route = r
                    best_f1 = F1[r]
        f1s.append(F1[best_route])
        toks.append(TOK[best_route])
        routes.append(best_route)
    return {
        "f1_mean": float(np.mean(f1s)),
        "tok_mean": float(np.mean(toks)),
        "n": len(f1s),
        "route_dist": dict(Counter(routes)),
    }


# ── Train 3-class GBM, 5-fold CV ─────────────────────────────────────────
COST_ORDER = ["STOP", "PRUNE", "ITER"]  # ascending cost; DECOMP excluded for primary 3-class router


def make_labels(table, delta, allowed=COST_ORDER):
    """y = best route per question with cost margin."""
    y = []
    qkeys = list(table.keys())
    for qk in qkeys:
        d = table[qk]
        F1 = d["F1"]
        sorted_r = [r for r in COST_ORDER if r in allowed]
        best = sorted_r[0]; best_f1 = F1[best]
        for r in sorted_r[1:]:
            if F1[r] >= best_f1 + delta:
                best = r; best_f1 = F1[r]
        y.append(best)
    return qkeys, y


def evaluate_held_out_router(table, delta=0.0, n_splits=5, seed=42):
    """5-fold CV: train 3-class GBM, evaluate F1 + tokens + route distribution."""
    qkeys, y_str = make_labels(table, delta=delta, allowed=COST_ORDER)
    X = np.array([table[q]["X"] for q in qkeys])
    LBL = {"STOP": 0, "PRUNE": 1, "ITER": 2}
    INV = {v: k for k, v in LBL.items()}
    y = np.array([LBL[v] for v in y_str])

    if len(set(y)) < 2:
        return {"err": f"only one class present: {Counter(y_str)}"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          subsample=0.8, random_state=seed)
        try:
            clf.fit(X[train_idx], y[train_idx])
        except ValueError:
            continue
        pred = clf.predict(X[test_idx])
        # Apply per-question route, compute simulated F1 + token
        f1s = []; toks = []; routes = []
        for i, idx in enumerate(test_idx):
            qk = qkeys[idx]
            chosen = INV[pred[i]]
            if chosen not in table[qk]["F1"]:
                chosen = "STOP"
            f1s.append(table[qk]["F1"][chosen])
            toks.append(table[qk]["TOK"][chosen])
            routes.append(chosen)
        fold_results.append({
            "f1_mean": float(np.mean(f1s)),
            "tok_mean": float(np.mean(toks)),
            "route_dist": dict(Counter(routes)),
        })
    if not fold_results:
        return {"err": "no fold trained"}
    f1 = float(np.mean([r["f1_mean"] for r in fold_results]))
    f1_std = float(np.std([r["f1_mean"] for r in fold_results]))
    tok = float(np.mean([r["tok_mean"] for r in fold_results]))
    # aggregate route dist
    agg = Counter()
    for r in fold_results:
        agg.update(r["route_dist"])
    return {"f1_mean": f1, "f1_std": f1_std, "tok_mean": tok, "route_dist": dict(agg),
            "label_dist": dict(Counter(y_str))}


def evaluate_2route_baseline(table):
    """Current RASER 2-route (STOP/PRUNE) baseline via 5-fold CV — same setup."""
    qkeys = sorted(table.keys())  # fixed row order -> reproducible CV folds
    X = np.array([table[q]["X"] for q in qkeys])
    y = np.array([1 if table[q]["F1"]["PRUNE"] > table[q]["F1"]["STOP"] + 1e-6 else 0
                  for q in qkeys])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1, fold_tok = [], []
    per_q_f1 = [0.0] * len(qkeys)  # routed F1 for each question (each in one test fold)
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          subsample=0.8, random_state=42)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        decisions = (proba >= 0.20).astype(int)
        f1s = []; toks = []
        for j, idx in enumerate(te):
            d = table[qkeys[idx]]
            r = "PRUNE" if decisions[j] == 1 else "STOP"
            f1s.append(d["F1"][r]); toks.append(d["TOK"][r])
            per_q_f1[idx] = float(d["F1"][r])
        fold_f1.append(np.mean(f1s)); fold_tok.append(np.mean(toks))
    return {"f1_mean": float(np.mean(fold_f1)), "f1_std": float(np.std(fold_f1)),
            "tok_mean": float(np.mean(fold_tok)), "per_q_f1": per_q_f1}


# ── Main ────────────────────────────────────────────────────────────────
def main():
    out = {}
    print("="*100)
    print(" 3-ROUTE FEASIBILITY DRY-RUN  (no LLM calls, reads existing traces)")
    print("="*100)
    for reader in READER_TRACES:
        print(f"\n###  Reader: {reader}")
        table, sample_n = build_table(reader)
        N = len(table)
        print(f"  N(common qids across 4 methods) = {N}, per-dataset = {sample_n}")
        if N < 100:
            print(f"  SKIP (insufficient data)"); continue
        out[reader] = {"n_total": N, "per_dataset": sample_n, "results": {}}

        # Always-X baselines
        print("\n  --- Always-X baselines (per-question average over the union) ---")
        for r in ["STOP", "PRUNE", "ITER", "DECOMP"]:
            f1m = float(np.mean([d["F1"][r] for d in table.values()]))
            tkm = float(np.mean([d["TOK"][r] for d in table.values()]))
            print(f"    always-{r:7s}  F1 = {f1m:.4f}  avg_tokens = {tkm:>7.0f}")
            out[reader]["results"][f"always_{r}"] = {"f1": f1m, "tokens": tkm}

        # 2-route RASER baseline (CV-evaluated, same as deployed)
        baseline = evaluate_2route_baseline(table)
        print(f"\n  --- 2-route RASER (current, CV) ---")
        print(f"    F1 = {baseline['f1_mean']:.4f} ± {baseline['f1_std']:.4f}  "
              f"avg_tokens = {baseline['tok_mean']:>7.0f}")
        out[reader]["results"]["raser_2route_cv"] = baseline

        # Pure F1 oracles
        print("\n  --- Oracle ceilings (best-of-routes per question, F1 only) ---")
        configs = [
            ("oracle_S+P",     ["STOP", "PRUNE"]),
            ("oracle_S+P+I",   ["STOP", "PRUNE", "ITER"]),
            ("oracle_S+P+D",   ["STOP", "PRUNE", "DECOMP"]),
            ("oracle_S+P+I+D", ["STOP", "PRUNE", "ITER", "DECOMP"]),
        ]
        for name, allowed in configs:
            o = oracle_f1(table, allowed)
            print(f"    {name:18s}  F1 = {o['f1_mean']:.4f}  avg_tokens = {o['tok_mean']:>7.0f}  "
                  f"routes = {o['route_dist']}")
            out[reader]["results"][name] = o

        # Cost-sensitive oracles (3-route, with margin)
        print("\n  --- Cost-sensitive oracles (3-route S/P/I, only escalate if ΔF1 ≥ delta) ---")
        for delta in [0.00, 0.05, 0.10]:
            o = oracle_f1(table, ["STOP", "PRUNE", "ITER"], delta=delta, cost_order=COST_ORDER)
            print(f"    delta = {delta:.2f}  F1 = {o['f1_mean']:.4f}  "
                  f"avg_tokens = {o['tok_mean']:>7.0f}  routes = {o['route_dist']}")
            out[reader]["results"][f"oracle_costaware_d{delta:.2f}"] = o

        # Held-out 3-class GBM
        print("\n  --- Held-out 3-class GBM (5-fold CV, S/P/I, with cost margin) ---")
        for delta in [0.00, 0.05, 0.10]:
            r = evaluate_held_out_router(table, delta=delta)
            if "err" in r:
                print(f"    delta = {delta:.2f}  ERR: {r['err']}")
                continue
            print(f"    delta = {delta:.2f}  F1 = {r['f1_mean']:.4f} ± {r['f1_std']:.4f}  "
                  f"avg_tokens = {r['tok_mean']:>7.0f}  routes = {r['route_dist']}")
            print(f"                        label_dist (training y) = {r['label_dist']}")
            out[reader]["results"][f"gbm3class_d{delta:.2f}"] = r

    # Save
    out_path = ROOT / "outputs" / "three_route_feasibility" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")

    # Adoption-criteria summary
    print("\n" + "="*100)
    print(" ADOPTION CRITERIA CHECK")
    print("="*100)
    for reader, info in out.items():
        r = info["results"]
        baseline_f1 = r.get("raser_2route_cv", {}).get("f1_mean", 0)
        always_iter_tok = r.get("always_ITER", {}).get("tokens", 0)
        for delta in [0.00, 0.05, 0.10]:
            key = f"gbm3class_d{delta:.2f}"
            if key not in r:
                continue
            new = r[key]
            df = new["f1_mean"] - baseline_f1
            tok_ratio = new["tok_mean"] / always_iter_tok if always_iter_tok else 0
            f1_ok = df >= 0.02
            cost_ok = tok_ratio <= 0.60
            verdict = "PASS" if (f1_ok and cost_ok) else "FAIL"
            print(f"  {reader:14s}  delta={delta:.2f}  ΔF1 vs 2-route = {df:+.4f}  "
                  f"token = {new['tok_mean']:.0f} ({tok_ratio:.2f}× always-ITER)  [{verdict}]")


if __name__ == "__main__":
    main()
