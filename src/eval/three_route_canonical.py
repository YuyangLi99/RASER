"""CANONICAL 3-route RASER evaluation (honest version).

This supersedes three_route_proper_eval_v2.py, whose v2 features included a
STOP-vs-PRUNE-agreement signal that secretly required running the abv_bridge
pipeline (~3k tokens) -- so the reported "0.59x always-ITER cost" was a
cost-undercount. Here we use ONLY features computable from the cheap STOP read
(+ question metadata), and the cost accounting is straightforward.

Routes (generic names -- we do NOT claim to reproduce KiRAG/ChainRAG):
  STOP            cheap top-K read, answer as-is
  TOP2_PRUNE      prune evidence to top-2, one more read
  ITER_RETRIEVE   IRCoT-style iterative retrieval primitive (triple-state variant)

Features (6, all available right after the STOP read, no escalation needed):
  1. heuristic answer confidence
  2. STOP answer length (#tokens)
  3. bridge-cue keyword present in question (boolean)
  4. question type (entity/date/yes_no/count/other)
  5. hop count (parsed from MuSiQue qid; 2 for 2Wiki/HotpotQA)
  6. STOP answer is "I don't know" (boolean)

Protocol:
  - outer 5-fold CV, repeated over SEEDS for error bars
  - per-route F1 regressors trained on outer-TRAIN only
  - routing: argmax_r [ pred_F1_r(x) - lambda * c_r ], c_r = route-LEVEL avg cost
    estimated on outer-TRAIN (no test-fold cost peek)
  - lambda chosen on outer-TRAIN by cost-budget rule ('use <= budget * always-ITER tokens')
  - per-(reader, dataset) breakdown; no cross-dataset averaging
  - operating points: low_cost (33% budget), balanced (60% budget)
  - Pareto curve over the full lambda sweep (fold-pooled predictions)

Outputs:
  outputs/three_route_canonical/summary.json
  outputs/three_route_canonical/main_table.md
  outputs/three_route_canonical/pareto_grid.png
  demo/three_route_data.json   (for the Streamlit demo)

Usage:
    python -m src.eval.three_route_canonical
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.methods.abv_bridge.trigger_gate import has_bridge_cues
from src.eval.answer_normalizer import classify_question_type
from src.eval.three_route_feasibility import READER_TRACES, _load_jsonl, evaluate_2route_baseline

QTYPE_MAP = {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}
ROUTES = ["STOP", "PRUNE", "ITER"]
ROUTE_LABEL = {"STOP": "STOP", "PRUNE": "TOP2_PRUNE", "ITER": "ITER_RETRIEVE"}
LAMBDA_SWEEP = [0.0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
OPERATING_POINTS = {"low_cost": 0.33, "balanced": 0.60}
SEEDS = [42, 1, 7, 123, 2024]
READER_ORDER = ["GPT-OSS-120B", "Mistral-Small-4-119B", "Gemma-3-31B", "Llama-3-8B", "Phi-4-mini", "Llama-3.1-8B"]
DATASET_ORDER = ["MuSiQue", "2Wiki", "HotpotQA"]
FEATURE_NAMES = ["confidence", "ans_len", "bridge_cues", "score_gap", "score_top1", "qtype"]


def _confidence(a: str, qt: str) -> float:
    if not a or a.strip().lower() in ("i don't know", "", "i dont know"):
        return 0.0
    if qt == "yes_no" and a.strip().lower() in ("yes", "no"):
        return 0.9
    if qt in ("date", "count"):
        return 0.8
    w = a.strip().split()
    return 0.6 if len(w) <= 2 else (0.3 if len(w) >= 10 else 0.7)


def _hop(qid: str) -> int:
    m = re.match(r"^(\d+)hop", qid)
    return int(m.group(1)) if m else 2


def _idk(s: str) -> bool:
    return (not s) or s.strip().lower() in ("i don't know", "", "i dont know", "unknown")


def _features(stop_rec: dict) -> List[float]:
    """Paper's 6 features (matches Table tab:features and the deployed
    abv_bridge 2-action router): confidence, ans_len, bridge_cues,
    score_gap, score_top1, qtype.
    """
    q = stop_rec.get("question", "")
    a = stop_rec.get("answer") or stop_rec.get("answer_raw") or ""
    qt = classify_question_type(q)
    chunks = stop_rec.get("text_evidence") or []
    s = [float(c.get("score") or 0.0) for c in chunks[:10]]
    t1 = s[0] if s else 0.0
    t5 = s[4] if len(s) >= 5 else 0.0
    return [
        round(_confidence(a, qt), 3),
        len(a.split()),
        1.0 if has_bridge_cues(q) else 0.0,
        round(t1 - t5, 4),
        round(t1, 4),
        QTYPE_MAP.get(qt, 4),
    ]


def build_table(reader: str) -> Dict:
    table = {}
    for ds, (sp, pp, ip, dp) in READER_TRACES[reader].items():
        s = _load_jsonl(sp); p = _load_jsonl(pp); i = _load_jsonl(ip); d = _load_jsonl(dp)
        if any(x is None for x in (s, p, i, d)):
            continue
        sm = {r["question_id"]: r for r in s}; pm = {r["question_id"]: r for r in p}
        im = {r["question_id"]: r for r in i}; dm = {r["question_id"]: r for r in d}
        # sorted(): set-intersection iteration order depends on per-process
        # string hashing (PYTHONHASHSEED), which made the CV row order --- and
        # hence every trained-router result --- vary run to run. Sorting fixes
        # the row order so the canonical numbers are reproducible.
        for q in sorted(set(sm) & set(pm) & set(im) & set(dm)):
            sr = sm[q]
            try:
                X = _features(sr)
            except Exception:
                continue
            table[(ds, q)] = {
                "dataset": ds, "X": X,
                "F1": {"STOP": float(sr.get("ans_f1") or 0),
                       "PRUNE": float(pm[q].get("ans_f1") or 0),
                       "ITER": float(im[q].get("ans_f1") or 0),
                       "DECOMP": float(dm[q].get("ans_f1") or 0)},
                "TOK": {"STOP": int(sr.get("planner_tokens") or 0),
                        "PRUNE": int(pm[q].get("planner_tokens") or 0),
                        "ITER": int(im[q].get("planner_tokens") or 0),
                        "DECOMP": int(dm[q].get("planner_tokens") or 0)},
            }
    return table


def _boot_ci(arr, n_boot: int = 10000, seed: int = 12345) -> float:
    """95% bootstrap CI half-width of the mean, resampling over questions.
    Deterministic (fixed RNG seed). Returns (hi-lo)/2 for symmetric error bars."""
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float((hi - lo) / 2.0)


def _route_choice(pred: Dict[str, np.ndarray], cost_r: Dict[str, float], lam: float) -> List[str]:
    n = len(pred[ROUTES[0]])
    scores = np.stack([pred[r] - lam * cost_r[r] * np.ones(n) for r in ROUTES], axis=1)
    return [ROUTES[k] for k in scores.argmax(axis=1)]


def _pick_lambda(pred_tr, TOK_tr, cost_tr, frac, ai_tok_tr):
    target = frac * ai_tok_tr
    chosen = LAMBDA_SWEEP[-1]
    for lam in LAMBDA_SWEEP:
        ch = _route_choice(pred_tr, cost_tr, lam)
        tm = float(np.mean([TOK_tr[ch[k]][k] for k in range(len(ch))]))
        if tm <= target:
            chosen = lam; break
    return chosen


def _eval_one_seed(view: Dict, seed: int):
    qk = sorted(view); n = len(qk)
    X = np.array([view[q]["X"] for q in qk])
    Y = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ROUTES}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ROUTES}
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = {r: np.zeros(n) for r in ROUTES}
    op_choices = {op: [None] * n for op in OPERATING_POINTS}
    op_lams = {op: [] for op in OPERATING_POINTS}
    cost_folds = {r: [] for r in ROUTES}
    r2_reg_quality = {r: [] for r in ROUTES}  # spearman-ish: corr(pred, true) on test fold
    for tr, te in kf.split(X):
        regs = {}
        for r in ROUTES:
            g = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          subsample=0.8, random_state=seed)
            g.fit(X[tr], Y[r][tr]); regs[r] = g
            pr = g.predict(X[te]); pred[r][te] = pr
            if np.std(pr) > 1e-9 and np.std(Y[r][te]) > 1e-9:
                r2_reg_quality[r].append(float(np.corrcoef(pr, Y[r][te])[0, 1]))
        cost_tr = {r: float(TOK[r][tr].mean()) for r in ROUTES}
        for r in ROUTES:
            cost_folds[r].append(cost_tr[r])
        ai = cost_tr["ITER"]
        pred_tr = {r: regs[r].predict(X[tr]) for r in ROUTES}
        TOK_tr = {r: TOK[r][tr] for r in ROUTES}
        for op, frac in OPERATING_POINTS.items():
            lam = _pick_lambda(pred_tr, TOK_tr, cost_tr, frac, ai)
            cht = _route_choice({r: pred[r][te] for r in ROUTES}, cost_tr, lam)
            for j, idx in enumerate(te):
                op_choices[op][idx] = cht[j]
            op_lams[op].append(lam)
    cost_mean = {r: float(np.mean(cost_folds[r])) for r in ROUTES}
    out = {"cost_mean": cost_mean, "reg_quality": {r: float(np.mean(v)) if v else 0.0 for r, v in r2_reg_quality.items()}}
    for op in OPERATING_POINTS:
        ch = op_choices[op]
        out[op] = {
            "f1": float(np.mean([view[qk[k]]["F1"][ch[k]] for k in range(n)])),
            "tok": float(np.mean([view[qk[k]]["TOK"][ch[k]] for k in range(n)])),
            "route_dist": {ROUTE_LABEL[k]: v for k, v in Counter(ch).items()},
            "lambda_median": float(np.median(op_lams[op])),
            # per-question routed F1 (qk is sorted(view) -> same order every seed)
            "per_q_f1": [float(view[qk[k]]["F1"][ch[k]]) for k in range(n)],
        }
    # Pareto over the *last* seed's fold-pooled predictions (cosmetic only)
    out["pareto"] = []
    for lam in LAMBDA_SWEEP:
        ch = _route_choice(pred, cost_mean, lam)
        out["pareto"].append({
            "lambda": lam,
            "f1": float(np.mean([view[qk[k]]["F1"][ch[k]] for k in range(n)])),
            "tok": float(np.mean([view[qk[k]]["TOK"][ch[k]] for k in range(n)])),
            "route_dist": {ROUTE_LABEL[k]: v for k, v in Counter(ch).items()},
        })
    return out


def _feature_ablation(view: Dict, seed: int = 42) -> Dict:
    """Leave-one-feature-out at the 'balanced' operating point. Returns {feat: ΔF1_drop}."""
    base = _eval_one_seed(view, seed)["balanced"]["f1"]
    out = {}
    qk = sorted(view); n = len(qk)
    full_X = np.array([view[q]["X"] for q in qk])
    Y = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ROUTES}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ROUTES}
    for fi, fname in enumerate(FEATURE_NAMES):
        keep = [j for j in range(full_X.shape[1]) if j != fi]
        Xk = full_X[:, keep]
        kf = KFold(5, shuffle=True, random_state=seed)
        pred = {r: np.zeros(n) for r in ROUTES}
        op_choices = [None] * n
        for tr, te in kf.split(Xk):
            cost_tr = {r: float(TOK[r][tr].mean()) for r in ROUTES}
            ai = cost_tr["ITER"]
            regs = {}
            for r in ROUTES:
                g = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                              subsample=0.8, random_state=seed)
                g.fit(Xk[tr], Y[r][tr]); regs[r] = g; pred[r][te] = g.predict(Xk[te])
            pred_tr = {r: regs[r].predict(Xk[tr]) for r in ROUTES}; TOK_tr = {r: TOK[r][tr] for r in ROUTES}
            lam = _pick_lambda(pred_tr, TOK_tr, cost_tr, 0.60, ai)
            cht = _route_choice({r: pred[r][te] for r in ROUTES}, cost_tr, lam)
            for j, idx in enumerate(te):
                op_choices[idx] = cht[j]
        f1_wo = float(np.mean([view[qk[k]]["F1"][op_choices[k]] for k in range(n)]))
        out[fname] = round(base - f1_wo, 4)  # positive => feature helps
    return out


def main():
    out_root = ROOT / "outputs/three_route_canonical"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print(" CANONICAL 3-ROUTE RASER  (6 cheap STOP-state features, correct cost, multi-seed CV)")
    print("=" * 120)

    summary = {}
    for reader in READER_ORDER:
        if reader not in READER_TRACES:
            continue
        table = build_table(reader)
        if not table:
            continue
        by_ds = {}
        for k, d in table.items():
            by_ds.setdefault(d["dataset"], {})[k] = d
        summary[reader] = {}
        print(f"\n###  {reader}")
        for ds in DATASET_ORDER:
            if ds not in by_ds:
                continue
            view = by_ds[ds]
            n = len(view)
            # baselines
            Y = {r: np.array([view[q]["F1"][r] for q in view]) for r in ROUTES + ["DECOMP"]}
            TOK = {r: np.array([view[q]["TOK"][r] for q in view]) for r in ROUTES + ["DECOMP"]}
            base = {f"always_{ROUTE_LABEL.get(r, r)}": {"f1": float(Y[r].mean()),
                    "tok": float(TOK[r].mean()), "f1_ci": _boot_ci(Y[r])}
                    for r in ROUTES + ["DECOMP"]}
            oi = np.argmax(np.stack([Y[r] for r in ROUTES], 1), 1)
            oracle_pq = np.array([Y[ROUTES[oi[k]]][k] for k in range(n)])
            base["oracle_3route"] = {
                "f1": float(oracle_pq.mean()),
                "tok": float(np.mean([TOK[ROUTES[oi[k]]][k] for k in range(n)])),
                "f1_ci": _boot_ci(oracle_pq),
            }
            r2 = evaluate_2route_baseline(view) if n >= 50 else None
            # multi-seed
            seed_runs = [_eval_one_seed(view, s) for s in SEEDS]
            agg = {"n": n, "baselines": base,
                   "raser_2route": {"f1": r2["f1_mean"], "tok": r2["tok_mean"],
                                    "f1_ci": _boot_ci(r2["per_q_f1"])} if r2 else None}
            for op in OPERATING_POINTS:
                f1s = [sr[op]["f1"] for sr in seed_runs]
                toks = [sr[op]["tok"] for sr in seed_runs]
                # union of route dists
                rd = Counter()
                for sr in seed_runs:
                    rd.update(sr[op]["route_dist"])
                # per-question routed F1 averaged across seeds, then bootstrapped
                pq_seed = np.mean([sr[op]["per_q_f1"] for sr in seed_runs], axis=0)
                agg[op] = {
                    "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                    "f1_ci": _boot_ci(pq_seed),
                    "tok_mean": float(np.mean(toks)),
                    "lambda_median": float(np.median([sr[op]["lambda_median"] for sr in seed_runs])),
                    "route_dist_total": dict(rd),
                }
            agg["reg_quality"] = {r: float(np.mean([sr["reg_quality"][r] for sr in seed_runs])) for r in ROUTES}
            # 5-seed-averaged Pareto sweep, per lambda (every seed sweeps the
            # same LAMBDA_SWEEP in the same order). Previously this used only
            # seed_runs[0], which made the figure's RASER-3 curve inconsistent
            # with the 5-seed `balanced`/`low_cost` points in the main table.
            agg["pareto"] = []
            for _i, _lam in enumerate(LAMBDA_SWEEP):
                _f1 = [sr["pareto"][_i]["f1"] for sr in seed_runs]
                _tk = [sr["pareto"][_i]["tok"] for sr in seed_runs]
                agg["pareto"].append({"lambda": _lam,
                                      "f1": float(np.mean(_f1)),
                                      "tok": float(np.mean(_tk))})
            agg["feature_ablation"] = _feature_ablation(view)
            summary[reader][ds] = agg

            # print
            r2f = r2["f1_mean"] if r2 else None
            iter_tok = base["always_ITER_RETRIEVE"]["tok"]
            for op in ["balanced"]:
                a = agg[op]
                df1 = a["f1_mean"] - r2f if r2f is not None else None
                trat = a["tok_mean"] / iter_tok
                ok = (df1 is not None and df1 >= 0.02 and trat <= 0.60)
                v = "✓ PASS" if ok else ("≈ marginal" if (df1 and df1 >= 0.02) or (df1 and df1 > 0 and trat <= 0.6) else "✗ FAIL")
                print(f"  {ds:9s} n={n}  RASER-2={r2f:.3f}  RASER-3={a['f1_mean']:.3f}±{a['f1_std']:.3f}  "
                      f"ΔF1={df1:+.3f}  tok={a['tok_mean']:.0f} ({trat:.2f}× ITER)  "
                      f"oracle={base['oracle_3route']['f1']:.3f}  routes={a['route_dist_total']}  [{v}]")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_root / 'summary.json'}")

    # ── main_table.md ──────────────────────────────────────────────────────
    md = []
    md.append("# 3-route RASER — canonical results (honest version)\n")
    md.append("Outer 5-fold CV × 5 seeds. Per-route F1 regressors trained on outer-TRAIN; routing")
    md.append("uses route-LEVEL avg cost (train-estimated, no test-fold peek); lambda chosen on")
    md.append("outer-TRAIN by a cost-budget rule ('use ≤60% of always-ITER tokens'). 6 cheap")
    md.append("STOP-state features only. Routes: STOP / TOP2_PRUNE / ITER_RETRIEVE (IRCoT-style,")
    md.append("NOT a reproduction of KiRAG).\n")
    md.append("| Reader | Dataset | STOP | PRUNE | ITER_RETRIEVE | oracle-3 | RASER-2 | **RASER-3** | ΔF1 vs R2 | tok/ITER | Verdict |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
    for reader in READER_ORDER:
        if reader not in summary:
            continue
        for ds in DATASET_ORDER:
            if ds not in summary[reader]:
                continue
            a = summary[reader][ds]; b = a["baselines"]
            r2f = a["raser_2route"]["f1"] if a["raser_2route"] else 0.0
            bal = a["balanced"]
            df1 = bal["f1_mean"] - r2f
            trat = bal["tok_mean"] / b["always_ITER_RETRIEVE"]["tok"]
            if df1 >= 0.02 and trat <= 0.60:
                v = "✓ PASS"
            elif df1 >= 0.02 or (df1 > 0 and trat <= 0.60):
                v = "≈ marginal"
            else:
                v = "✗ FAIL"
            md.append(f"| {reader} | {ds} | {b['always_STOP']['f1']:.3f} | {b['always_TOP2_PRUNE']['f1']:.3f} | "
                      f"{b['always_ITER_RETRIEVE']['f1']:.3f} | {b['oracle_3route']['f1']:.3f} | {r2f:.3f} | "
                      f"**{bal['f1_mean']:.3f}±{bal['f1_std']:.3f}** | {df1:+.3f} | {trat:.2f}× | {v} |")
        md.append("")
    md.append("## Per-reader summary\n")
    md.append("| Reader | PASS | marg | FAIL | avg ΔF1 vs R2 | uses % of ITER tok | recovers % of ITER F1 |")
    md.append("|---|:--:|:--:|:--:|---:|---:|---:|")
    for reader in READER_ORDER:
        if reader not in summary:
            continue
        np_ = nm = nf = 0; dr2s = []; ct = []; ff = []
        for ds in summary[reader]:
            a = summary[reader][ds]; b = a["baselines"]
            r2f = a["raser_2route"]["f1"]; bal = a["balanced"]
            df1 = bal["f1_mean"] - r2f; trat = bal["tok_mean"] / b["always_ITER_RETRIEVE"]["tok"]
            dr2s.append(df1); ct.append(trat); ff.append(bal["f1_mean"] / b["always_ITER_RETRIEVE"]["f1"] if b["always_ITER_RETRIEVE"]["f1"] > 0 else 0)
            if df1 >= 0.02 and trat <= 0.6: np_ += 1
            elif df1 >= 0.02 or (df1 > 0 and trat <= 0.6): nm += 1
            else: nf += 1
        md.append(f"| {reader} | {np_} | {nm} | {nf} | {np.mean(dr2s):+.3f} | {np.mean(ct):.0%} | {np.mean(ff):.0%} |")
    (out_root / "main_table.md").write_text("\n".join(md) + "\n")
    print(f"Saved → {out_root / 'main_table.md'}")
    print("\n" + "\n".join(md))

    # ── demo/three_route_data.json ─────────────────────────────────────────
    # New schema = per-dataset detail; we ALSO emit the legacy per-reader keys
    # (aggregated across datasets, weighted by n) so the existing Streamlit page
    # keeps working without code changes.
    demo = {
        "_note": "Generated by src.eval.three_route_canonical -- HONEST version (6 cheap STOP-state features, correct cost accounting, 5-seed CV). Supersedes the old test-set-lambda-tuned / cost-undercounted numbers.",
        "samples": {r: sum(summary[r][ds]["n"] for ds in summary[r]) for r in summary},
        "feature_names": FEATURE_NAMES,
        "per_dataset": {},
        # legacy per-reader keys ↓
        "always_baselines": {}, "two_route_raser": {}, "three_route_raser": {},
        "lambda_sweep": {}, "regression_quality": {}, "feature_ablation": {},
        "cross_reader": {},  # not recomputed in the canonical eval; left empty
    }
    for reader in summary:
        demo["per_dataset"][reader] = {}
        # accumulators for per-reader aggregate
        tot_n = sum(summary[reader][ds]["n"] for ds in summary[reader])
        w = {ds: summary[reader][ds]["n"] / tot_n for ds in summary[reader]}
        ab_keys = ["always_STOP", "always_TOP2_PRUNE", "always_ITER_RETRIEVE", "always_DECOMP", "oracle_3route"]
        agg_ab = {k: {"f1": 0.0, "tok": 0.0} for k in ab_keys}
        agg_r2 = {"f1": 0.0, "tok": 0.0}
        agg_r3 = {"f1": 0.0, "tok": 0.0, "route_dist": Counter()}
        agg_fa = {fn: 0.0 for fn in FEATURE_NAMES}
        agg_rq = {r: 0.0 for r in ROUTES}
        biggest_ds = max(summary[reader], key=lambda d: summary[reader][d]["n"])
        for ds in summary[reader]:
            a = summary[reader][ds]; b = a["baselines"]
            r2f = a["raser_2route"]["f1"] if a["raser_2route"] else None
            r2t = a["raser_2route"]["tok"] if a["raser_2route"] else None
            bal = a["balanced"]; low = a["low_cost"]
            df1 = (bal["f1_mean"] - r2f) if r2f is not None else None
            trat = bal["tok_mean"] / b["always_ITER_RETRIEVE"]["tok"]
            demo["per_dataset"][reader][ds] = {
                "n": a["n"],
                "always": {k: v for k, v in b.items()},
                "raser_2route": {"f1": r2f, "tok": r2t},
                "raser_3route_low_cost": {"f1": low["f1_mean"], "f1_std": low["f1_std"], "tok": low["tok_mean"],
                                          "route_dist": low["route_dist_total"], "lambda_median": low["lambda_median"]},
                "raser_3route_balanced": {"f1": bal["f1_mean"], "f1_std": bal["f1_std"], "tok": bal["tok_mean"],
                                          "route_dist": bal["route_dist_total"], "lambda_median": bal["lambda_median"]},
                "delta_f1_vs_2route": df1,
                "tok_ratio_vs_always_iter": trat,
                "passed": bool(df1 is not None and df1 >= 0.02 and trat <= 0.60),
                "pareto": a["pareto"],
                "reg_quality": a["reg_quality"],
                "feature_ablation": a["feature_ablation"],
            }
            for k in ab_keys:
                agg_ab[k]["f1"] += w[ds] * b[k]["f1"]; agg_ab[k]["tok"] += w[ds] * b[k]["tok"]
            if r2f is not None:
                agg_r2["f1"] += w[ds] * r2f; agg_r2["tok"] += w[ds] * r2t
            agg_r3["f1"] += w[ds] * bal["f1_mean"]; agg_r3["tok"] += w[ds] * bal["tok_mean"]
            agg_r3["route_dist"].update(bal["route_dist_total"])
            for fn in FEATURE_NAMES:
                agg_fa[fn] += w[ds] * a["feature_ablation"].get(fn, 0.0)
            for r in ROUTES:
                agg_rq[r] += w[ds] * a["reg_quality"].get(r, 0.0)
        # legacy keys
        demo["always_baselines"][reader] = {
            "always_iter_f1": agg_ab["always_ITER_RETRIEVE"]["f1"], "always_iter_tok": agg_ab["always_ITER_RETRIEVE"]["tok"],
            "always_stop_f1": agg_ab["always_STOP"]["f1"], "always_stop_tok": agg_ab["always_STOP"]["tok"],
            "always_prune_f1": agg_ab["always_TOP2_PRUNE"]["f1"], "always_prune_tok": agg_ab["always_TOP2_PRUNE"]["tok"],
            "always_decomp_f1": agg_ab["always_DECOMP"]["f1"], "always_decomp_tok": agg_ab["always_DECOMP"]["tok"],
            "oracle_3route_f1": agg_ab["oracle_3route"]["f1"], "oracle_3route_tok": agg_ab["oracle_3route"]["tok"],
        }
        demo["two_route_raser"][reader] = {"f1": agg_r2["f1"], "tok": agg_r2["tok"]}
        passed = (agg_r3["f1"] - agg_r2["f1"]) >= 0.02 and agg_r3["tok"] <= 0.60 * agg_ab["always_ITER_RETRIEVE"]["tok"]
        demo["three_route_raser"][reader] = {
            "f1": agg_r3["f1"], "tok": agg_r3["tok"],
            "delta_f1_vs_2route": agg_r3["f1"] - agg_r2["f1"],
            "tok_ratio_vs_always_iter": agg_r3["tok"] / agg_ab["always_ITER_RETRIEVE"]["tok"],
            "route_dist": dict(agg_r3["route_dist"]),
            "lambda": summary[reader][biggest_ds]["balanced"]["lambda_median"],
            "passed": bool(passed),
        }
        # lambda_sweep: use the biggest dataset's pareto as the representative curve
        demo["lambda_sweep"][reader] = {f"{p['lambda']:.2e}": {"f1_mean": p["f1"], "tok_mean": p["tok"], "route_dist": p["route_dist"]}
                                        for p in summary[reader][biggest_ds]["pareto"]}
        demo["regression_quality"][reader] = dict(agg_rq)
        demo["feature_ablation"][reader] = dict(agg_fa)
    (ROOT / "demo/three_route_data.json").write_text(json.dumps(demo, indent=2))
    print(f"\nSaved → {ROOT / 'demo/three_route_data.json'}  (per-dataset + legacy per-reader keys)")

    # ── pareto grid figure ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(READER_ORDER), len(DATASET_ORDER), figsize=(12, 14))
        for i, reader in enumerate(READER_ORDER):
            for j, ds in enumerate(DATASET_ORDER):
                ax = axes[i][j]
                if reader not in summary or ds not in summary[reader]:
                    ax.axis("off"); continue
                a = summary[reader][ds]; b = a["baselines"]
                f1 = [p["f1"] for p in a["pareto"]]; tok = [p["tok"] for p in a["pareto"]]
                for k, c, lbl in [("always_STOP", "tab:gray", "STOP"),
                                  ("always_TOP2_PRUNE", "tab:orange", "PRUNE"),
                                  ("always_ITER_RETRIEVE", "tab:red", "ITER"),
                                  ("oracle_3route", "black", "oracle")]:
                    ax.scatter([b[k]["tok"]], [b[k]["f1"]], s=55, color=c, marker="x", label=lbl, zorder=4)
                if a["raser_2route"]:
                    ax.scatter([a["raser_2route"]["tok"]], [a["raser_2route"]["f1"]], s=75,
                               color="tab:blue", marker="s", label="RASER-2", zorder=5)
                ax.plot(tok, f1, "o-", color="tab:green", ms=4, lw=1.5, label="RASER-3", zorder=3)
                for op, mk in [("low_cost", "v"), ("balanced", "^")]:
                    ax.scatter([a[op]["tok_mean"]], [a[op]["f1_mean"]], s=85, marker=mk,
                               edgecolor="black", facecolor="tab:green", zorder=6, label=f"RASER-3 ({op})")
                ax.set_title(f"{reader} / {ds} (n={a['n']})", fontsize=9)
                if i == len(READER_ORDER) - 1: ax.set_xlabel("planner tokens")
                if j == 0: ax.set_ylabel("answer F1")
                if i == 0 and j == len(DATASET_ORDER) - 1: ax.legend(fontsize=6, loc="lower right")
                ax.grid(alpha=0.3)
        fig.suptitle("3-route RASER vs baselines/oracle (canonical, honest cost)", y=0.995, fontsize=11)
        plt.tight_layout()
        fig.savefig(out_root / "pareto_grid.png", dpi=120, bbox_inches="tight")
        fig.savefig(ROOT / "demo/pareto_4reader.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_root / 'pareto_grid.png'}  and  demo/pareto_4reader.png")
    except Exception as e:
        print(f"(pareto figure skipped: {e})")


if __name__ == "__main__":
    main()
