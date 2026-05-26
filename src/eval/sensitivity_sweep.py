"""Threshold (theta) and cost-budget sensitivity sweep.

For each theta in {0.10, 0.15, 0.20, 0.25, 0.30}, re-evaluate RASER-2
with the deployed sklearn GBM classifier, keeping everything else
fixed (same 6 features, same 5-fold CV per (LLM, dataset) cell).

For each cost-budget fraction in {0.33, 0.50, 0.60, 0.75, 1.00},
re-evaluate RASER-3 with the deployed sklearn GBM regressors, keeping
everything else fixed. The cost-budget rule sets lambda such that
training-fold spend stays <= frac * always-IRCoT* tokens.

Output: outputs/sensitivity/summary.json with (theta_sweep, cost_sweep)
and prints a Markdown table at the end.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.three_route_canonical import (
    READER_ORDER, DATASET_ORDER, ROUTES, ROUTE_LABEL,
    build_table, LAMBDA_SWEEP,
)

THETA_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30]
COST_FRAC_SWEEP = [0.33, 0.50, 0.60, 0.75, 1.00]
SEED = 42


def make_clf():
    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=SEED,
    )


def make_reg():
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=SEED,
    )


def eval_r2_cell(view, theta):
    """Re-fit binary classifier with given theta, return per-cell stats."""
    qk = sorted(view); n = len(qk)
    X = np.array([view[q]["X"] for q in qk], dtype=float)
    F1 = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ["STOP", "PRUNE"]}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ["STOP", "PRUNE"]}
    y = (F1["PRUNE"] > F1["STOP"] + 1e-6).astype(int)
    if y.sum() < 3 or (n - y.sum()) < 3:
        return None
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    routed_f1, routed_tok, esc = 0.0, 0.0, 0
    for tr, te in skf.split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        decisions = (p >= theta).astype(int)
        for j, idx in enumerate(te):
            r = "PRUNE" if decisions[j] == 1 else "STOP"
            routed_f1 += F1[r][idx]
            routed_tok += TOK[r][idx]
            if decisions[j] == 1:
                esc += 1
    return {"f1": routed_f1 / n, "tok": routed_tok / n, "esc_pct": 100.0 * esc / n}


def _route_choice(pred, cost_r, lam):
    n = len(pred[ROUTES[0]])
    scores = np.stack(
        [pred[r] - lam * cost_r[r] * np.ones(n) for r in ROUTES], axis=1
    )
    return [ROUTES[k] for k in scores.argmax(axis=1)]


def _pick_lambda(pred_tr, TOK_tr, cost_tr, frac, ai_tok_tr):
    target = frac * ai_tok_tr
    chosen = LAMBDA_SWEEP[-1]
    for lam in LAMBDA_SWEEP:
        ch = _route_choice(pred_tr, cost_tr, lam)
        tm = float(np.mean([TOK_tr[ch[k]][k] for k in range(len(ch))]))
        if tm <= target:
            chosen = lam
            break
    return chosen


def eval_r3_cell(view, cost_frac):
    """Re-fit three regressors, pick lambda via cost-budget rule with given frac."""
    qk = sorted(view); n = len(qk)
    X = np.array([view[q]["X"] for q in qk], dtype=float)
    Y = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ROUTES}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ROUTES}
    kf = KFold(5, shuffle=True, random_state=SEED)
    pred = {r: np.zeros(n) for r in ROUTES}
    op_choices = [None] * n
    for tr, te in kf.split(X):
        regs = {}
        for r in ROUTES:
            g = make_reg().fit(X[tr], Y[r][tr])
            regs[r] = g
            pred[r][te] = g.predict(X[te])
        cost_tr = {r: float(TOK[r][tr].mean()) for r in ROUTES}
        ai = cost_tr["ITER"]
        pred_tr = {r: regs[r].predict(X[tr]) for r in ROUTES}
        TOK_tr = {r: TOK[r][tr] for r in ROUTES}
        lam = _pick_lambda(pred_tr, TOK_tr, cost_tr, cost_frac, ai)
        cht = _route_choice({r: pred[r][te] for r in ROUTES}, cost_tr, lam)
        for j, idx in enumerate(te):
            op_choices[idx] = cht[j]
    routed_f1 = float(np.mean([Y[op_choices[k]][k] for k in range(n)]))
    routed_tok = float(np.mean([TOK[op_choices[k]][k] for k in range(n)]))
    rd = Counter(op_choices)
    return {
        "f1": routed_f1,
        "tok": routed_tok,
        "route_pct": {ROUTE_LABEL[r]: 100.0 * rd.get(r, 0) / n for r in ROUTES},
    }


def main():
    out_root = ROOT / "outputs/sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading cell tables ...")
    tables = {}
    for reader in READER_ORDER:
        t = build_table(reader)
        if not t:
            continue
        by_ds = {}
        for k, d in t.items():
            by_ds.setdefault(d["dataset"], {})[k] = d
        tables[reader] = by_ds
    n_cells = sum(len(v) for v in tables.values())
    print(f"  loaded {n_cells} cells across {len(tables)} LLMs")

    summary = {"theta_sweep": [], "cost_frac_sweep": []}

    # ---- theta sweep ----
    print("\n=== RASER-2 threshold sweep ===")
    for theta in THETA_SWEEP:
        f1_sum = tok_sum = esc_sum = N = 0
        for reader, by_ds in tables.items():
            for ds, view in by_ds.items():
                res = eval_r2_cell(view, theta)
                if res is None:
                    continue
                n = len(view)
                f1_sum += res["f1"] * n
                tok_sum += res["tok"] * n
                esc_sum += res["esc_pct"] * n
                N += n
        f1 = f1_sum / N; tok = tok_sum / N; esc = esc_sum / N
        print(f"  theta = {theta:.2f}  ->  F1 = {f1:.3f}  tok = {tok:>5.0f}  esc = {esc:>4.1f}%")
        summary["theta_sweep"].append(
            {"theta": theta, "f1": f1, "tok": tok, "esc_pct": esc}
        )

    # ---- cost-budget sweep ----
    print("\n=== RASER-3 cost-budget sweep ===")
    for frac in COST_FRAC_SWEEP:
        f1_sum = tok_sum = N = 0
        route_mix = Counter()
        for reader, by_ds in tables.items():
            for ds, view in by_ds.items():
                res = eval_r3_cell(view, frac)
                n = len(view)
                f1_sum += res["f1"] * n
                tok_sum += res["tok"] * n
                N += n
                for r, p in res["route_pct"].items():
                    route_mix[r] += p * n
        f1 = f1_sum / N; tok = tok_sum / N
        mix = {r: route_mix[r] / N for r in route_mix}
        mix_str = " ".join(
            f"{r}={mix.get(r,0):.0f}%" for r in ["STOP", "TOP2_PRUNE", "ITER_RETRIEVE"]
        )
        print(f"  frac = {frac:.2f}  ->  F1 = {f1:.3f}  tok = {tok:>5.0f}  {mix_str}")
        summary["cost_frac_sweep"].append(
            {"cost_frac": frac, "f1": f1, "tok": tok, "route_pct": mix}
        )

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
