"""Unified classifier/regressor ablation for RASER-2 and RASER-3.

For each model family we re-fit the deployed router heads, keeping
everything else fixed (same 6 features, same 5-fold CV per
(LLM, dataset) cell, same threshold/cost-budget rule), and report the
N-weighted pooled routed F1 + tokens.

RASER-2 ablation: swap the binary classifier head.
RASER-3 ablation: swap the regressor head (we train 3 of them, one per
route, and combine via the same cost-aware argmax with the same
training-fold cost-budget lambda rule).

Output: outputs/router_model_ablation/summary.json with one row per
(variant, model). Prints a combined Markdown table at the end.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.three_route_canonical import (
    READER_ORDER, DATASET_ORDER, ROUTES, ROUTE_LABEL,
    build_table, LAMBDA_SWEEP,
)

# Threshold for RASER-2 escalation; lambda budget rule fraction for RASER-3
THETA = 0.20
COST_BUDGET_FRAC = 0.60  # balanced operating point
SEED = 42

# ---------------- model factories -------------------------------------

def clf_models():
    """Binary classifiers for RASER-2 ablation."""
    out = [
        ("sklearn GBM (deployed)",
         lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.1, subsample=0.8,
                                            random_state=SEED), False),
        ("LogReg (scaled)",
         lambda: LogisticRegression(max_iter=2000, random_state=SEED), True),
        ("MLP-32 (scaled)",
         lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                               random_state=SEED), True),
    ]
    try:
        from xgboost import XGBClassifier
        out.append(("XGBoost",
                    lambda: XGBClassifier(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, subsample=0.8,
                                          eval_metric="logloss",
                                          use_label_encoder=False,
                                          random_state=SEED), False))
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier
        out.append(("LightGBM",
                    lambda: LGBMClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, subsample=0.8,
                                           verbose=-1, random_state=SEED), False))
    except Exception: pass
    try:
        from catboost import CatBoostClassifier
        out.append(("CatBoost",
                    lambda: CatBoostClassifier(iterations=100, depth=3,
                                               learning_rate=0.1, subsample=0.8,
                                               bootstrap_type="Bernoulli",
                                               verbose=0, random_seed=SEED), False))
    except Exception: pass
    return out


def reg_models():
    """Regressors for RASER-3 ablation."""
    out = [
        ("sklearn GBM (deployed)",
         lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, subsample=0.8,
                                           random_state=SEED), False),
        ("Ridge (scaled)",
         lambda: Ridge(alpha=1.0, random_state=SEED), True),
        ("MLP-32 (scaled)",
         lambda: MLPRegressor(hidden_layer_sizes=(32,), max_iter=2000,
                              random_state=SEED), True),
    ]
    try:
        from xgboost import XGBRegressor
        out.append(("XGBoost",
                    lambda: XGBRegressor(n_estimators=100, max_depth=3,
                                         learning_rate=0.1, subsample=0.8,
                                         random_state=SEED), False))
    except Exception: pass
    try:
        from lightgbm import LGBMRegressor
        out.append(("LightGBM",
                    lambda: LGBMRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, subsample=0.8,
                                          verbose=-1, random_state=SEED), False))
    except Exception: pass
    try:
        from catboost import CatBoostRegressor
        out.append(("CatBoost",
                    lambda: CatBoostRegressor(iterations=100, depth=3,
                                              learning_rate=0.1, subsample=0.8,
                                              bootstrap_type="Bernoulli",
                                              verbose=0, random_seed=SEED), False))
    except Exception: pass
    return out


# ---------------- R2 evaluation per cell ------------------------------

def eval_r2_cell(view, make_clf, scale):
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
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = make_clf(); clf.fit(Xtr, y[tr])
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(Xte)[:, 1]
        else:
            d = clf.decision_function(Xte); p = 1.0 / (1.0 + np.exp(-d))
        decisions = (p >= THETA).astype(int)
        for j, idx in enumerate(te):
            r = "PRUNE" if decisions[j] == 1 else "STOP"
            routed_f1 += F1[r][idx]; routed_tok += TOK[r][idx]
            if decisions[j] == 1: esc += 1
    return {"f1": float(routed_f1 / n), "tok": float(routed_tok / n),
            "esc_pct": 100.0 * esc / n}


# ---------------- R3 evaluation per cell ------------------------------

def _route_choice(pred, cost_r, lam):
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


def eval_r3_cell(view, make_reg, scale):
    qk = sorted(view); n = len(qk)
    X = np.array([view[q]["X"] for q in qk], dtype=float)
    Y = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ROUTES}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ROUTES}
    kf = KFold(5, shuffle=True, random_state=SEED)
    pred = {r: np.zeros(n) for r in ROUTES}
    op_choices = [None] * n
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        regs = {}
        for r in ROUTES:
            g = make_reg(); g.fit(Xtr, Y[r][tr]); regs[r] = g
            pred[r][te] = g.predict(Xte)
        cost_tr = {r: float(TOK[r][tr].mean()) for r in ROUTES}
        ai = cost_tr["ITER"]
        pred_tr = {r: regs[r].predict(Xtr) for r in ROUTES}
        TOK_tr = {r: TOK[r][tr] for r in ROUTES}
        lam = _pick_lambda(pred_tr, TOK_tr, cost_tr, COST_BUDGET_FRAC, ai)
        cht = _route_choice({r: pred[r][te] for r in ROUTES}, cost_tr, lam)
        for j, idx in enumerate(te):
            op_choices[idx] = cht[j]
    routed_f1 = float(np.mean([Y[op_choices[k]][k] for k in range(n)]))
    routed_tok = float(np.mean([TOK[op_choices[k]][k] for k in range(n)]))
    rd = Counter(op_choices)
    return {"f1": routed_f1, "tok": routed_tok,
            "route_pct": {ROUTE_LABEL[r]: 100.0 * rd.get(r, 0) / n for r in ROUTES}}


# ---------------- driver ----------------------------------------------

def main():
    out_root = ROOT / "outputs/router_model_ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    # Build tables once per LLM (shared across model variants)
    print("Loading cell tables ...")
    tables = {}
    for reader in READER_ORDER:
        t = build_table(reader)
        if not t: continue
        by_ds = {}
        for k, d in t.items(): by_ds.setdefault(d["dataset"], {})[k] = d
        tables[reader] = by_ds
    print(f"  loaded {sum(len(v) for v in tables.values())} cells "
          f"across {len(tables)} LLMs")

    summary = {"raser_2": [], "raser_3": []}

    # ---- RASER-2 -----------------------------------------------------
    print("\n=== RASER-2 (classifier) ablation ===")
    for name, factory, scale in clf_models():
        f1_sum = tok_sum = esc_sum = N = 0
        for reader, by_ds in tables.items():
            for ds, view in by_ds.items():
                res = eval_r2_cell(view, factory, scale)
                if res is None: continue
                n = len(view)
                f1_sum += res["f1"] * n; tok_sum += res["tok"] * n
                esc_sum += res["esc_pct"] * n; N += n
        f1, tok, esc = f1_sum/N, tok_sum/N, esc_sum/N
        print(f"  {name:28s}  routed F1 = {f1:.3f}   esc = {esc:>4.1f}%   tok = {tok:>5.0f}")
        summary["raser_2"].append({"model": name, "f1": f1, "tok": tok, "esc_pct": esc})

    # ---- RASER-3 -----------------------------------------------------
    print("\n=== RASER-3 (regressor) ablation ===")
    for name, factory, scale in reg_models():
        f1_sum = tok_sum = N = 0
        route_mix = Counter()
        for reader, by_ds in tables.items():
            for ds, view in by_ds.items():
                res = eval_r3_cell(view, factory, scale)
                n = len(view)
                f1_sum += res["f1"] * n; tok_sum += res["tok"] * n; N += n
                for r, p in res["route_pct"].items(): route_mix[r] += p * n
        f1, tok = f1_sum/N, tok_sum/N
        mix = {r: route_mix[r] / N for r in route_mix}
        mix_str = " ".join(f"{r}={mix.get(r,0):.0f}%" for r in ["STOP", "TOP2_PRUNE", "ITER_RETRIEVE"])
        print(f"  {name:28s}  routed F1 = {f1:.3f}   tok = {tok:>5.0f}   {mix_str}")
        summary["raser_3"].append({"model": name, "f1": f1, "tok": tok, "route_pct": mix})

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
