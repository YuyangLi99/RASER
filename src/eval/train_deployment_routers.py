"""Train deployment routers per (LLM, dataset) cell and save checkpoints.

For each (LLM, dataset) cell, this script:
  - Trains one RASER-2 binary GBM classifier on ALL questions in the cell
    (no held-out split; this is the "deployment" classifier the operator
    would actually use).
  - Trains three RASER-3 GBM regressors, one per route, on ALL questions
    in the cell.
  - Derives the deployed lambda from the cost-budget rule on training
    data (60% of always-IRCoT* tokens).

Output: checkpoints/<reader>/<dataset>/
  - raser2_classifier.pkl       sklearn GBM binary classifier
  - raser3_stop.pkl             sklearn GBM regressor (STOP route)
  - raser3_prune.pkl            sklearn GBM regressor (PRUNE route)
  - raser3_iter.pkl             sklearn GBM regressor (IRCoT* route)
  - metadata.json               feature_names, qtype_map, threshold,
                                deployed_lambda, route_costs, training_n,
                                bridgeable_rate, etc.

Usage:
  python -m src.eval.train_deployment_routers

Requires the trace files described in src/eval/three_route_feasibility.py
to be present under outputs/.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.three_route_canonical import (
    READER_ORDER, ROUTES, build_table, LAMBDA_SWEEP,
)

THETA = 0.20
COST_BUDGET_FRAC = 0.60
TAU = 0.10  # bridgeable margin for R2 label
SEED = 42

FEATURE_NAMES = ["confidence", "ans_len", "bridge_cues",
                 "score_gap", "score_top1", "qtype"]
QTYPE_MAP = {"entity": 0, "date": 1, "yes_no": 2, "count": 3, "other": 4}


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


def train_one_cell(view, reader, dataset, out_dir: Path):
    qk = sorted(view); n = len(qk)
    X = np.array([view[q]["X"] for q in qk], dtype=float)
    F1 = {r: np.array([view[q]["F1"][r] for q in qk]) for r in ROUTES}
    TOK = {r: np.array([view[q]["TOK"][r] for q in qk]) for r in ROUTES}
    y = (F1["PRUNE"] > F1["STOP"] + TAU - 1e-6).astype(int)

    # ---- RASER-2 ----
    if y.sum() < 3 or (n - y.sum()) < 3:
        return {"skipped": True, "reason": "too few positives or negatives"}
    r2 = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=SEED,
    ).fit(X, y)
    r2_path = out_dir / "raser2_classifier.pkl"
    pickle.dump(r2, open(r2_path, "wb"))

    # ---- RASER-3 ----
    r3 = {}
    for r in ROUTES:
        g = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=SEED,
        ).fit(X, F1[r])
        r3[r] = g
    # Derive deployed lambda from cost-budget rule (using training-data predictions)
    pred_tr = {r: r3[r].predict(X) for r in ROUTES}
    cost_tr = {r: float(TOK[r].mean()) for r in ROUTES}
    deployed_lambda = _pick_lambda(
        pred_tr, TOK, cost_tr, COST_BUDGET_FRAC, cost_tr["ITER"]
    )
    label_map = {"STOP": "stop", "PRUNE": "prune", "ITER": "iter"}
    for r in ROUTES:
        pickle.dump(r3[r], open(out_dir / f"raser3_{label_map[r]}.pkl", "wb"))

    metadata = {
        "reader": reader,
        "dataset": dataset,
        "n_training_questions": n,
        "feature_names": FEATURE_NAMES,
        "qtype_map": QTYPE_MAP,
        "raser2": {
            "threshold_theta": THETA,
            "bridgeable_label_tau": TAU,
            "n_bridgeable_positives": int(y.sum()),
            "bridgeable_rate": float(y.mean()),
        },
        "raser3": {
            "cost_budget_fraction": COST_BUDGET_FRAC,
            "deployed_lambda": float(deployed_lambda),
            "training_route_costs": {r: cost_tr[r] for r in ROUTES},
            "training_route_f1": {r: float(F1[r].mean()) for r in ROUTES},
        },
        "gbm_hyperparameters": {
            "n_estimators": 100, "max_depth": 3,
            "learning_rate": 0.1, "subsample": 0.8, "random_state": SEED,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"skipped": False, **metadata}


def main():
    ckpt_root = ROOT / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

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
    print(f"  Loaded {n_cells} cells across {len(tables)} LLMs\n")

    summary = []
    for reader, by_ds in tables.items():
        # File-safe reader name
        safe_reader = reader.replace(".", "_").replace("/", "_")
        for ds, view in by_ds.items():
            safe_ds = ds.replace(".", "_").replace("/", "_")
            out_dir = ckpt_root / safe_reader / safe_ds
            out_dir.mkdir(parents=True, exist_ok=True)
            res = train_one_cell(view, reader, ds, out_dir)
            if res.get("skipped"):
                print(f"  [skip] {reader} / {ds}: {res['reason']}")
            else:
                print(f"  [ok]   {reader} / {ds}: "
                      f"n={res['n_training_questions']}, "
                      f"bridgeable_rate={res['raser2']['bridgeable_rate']:.2f}, "
                      f"lambda={res['raser3']['deployed_lambda']:.0e}")
                summary.append(res)

    (ckpt_root / "INDEX.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} per-cell checkpoint sets to {ckpt_root}")


if __name__ == "__main__":
    main()
