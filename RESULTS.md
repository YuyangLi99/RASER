# Paper results rendered from `results/*/summary.json`

This file renders the paper's main tables directly from the JSON
summary files in this repo, so reviewers can compare them to the paper
without parsing JSON by hand. All numbers below come from running the
scripts in this repo on the trace files described in
[README](README.md#3-reproducing-the-paper).

## Table 2 — Main results (F1 / mean tokens per question)

**Source**: `results/three_route_canonical/summary.json`

Columns: STOP = one-shot RAG · a-PRUNE = always-PRUNE · IRCoT\* = always-IRCoT ·
Self-Ask\* = always-Self-Ask · **RASER-2** / **RASER-3** = the routers this paper
contributes. (ChainRAG column in the paper is sourced from a separate report
under `outputs/chainrag_faithful/` and is not included here for compactness.)

| Reader / Dataset | STOP F1 | tk | a-PRUNE F1 | tk | IRCoT\* F1 | tk | Self-Ask\* F1 | tk | **R2 F1** | **tk** | **R3 F1** | **tk** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **_GPT-OSS-120B_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.488 | 1.5k | 0.528 | 4.8k | 0.455 | 4.3k | 0.482 | 4.6k | **0.502** | **2.1k** | **0.510** | **2.2k** |
| &nbsp;&nbsp;2Wiki | 0.754 | 1.1k | 0.761 | 3.3k | 0.781 | 3.0k | 0.743 | 3.5k | **0.763** | **1.3k** | **0.774** | **1.6k** |
| &nbsp;&nbsp;HotpotQA | 0.777 | 1.4k | 0.769 | 3.7k | 0.763 | 3.6k | 0.724 | 4.1k | **0.779** | **1.5k** | **0.787** | **1.8k** |
| **_Mistral-S-119B_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.283 | 1.2k | 0.397 | 3.8k | 0.499 | 4.3k | 0.380 | 3.3k | **0.306** | **1.9k** | **0.408** | **2.4k** |
| &nbsp;&nbsp;2Wiki | 0.450 | 969 | 0.570 | 2.9k | 0.658 | 2.9k | 0.562 | 2.8k | **0.485** | **1.5k** | **0.451** | **979** |
| &nbsp;&nbsp;HotpotQA | 0.699 | 1.3k | 0.716 | 3.4k | 0.773 | 3.4k | 0.638 | 3.3k | **0.702** | **1.4k** | **0.723** | **2.2k** |
| **_Gemma-3-31B_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.393 | 1.2k | 0.503 | 3.7k | 0.577 | 3.8k | 0.360 | 3.0k | **0.458** | **2.1k** | **0.502** | **2.4k** |
| &nbsp;&nbsp;2Wiki | 0.630 | 961 | 0.694 | 2.8k | 0.722 | 2.5k | 0.664 | 2.5k | **0.656** | **1.2k** | **0.700** | **1.6k** |
| &nbsp;&nbsp;HotpotQA | 0.787 | 1.3k | 0.797 | 3.4k | 0.810 | 3.0k | 0.734 | 2.8k | **0.796** | **1.4k** | **0.812** | **1.8k** |
| **_Llama-3-8B_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.223 | 1.2k | 0.256 | 3.7k | 0.354 | 4.7k | 0.292 | 3.2k | **0.229** | **1.6k** | **0.309** | **2.8k** |
| &nbsp;&nbsp;2Wiki | 0.367 | 931 | 0.406 | 2.8k | 0.507 | 3.5k | 0.460 | 2.2k | **0.380** | **1.2k** | **0.472** | **2.1k** |
| &nbsp;&nbsp;HotpotQA | 0.632 | 1.2k | 0.656 | 3.3k | 0.688 | 4.1k | 0.621 | 3.5k | **0.635** | **1.3k** | **0.655** | **2.2k** |
| **_Llama-3.1-8B_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.236 | 1.2k | 0.264 | 3.8k | 0.376 | 5.1k | 0.276 | 3.2k | **0.253** | **1.5k** | **0.330** | **3.0k** |
| &nbsp;&nbsp;2Wiki | 0.307 | 954 | 0.387 | 2.9k | 0.532 | 4.0k | 0.353 | 2.3k | **0.355** | **1.4k** | **0.487** | **2.2k** |
| &nbsp;&nbsp;HotpotQA | 0.612 | 1.3k | 0.626 | 3.4k | 0.676 | 4.8k | 0.625 | 3.6k | **0.624** | **1.5k** | **0.654** | **2.4k** |
| **_Phi-4-mini_** | | | | | | | | | | | | |
| &nbsp;&nbsp;MuSiQue | 0.331 | 1.1k | 0.353 | 3.6k | 0.372 | 5.4k | 0.407 | 3.0k | **0.335** | **1.4k** | **0.357** | **2.7k** |
| &nbsp;&nbsp;2Wiki | 0.391 | 903 | 0.459 | 2.6k | 0.532 | 3.8k | 0.457 | 2.6k | **0.402** | **1.1k** | **0.474** | **2.3k** |
| &nbsp;&nbsp;HotpotQA | 0.634 | 1.2k | 0.643 | 3.2k | 0.670 | 4.7k | 0.577 | 3.2k | **0.635** | **1.4k** | **0.652** | **2.2k** |

## Table 5 — RASER-2 threshold sensitivity

**Source**: `results/sensitivity/summary.json` (`theta_sweep`)

Pooled over 6 LLMs × 3 datasets ($N{=}5{,}500$ questions).

| θ | F1 | Avg tokens | Escalation % |
|---|---:|---:|---:|
| 0.10 | 0.530 | 1756 | 24.0% |
| 0.15 | 0.525 | 1604 | 17.9% |
| **0.20 (deployed)** | **0.520** | **1514** | **14.1%** |
| 0.25 | 0.518 | 1455 | 11.6% |
| 0.30 | 0.516 | 1407 | 9.7% |

## Table 6 — RASER-3 cost-budget sensitivity

**Source**: `results/sensitivity/summary.json` (`cost_frac_sweep`)

Pooled over 6 LLMs × 3 datasets. Route rate = STOP / PRUNE / IRCoT\* %.

| Cost budget | F1 | Avg tokens | Route rate |
|---|---:|---:|---:|
| 0.33 | 0.504 | 1190 | 99 / 1 / 0 |
| 0.50 | 0.525 | 1653 | 82 / 7 / 11 |
| **0.60 (deployed)** | **0.562** | **2157** | **64 / 12 / 24** |
| 0.75 | 0.578 | 2665 | 44 / 19 / 37 |
| 1.00 | 0.583 | 3141 | 25 / 28 / 47 |

## Table 12 — Router head ablation

**Source**: `results/router_model_ablation/summary.json`

Pooled over 6 LLMs × 3 datasets. The deployed sklearn GBM is highlighted; all
alternatives fall within the 95% bootstrap CI on F1, so the result does not depend
on the choice of head model.

### RASER-2 (binary classifier head)

| Head model | F1 | Avg tokens | Escalation % |
|---|---:|---:|---:|
| **sklearn GBM (deployed)** | **0.520** | **1514** | **14.1%** |
| LogReg (scaled) | 0.531 | 1549 | 15.4% |
| MLP-32 (scaled) | 0.530 | 1597 | 17.6% |
| XGBoost | 0.526 | 1575 | 16.8% |
| LightGBM | 0.526 | 1607 | 18.1% |
| CatBoost | 0.529 | 1555 | 15.5% |

### RASER-3 (regressor heads)

Route mix = STOP / PRUNE / IRCoT\* %.

| Head model | F1 | Avg tokens | Route mix |
|---|---:|---:|---:|
| **sklearn GBM (deployed)** | **0.562** | **2157** | **64 / 12 / 24** |
| Ridge (scaled) | 0.546 | 1844 | 76 / 5 / 18 |
| MLP-32 (scaled) | 0.559 | 1908 | 74 / 6 / 19 |
| XGBoost | 0.562 | 2094 | 67 / 10 / 23 |
| LightGBM | 0.558 | 1937 | 73 / 9 / 18 |
| CatBoost | 0.562 | 1989 | 72 / 7 / 21 |

---

## How to regenerate these tables

```bash
python -m src.eval.three_route_canonical          # paper Table 2
python -m src.eval.router_model_ablation          # paper Table 12
python -m src.eval.sensitivity_sweep              # paper Tables 5 + 6
```
Each script reads its trace files from `outputs/sweep_*/` and writes a
fresh `summary.json` under `results/` matching what is in this repo.
