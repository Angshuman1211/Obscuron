#!/usr/bin/env python
"""Density-normalised score on the retrospective task (34,138 dark queries, 29,862 references, layer 22).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/revision_normalised_dark.py
Writes results/revision/normalised_dark.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.scoring.knn import knn_distance, l2_normalize  # noqa: E402
from revision_scoring import scores_against, topk  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
rng = np.random.default_rng(42)


def auc(y, s):
    r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
r1 = np.load(PROC / "esm2_reembed_L33_22.npz"); q1 = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
rem = np.load(PROC / "esm2_L22_L33_remaining.npz")
ids = np.array([str(i) for i in np.r_[r1["ids"], q1["ids"], rem["ids"]]])
E = np.concatenate([r1["L22"], q1["L22"], rem["L22"]]).astype(np.float64)
c = samp.category.reindex(ids).to_numpy()
R = l2_normalize(E[c == "characterised_at_t0"]); Q = l2_normalize(E[np.isin(c, ["positive", "dark_negative"])])
y = (c[np.isin(c, ["positive", "dark_negative"])] == "positive").astype(int)
loo = knn_distance(R, R, k=5, exclude_self=True)
d, nu = scores_against(Q, R, loo)
v = [auc(y[i], nu[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
order = np.argsort(-nu, kind="stable")
res = {"auroc_normalised": auc(y, nu), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
       "auroc_distance": auc(y, d), "spearman_normalised_vs_distance": float(stats.spearmanr(nu, d)[0]),
       "precision_at_k": {str(k): float(y[order[:k]].mean()) for k in (100, 500, 1000)}, "base_rate": float(y.mean())}
(ROOT / "results" / "revision" / "normalised_dark.json").write_text(json.dumps(res, indent=2))
print(res)
