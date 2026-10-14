#!/usr/bin/env python
"""Protein length as a baseline for the dark-query task (Review II, comment 3/6).

AUROC of length alone for positives vs still-dark on the 34,138 set, with a bootstrap interval,
and the paired bootstrap difference from the last-layer ESM-2 kNN distance (the primary scorer there).

    .venv/Scripts/python.exe scripts/reviewer_length_baseline.py
Writes results/reviewer/length_baseline.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)


def auc(y, s):
    r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


es = pd.read_csv(ROOT / "data/processed/gtdb_R207/esm2_novelty_scores.csv").set_index("protein_id")
ln = pd.read_csv(ROOT / "results/scorer_esm2_ranked.csv").set_index("protein_id")["length_aa"]
y = es.is_positive.astype(str).eq("True").astype(int).to_numpy()
L = ln.reindex(es.index).to_numpy(float); D = es.raw_knn_distance.to_numpy()
v, dd = [], []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    a, b = auc(y[i], L[i]), auc(y[i], D[i]); v.append(a); dd.append(a - b)
out = {"n": int(len(y)), "length_auroc": auc(y, L), "length_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
       "esm2_last_layer_auroc": auc(y, D), "diff": auc(y, L) - auc(y, D),
       "diff_ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
       "median_length_positive": float(np.median(L[y == 1])), "median_length_still_dark": float(np.median(L[y == 0]))}
(ROOT / "results/reviewer/length_baseline.json").write_text(json.dumps(out, indent=2))
print(out)
