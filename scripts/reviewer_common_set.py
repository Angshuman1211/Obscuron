#!/usr/bin/env python
"""Layer 22 and the last layer on ONE query set and ONE reference (Review II, comments 1 and 3).

Query set: the 34,138 dark queries (4,138 positives + 30,000 still dark, base rate 0.1212).
Reference: the 29,862 proteins characterised at T0. Both layers use the same proteins, k = 5,
cosine distance, leave-one-out reference distances, GPD above the 90th percentile.
Needs esm2_reembed_*.npz and esm2_L22_L33_remaining.npz (scripts/embed_full_query_reference.py).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reviewer_common_set.py
Writes results/reviewer/common_set.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from darkmatter.scoring.gpd import diagnostics, fit_gpd_tail, tail_pvalue  # noqa: E402
from darkmatter.scoring.knn import knn_distance, l2_normalize  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
rng = np.random.default_rng(42)


def auc(y, s):
    r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    ref1 = np.load(PROC / "esm2_reembed_L33_22.npz")
    q1 = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
    rem = np.load(PROC / "esm2_L22_L33_remaining.npz")
    ids = {"ref": [str(i) for i in ref1["ids"]], "q": [str(i) for i in q1["ids"]], "rem": [str(i) for i in rem["ids"]]}
    cat = samp.set_index("protein_id").category
    allids = ids["ref"] + ids["q"] + ids["rem"]
    E = {L: np.concatenate([ref1[f"L{L}"], q1[f"L{L}"], rem[f"L{L}"]]).astype(np.float64) for L in (22, 33)}
    c = cat.reindex(allids).to_numpy()
    is_ref = c == "characterised_at_t0"; is_q = np.isin(c, ["positive", "dark_negative"])
    pid = np.array(allids)
    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv").set_index("protein_id")["length_aa"]
    y = (c[is_q] == "positive").astype(int); length = lens.reindex(pid[is_q]).to_numpy(float)
    out = {"n_reference": int(is_ref.sum()), "n_queries": int(is_q.sum()), "n_positive": int(y.sum()), "base_rate": float(y.mean())}
    for L in (22, 33):
        R = l2_normalize(E[L][is_ref]); Q = l2_normalize(E[L][is_q])
        loo = knn_distance(R, R, k=5, exclude_self=True)
        d = knn_distance(Q, R, k=5)
        tail = fit_gpd_tail(loo, 0.90); dg = diagnostics(loo, tail)
        order = np.argsort(-d, kind="stable")
        res = {"auroc": auc(y, d),
               "gpd": {"u": tail.u, "xi": tail.xi, "beta": tail.beta, "n_exceedances": tail.n_exceedances, "ks_p_naive": dg["ks_pvalue"]},
               "median_query_distance": float(np.median(d)), "median_reference_loo": float(np.median(loo)),
               "frac_queries_p_le_0.05": float((tail_pvalue(tail, d, ref_distances=loo) <= 0.05).mean())}
        v = [auc(y[i], d[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
        res["auroc_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res["precision_at_k"] = [{"K": k, "precision": float(y[order[:k]].mean()), "lift": float(y[order[:k]].mean() / y.mean())}
                                 for k in (50, 100, 500, 1000, 2000, 5000)]
        b = pd.qcut(stats.rankdata(length, method="ordinal"), 10, labels=False)
        res["within_length_decile_auroc"] = float(np.mean([auc(y[b == k], d[b == k]) for k in range(10)]))
        ranks = np.empty(len(d), int); ranks[order] = np.arange(len(d))
        res["deciles_positive_rate"] = [float(y[(ranks >= i * len(d) // 10) & (ranks < (i + 1) * len(d) // 10)].mean()) for i in range(10)]
        out[f"layer{L}"] = res
        print(L, res["auroc"], res["gpd"], [r["lift"] for r in res["precision_at_k"]], flush=True)
    out["length_only_auroc"] = auc(y, length)
    (ROOT / "results" / "reviewer" / "common_set.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
