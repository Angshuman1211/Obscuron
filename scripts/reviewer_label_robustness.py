#!/usr/bin/env python
"""Does the dark-query result survive label noise and the length confound? (Review II, comments 2-3)

Uses results/reviewer/label_audit_hits.csv (scripts/label_audit.py). Re-scores the real dark-query
task (positives vs still-dark) for the layer-22 and layer-33 ESM-2 distance, and for protein length
alone, on (a) all labels, (b) stringent labels (positives at least 10 bits above their family's GA
and no Pfam-35 hit within 10 bits of GA; negatives with no Pfam-35 or Pfam-37 new-family hit within
10 bits of GA), and (c) within deciles of protein length, averaged with equal weight, which removes
whatever length explains.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reviewer_label_robustness.py
Writes results/reviewer/label_robustness.json.
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
from darkmatter.scoring.knn import knn_distance, l2_normalize  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
rng = np.random.default_rng(42)


def auc(y, s):
    r = stats.rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(y, s, n=500):
    v = [auc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def strat(y, s, length, q=10):
    b = pd.qcut(stats.rankdata(length, method="ordinal"), q, labels=False)
    a = [auc(y[b == k], s[b == k]) for k in range(q) if 0 < y[b == k].sum() < (b == k).sum()]
    return float(np.mean(a))


def main() -> None:
    aud = pd.read_csv(ROOT / "results" / "reviewer" / "label_audit_hits.csv").set_index("protein_id")
    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv").set_index("protein_id")
    es = pd.read_csv(PROC / "esm2_novelty_scores.csv").set_index("protein_id")
    qry = np.load(PROC / "esm2_reembed_queries_L33_22.npz", allow_pickle=True)
    ref = np.load(PROC / "esm2_reembed_L33_22.npz", allow_pickle=True)
    ids = [str(i) for i in qry["ids"]]
    y = qry["is_positive"].astype(int)
    d = pd.DataFrame({"y": y, "length": lens.loc[ids, "length_aa"].to_numpy()}, index=ids)
    for L in (22, 33):
        d[f"L{L}"] = knn_distance(l2_normalize(qry[f"L{L}"].astype(np.float64)),
                                  l2_normalize(ref[f"L{L}"].astype(np.float64)), k=5)
    a = aud.loc[ids]
    pos_ok = (a.m37_new >= 10) & (a.m35.isna() | (a.m35 < -10))
    neg_ok = (a.m35.isna() | (a.m35 < -10)) & (a.m37_new.isna() | (a.m37_new < -10))
    sets = {"all_labels": np.ones(len(d), bool),
            "stringent_labels": np.where(d.y == 1, pos_ok, neg_ok).astype(bool),
            "positives_with_no_pfam35_subthreshold_hit_vs_all_still_dark": np.where(d.y == 1, a.m35.isna(), True).astype(bool)}
    out = {}
    for name, m in sets.items():
        dd = d[m]; yy = dd.y.to_numpy()
        out[name] = {"n": int(len(dd)), "n_pos": int(yy.sum()), "base_rate": float(yy.mean())}
        for s in ("L22", "L33", "length"):
            v = dd[s].to_numpy()
            out[name][s] = {"auroc": auc(yy, v), "ci95": ci(yy, v), "within_length_decile_auroc": strat(yy, v, dd.length.to_numpy()) if s != "length" else None}
        print(name, {k: round(v["auroc"], 3) for k, v in out[name].items() if isinstance(v, dict)}, flush=True)

    # the 34,138 set, last layer, same checks
    y34 = es.is_positive.astype(str).eq("True").astype(int).to_numpy()
    ids34 = es.index
    dd = pd.DataFrame({"y": y34, "d": es.raw_knn_distance.to_numpy(), "length": lens.loc[ids34, "length_aa"].to_numpy()}, index=ids34)
    a34 = aud.reindex(ids34)
    pos_ok = (a34.m37_new >= 10) & (a34.m35.isna() | (a34.m35 < -10))
    neg_ok = (a34.m35.isna() | (a34.m35 < -10)) & (a34.m37_new.isna() | (a34.m37_new < -10))
    m = np.where(dd.y == 1, pos_ok, neg_ok).astype(bool)
    out["set_34138_last_layer"] = {
        "all": {"auroc": auc(dd.y.to_numpy(), dd.d.to_numpy()), "within_length_decile": strat(dd.y.to_numpy(), dd.d.to_numpy(), dd.length.to_numpy())},
        "stringent": {"n": int(m.sum()), "auroc": auc(dd.y.to_numpy()[m], dd.d.to_numpy()[m]),
                      "within_length_decile": strat(dd.y.to_numpy()[m], dd.d.to_numpy()[m], dd.length.to_numpy()[m])},
        "length_only_all": auc(dd.y.to_numpy(), dd.length.to_numpy())}
    (ROOT / "results" / "reviewer" / "label_robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["set_34138_last_layer"], indent=1))


if __name__ == "__main__":
    main()
