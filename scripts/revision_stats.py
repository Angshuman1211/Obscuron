#!/usr/bin/env python
"""Statistical fixes requested by the simulated TCBB review (track-2-research-extension).

1. Multiple testing: every hypothesis test reported in the manuscript is collected into one
   family and adjusted with Benjamini-Hochberg (false discovery rate 0.05); adjusted values are
   reported next to the raw ones.
2. Convergence nulls: the Gaussian copula of results/reviewer/corroboration_stats.json is joined by
   (a) an exact permutation null (each signal's ranking shuffled independently, which is the
   assumption-free independence null) and (b) a t copula with 4 degrees of freedom, which keeps the
   measured rank correlations but puts more mass on joint extremes than a Gaussian does. If the
   three-way excess survives the t copula, the conclusion does not hinge on the Gaussian choice.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/revision_stats.py
Writes results/revision/stats.json.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from darkmatter.convergence import genomic_context_novelty  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
SIMS = 400


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def signals():
    es = pd.read_csv(PROC / "esm2_novelty_scores.csv")
    d = es[["protein_id", "raw_knn_distance", "is_positive"]].rename(columns={"raw_knn_distance": "evt"})
    d["y"] = d.is_positive.astype(str).eq("True").astype(int)
    for f, c, n in (("composition_novelty.csv", "composition_novelty", "composition"),
                    ("prostt5_query_novelty.csv", "structural_novelty", "prostt5")):
        d = d.merge(pd.read_csv(PROC / f)[["protein_id", c]].rename(columns={c: n}), on="protein_id", how="left")
    g = pd.read_csv(ROOT / "results" / "genos-m_novelty_scores.csv").rename(columns={"genos_m_novelty": "genos-m"})
    d = d.merge(g, on="protein_id", how="left")
    lab = pd.read_csv(PROC / "panel_protein_labels.csv", usecols=["protein_id", "dark_at_t0"])
    d["context"] = d.protein_id.map(genomic_context_novelty(lab, 5, 4))
    names = ["evt", "context", "composition", "genos-m", "prostt5"]
    d = d.dropna(subset=names).reset_index(drop=True)
    return d, names


def convergence(res):
    d, names = signals()
    X = d[names].to_numpy(); N = len(X); y = d.y.to_numpy()
    top = np.column_stack([X[:, j] >= np.quantile(X[:, j], 0.90) for j in range(X.shape[1])])
    share = top.mean(0)
    # copula correlation from Kendall's tau: sin(pi*tau/2) holds for every elliptical copula,
    # so the Gaussian and t nulls share one correctly converted matrix
    k = X.shape[1]
    tau = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            tau[i, j] = tau[j, i] = stats.kendalltau(X[:, i], X[:, j]).statistic
    R = np.sin(np.pi * tau / 2)
    res_tau = tau
    L = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
    thr_g = stats.norm.ppf(1 - share)
    nu = 4
    thr_t = stats.t.ppf(1 - share, df=nu)
    combos = [c for r_ in (2, 3, 4, 5) for c in combinations(range(len(names)), r_)]
    null = {k: {c: [] for c in combos} for k in ("permutation", "gaussian", "t4")}
    for _ in range(SIMS):
        perm_top = np.column_stack([rng.permutation(top[:, j]) for j in range(top.shape[1])])
        Z = rng.standard_normal((N, len(names))) @ L.T
        W = Z / np.sqrt(rng.chisquare(nu, size=(N, 1)) / nu)
        g_top, t_top = Z >= thr_g, W >= thr_t
        for c in combos:
            cc = list(c)
            null["permutation"][c].append(int(perm_top[:, cc].all(1).sum()))
            null["gaussian"][c].append(int(g_top[:, cc].all(1).sum()))
            null["t4"][c].append(int(t_top[:, cc].all(1).sum()))
    rows = []
    for c in combos:
        m = top[:, list(c)].all(1); obs = int(m.sum()); k = int(y[m].sum())
        row = {"axes": [names[i] for i in c], "observed": obs, "positives": k,
               "p_enrichment": float(stats.hypergeom.sf(k - 1, N, int(y.sum()), obs)) if obs else 1.0}
        for nm in null:
            arr = np.array(null[nm][c])
            row[f"{nm}_mean"] = float(arr.mean()); row[f"{nm}_p"] = float((1 + (arr >= obs).sum()) / (SIMS + 1))
        rows.append(row)
    for key in ("permutation_p", "gaussian_p", "t4_p", "p_enrichment"):
        adj = bh([r[key] for r in rows])
        for r, a in zip(rows, adj):
            r[key + "_bh"] = float(a)
    res["convergence"] = {"n": int(N), "kendall_tau": res_tau.tolist(), "axes": names, "rows": rows}
    tri = [r for r in rows if r["axes"] == ["evt", "context", "composition"]][0]
    print("3-way evt+context+composition:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tri.items()}, flush=True)
    print("any intersection enriched after BH:", any(r["p_enrichment_bh"] < 0.05 for r in rows), flush=True)


def multiple_testing(res):
    """Collect the reported hypothesis tests into one family and adjust them together."""
    co = json.load(open(ROOT / "results" / "reviewer" / "corroboration_stats.json"))
    tests = []
    for sec in ("per_signal_34138", "per_signal_14138"):
        for name, v in co[sec].items():
            if isinstance(v, dict) and "mannwhitney_p" in v:
                tests.append((f"{sec}:{name}:mannwhitney", v["mannwhitney_p"]))
                if "vs_primary" in v:
                    tests.append((f"{sec}:{name}:paired_vs_primary", max(v["vs_primary"]["p_boot"], 1 / 1001)))
    im = co["immune_flag"]
    tests += [("immune:fisher", im["fisher_p"]), ("immune:cmh", im["cmh_stratified_by_layer1_decile"]["p"]),
              ("prostt5_vs_esm2:wilcoxon", co["prostt5_vs_esm2_heldout"]["wilcoxon_p"])]
    for r in res["convergence"]["rows"]:
        tests.append(("convergence:" + "+".join(r["axes"]) + ":gaussian_excess", r["gaussian_p"]))
        tests.append(("convergence:" + "+".join(r["axes"]) + ":enrichment", r["p_enrichment"]))
    adj = bh([p for _, p in tests])
    res["multiple_testing"] = {"method": "Benjamini-Hochberg over all reported tests", "n_tests": len(tests),
                               "n_significant_raw_0.05": int(sum(p < 0.05 for _, p in tests)),
                               "n_significant_bh_0.05": int((adj < 0.05).sum()),
                               "tests": [{"test": t, "p": float(p), "q_bh": float(q)} for (t, p), q in zip(tests, adj)]}
    lost = [t for (t, p), q in zip(tests, adj) if p < 0.05 <= q]
    res["multiple_testing"]["significant_raw_but_not_after_bh"] = lost
    print(f"tests {len(tests)}: significant raw {res['multiple_testing']['n_significant_raw_0.05']}, after BH {res['multiple_testing']['n_significant_bh_0.05']}; lost: {lost}", flush=True)


def main():
    res = {}
    convergence(res)
    multiple_testing(res)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2))
    print("written", OUT / "stats.json")


if __name__ == "__main__":
    main()
