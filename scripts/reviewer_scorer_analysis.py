#!/usr/bin/env python
"""Scorer-level checks asked for in the Review II feedback (comments 3 and 5).

Part A (comment 3): what does AUROC 0.962 mean for ranking real dark queries?
  The 0.962 is a held-out-family AUROC (a Pfam family is hidden from the reference,
  then its members are scored against ordinary known proteins). Precision@K is a
  different task: among dark proteins, rank the ones that later gain a Pfam-37 family
  above the ones that stay dark. This part scores BOTH tasks with the same layer-22 and
  layer-33 embeddings and the same 14,138 queries, adds the AUROC of the query task with
  a bootstrap CI, a decile table, a length-only baseline, and the positive predictive
  value implied by the held-out-family operating points at several prevalences.

Part B (comment 5): the tail model.
  Sensitivity of the GPD fit to the threshold quantile and to k, a parametric-bootstrap
  goodness-of-fit test (the plain KS p-value is optimistic because the parameters are
  fitted on the same data), mean residual life stability, and the calibration of p-values
  against observed false-flag rates with family-clustered confidence intervals over 20
  random calibration/test splits.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reviewer_scorer_analysis.py
Writes results/reviewer/scorer_analysis.json.
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
sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.scoring.gpd import fit_gpd_tail, tail_pvalue  # noqa: E402
from darkmatter.scoring.knn import knn_distance, l2_normalize  # noqa: E402
from reembed_eval import _knn_dist_excluding, _topm_neighbors, auroc  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
K = 5
ALPHAS = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
rng = np.random.default_rng(42)


def boot_auroc(y, s, n=1000):
    pos, neg = s[y == 1], s[y == 0]
    vals = []
    for _ in range(n):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        vals.append(auroc(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def main() -> None:
    res: dict = {}
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    qry = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
    fam_all = np.array([str(x) for x in ref["families"]])
    y = qry["is_positive"].astype(int)
    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv").set_index("protein_id")["length_aa"]
    length = lens.loc[[str(i) for i in qry["ids"]]].to_numpy(dtype=float)

    # ---------------- Part A ----------------
    A: dict = {"n_queries": int(len(y)), "n_positive": int(y.sum()), "base_rate": float(y.mean())}
    dist = {}
    for layer in (22, 33):
        R = l2_normalize(ref[f"L{layer}"].astype(np.float64))
        Q = l2_normalize(qry[f"L{layer}"].astype(np.float64))
        d = knn_distance(Q, R, k=K)
        dist[layer] = d
        loo = knn_distance(R, R, k=K, exclude_self=True)
        order = np.argsort(-d, kind="stable")
        a = auroc(y, d)
        info = {
            "query_task_auroc": float(a), "query_task_auroc_ci95": boot_auroc(y, d),
            "median_query_distance": float(np.median(d)),
            "median_reference_loo_distance": float(np.median(loo)),
            "max_reference_loo_distance": float(loo.max()),
            "frac_queries_beyond_reference_max": float((d > loo.max()).mean()),
            "frac_queries_beyond_reference_p99": float((d > np.quantile(loo, 0.99)).mean()),
        }
        rows = []
        for kk in (50, 100, 500, 1000, 2000, 5000):
            p = float(y[order[:kk]].mean())
            rows.append({"K": kk, "precision": p, "lift": p / y.mean()})
        info["precision_at_k"] = rows
        dec = []
        ranks = np.empty(len(d), dtype=int); ranks[order] = np.arange(len(d))
        for i in range(10):
            sel = (ranks >= i * len(d) // 10) & (ranks < (i + 1) * len(d) // 10)
            dec.append({"decile_from_most_novel": i + 1, "n": int(sel.sum()),
                        "positive_rate": float(y[sel].mean()),
                        "mean_distance": float(d[sel].mean())})
        info["deciles"] = dec
        A[f"layer{layer}"] = info
        print(f"layer {layer}: query-task AUROC {a:.3f}", flush=True)

    A["length_only_auroc"] = float(auroc(y, length))
    A["length_only_auroc_ci95"] = boot_auroc(y, length)
    A["spearman_distance_vs_length_L22"] = float(stats.spearmanr(dist[22], length)[0])
    A["median_length_positive"] = float(np.median(length[y == 1]))
    A["median_length_still_dark"] = float(np.median(length[y == 0]))
    A["spearman_L22_vs_L33_distance"] = float(stats.spearmanr(dist[22], dist[33])[0])

    # held-out-family operating points (layer 22), then PPV at several prevalences
    fams_all = fam_all
    keep = pd.Series(fams_all).groupby(fams_all).transform("size").to_numpy() >= 5
    fams = fams_all[keep]
    X = l2_normalize(ref["L22"].astype(np.float64)[keep])
    perm = rng.permutation(len(X)); n_test = len(X) // 5
    test, cal = X[perm[:n_test]], X[perm[n_test:]]
    loo = knn_distance(cal, cal, k=K, exclude_self=True)
    tail = fit_gpd_tail(loo, 0.90)
    p_known = tail_pvalue(tail, knn_distance(test, cal, k=K), ref_distances=loo)
    idx, sim = _topm_neighbors(X, 500)
    d_novel = np.concatenate([_knn_dist_excluding(idx[fams == f], sim[fams == f], fams == f, K)
                              for f in np.unique(fams)])
    p_novel = tail_pvalue(tail, d_novel, ref_distances=loo)
    ppv = []
    for a_ in ALPHAS:
        tpr, fpr = float((p_novel <= a_).mean()), float((p_known <= a_).mean())
        row = {"alpha": float(a_), "tpr": tpr, "fpr": fpr}
        for prev in (0.0139, 0.121, 0.293):
            row[f"ppv_at_prevalence_{prev}"] = float(prev * tpr / (prev * tpr + (1 - prev) * fpr))
        ppv.append(row)
    A["heldout_operating_points_ppv"] = ppv
    # where do real dark queries sit relative to the held-out-family task?
    A["heldout_family_novel_distance_median"] = float(np.median(d_novel))
    A["heldout_family_novel_distance_p95"] = float(np.quantile(d_novel, 0.95))
    A["dark_query_distance_p05"] = float(np.quantile(dist[22], 0.05))
    A["frac_dark_queries_with_pvalue_below_1e-3"] = float(
        (tail_pvalue(tail, dist[22], ref_distances=loo) <= 1e-3).mean())
    res["A_query_task_vs_heldout_task"] = A

    # ---------------- Part B ----------------
    B: dict = {}
    X = l2_normalize(ref["L22"].astype(np.float64)[keep])
    loo_all = knn_distance(X, X, k=K, exclude_self=True)

    # B1: threshold sensitivity
    rows = []
    for q in (0.80, 0.85, 0.90, 0.95):
        t = fit_gpd_tail(loo_all, q)
        ex = loo_all[loo_all > t.u] - t.u
        ks = stats.kstest(ex, "genpareto", args=(t.xi, 0, t.beta))
        rows.append({"threshold_quantile": q, "u": t.u, "n_exceedances": t.n_exceedances,
                     "xi": t.xi, "beta": t.beta, "ks_pvalue_naive": float(ks.pvalue)})
    B["threshold_sensitivity"] = rows

    # B2: parametric-bootstrap goodness of fit at the frozen 0.90 threshold
    t = fit_gpd_tail(loo_all, 0.90)
    ex = np.sort(loo_all[loo_all > t.u] - t.u)
    n = len(ex)

    def ks_ad(sample, xi, beta):
        s = np.sort(sample)
        F = np.clip(stats.genpareto.cdf(s, xi, 0, beta), 1e-12, 1 - 1e-12)
        i = np.arange(1, len(s) + 1)
        ks_stat = max((i / len(s) - F).max(), (F - (i - 1) / len(s)).max())
        ad = -len(s) - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / len(s)
        return ks_stat, ad

    ks0, ad0 = ks_ad(ex, t.xi, t.beta)
    ks_b, ad_b = [], []
    for _ in range(500):
        sim_ex = stats.genpareto.rvs(t.xi, 0, t.beta, size=n, random_state=rng.integers(1 << 31))
        xi_b, _, beta_b = stats.genpareto.fit(sim_ex, floc=0)
        a_, b_ = ks_ad(sim_ex, xi_b, beta_b)
        ks_b.append(a_); ad_b.append(b_)
    B["gof_parametric_bootstrap"] = {
        "n_exceedances": n, "xi": t.xi, "beta": t.beta,
        "ks_stat": float(ks0), "ks_pvalue_bootstrap": float((np.array(ks_b) >= ks0).mean()),
        "ad_stat": float(ad0), "ad_pvalue_bootstrap": float((np.array(ad_b) >= ad0).mean()),
        "n_bootstrap": 500,
    }
    # xi uncertainty: nonparametric bootstrap of the exceedances
    xis = [stats.genpareto.fit(rng.choice(ex, n), floc=0)[0] for _ in range(300)]
    B["xi_bootstrap_ci95"] = [float(np.percentile(xis, 2.5)), float(np.percentile(xis, 97.5))]

    # B3: mean residual life (linear if GPD holds above u)
    mrl = []
    for q in np.linspace(0.80, 0.98, 10):
        u_ = np.quantile(loo_all, q); e = loo_all[loo_all > u_] - u_
        mrl.append({"quantile": float(q), "u": float(u_), "mean_excess": float(e.mean()), "n": int(len(e))})
    B["mean_residual_life"] = mrl

    # B4: k sensitivity of the held-out-family AUROC and GPD shape (k was frozen at 5 before results)
    Xfam = X
    idx, sim = _topm_neighbors(Xfam, 500)
    ksens = []
    for kk in (1, 5, 10, 20):
        aucs = []
        for f in np.unique(fams):
            is_f = fams == f
            novel = _knn_dist_excluding(idx[is_f], sim[is_f], is_f, kk)
            known = _knn_dist_excluding(idx[~is_f], sim[~is_f], is_f, kk)
            aucs.append(auroc(np.r_[np.ones(len(novel)), np.zeros(len(known))], np.r_[novel, known]))
        aucs = np.array([a for a in aucs if not np.isnan(a)])
        tk = fit_gpd_tail(knn_distance(Xfam, Xfam, k=kk, exclude_self=True), 0.90)
        ksens.append({"k": kk, "heldout_auroc_mean": float(aucs.mean()),
                      "heldout_auroc_median": float(np.median(aucs)), "xi": tk.xi, "beta": tk.beta})
    B["k_sensitivity"] = ksens

    # B5: p-value calibration over 20 random calibration/test splits, family-clustered CIs
    fam_codes = pd.factorize(fams)[0]
    all_ff = []
    for rep in range(20):
        r = np.random.default_rng(1000 + rep)
        perm = r.permutation(len(Xfam)); nt = len(Xfam) // 5
        te, ca = perm[:nt], perm[nt:]
        loo_c = knn_distance(Xfam[ca], Xfam[ca], k=K, exclude_self=True)
        tl = fit_gpd_tail(loo_c, 0.90)
        pk = tail_pvalue(tl, knn_distance(Xfam[te], Xfam[ca], k=K), ref_distances=loo_c)
        all_ff.append(((pk[:, None] <= ALPHAS[None, :]).mean(0), pk, fam_codes[te]))
    mean_ff = np.mean([a[0] for a in all_ff], axis=0)
    sd_ff = np.std([a[0] for a in all_ff], axis=0)
    pk0, fc0 = all_ff[0][1], all_ff[0][2]
    ufam = np.unique(fc0)
    boots = []
    for _ in range(500):
        pick = r_ = rng.choice(ufam, len(ufam))
        sel = np.concatenate([np.where(fc0 == f)[0] for f in pick])
        boots.append((pk0[sel][:, None] <= ALPHAS[None, :]).mean(0))
    boots = np.array(boots)
    B["calibration_20_splits"] = {
        "alpha": ALPHAS.tolist(), "mean_false_flag": mean_ff.tolist(), "sd_across_splits": sd_ff.tolist(),
        "family_clustered_ci95_split0": [[float(np.percentile(boots[:, i], 2.5)),
                                          float(np.percentile(boots[:, i], 97.5))] for i in range(len(ALPHAS))],
        "naive_wilson_ci95_split0": [wilson(int(round(((pk0 <= a_).mean()) * len(pk0))), len(pk0)) for a_ in ALPHAS],
        "max_abs_gap": float(np.abs(mean_ff - ALPHAS).max()),
    }
    # uniformity of p-values among tail points of known proteins: p<=0.2 scaled to [0,1]
    tail_p = np.concatenate([a[1][a[1] <= 0.2] for a in all_ff]) / 0.2
    B["tail_pvalue_uniformity_ks"] = {"n": int(len(tail_p)), "ks_stat": float(stats.kstest(tail_p, "uniform").statistic)}
    res["B_tail_model"] = B

    (OUT / "scorer_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"A": {k: v for k, v in A.items() if k in ("length_only_auroc", "spearman_L22_vs_L33_distance")},
                      "B_gof": B["gof_parametric_bootstrap"], "B_cal": B["calibration_20_splits"]["mean_false_flag"]},
                     indent=1))


if __name__ == "__main__":
    main()
