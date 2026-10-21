#!/usr/bin/env python
"""Statistical tests for the corroborating layers (Review II, comment 6).

The report showed point estimates for Layers 2-5 next to the primary scorer but no
uncertainty and no test against it. This script adds, on the real dark queries
(positives = dark at T0 and later assigned a new Pfam-37 family; negatives = still dark):

  1. AUROC of every per-protein signal, with a bootstrap CI and a Mann-Whitney p-value
     against 0.5, and the paired bootstrap difference against the primary ESM-2 score
     on the same proteins.
  2. Layer 3 (immune): Fisher exact test on the flag, and a Cochran-Mantel-Haenszel test
     that stratifies by deciles of the Layer-1 score, so the test asks whether the flag
     says anything beyond Layer 1.
  3. Layer 2 (structure) against ESM-2 on the held-out-family task: paired Wilcoxon
     signed-rank test over the families both were scored on.
  4. Layer 4 (convergence): the observed size of each intersection of top-decile sets
     against (a) the independence expectation and (b) a Gaussian-copula null that keeps
     the measured rank correlations between axes, and the positive rate of each
     intersection against the base rate (hypergeometric test).

Axis orientation is fixed in advance (higher = more novel); nothing here is tuned.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reviewer_corroboration_stats.py
Writes results/reviewer/corroboration_stats.json.
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
sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.convergence import genomic_context_novelty  # noqa: E402
from darkmatter.scoring.knn import l2_normalize  # noqa: E402
from reembed_eval import _knn_dist_excluding, _topm_neighbors  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
B = 1000


def auc(y, s):
    r = stats.rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, s):
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        v.append(auc(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_diff(y, s1, s2):
    d = auc(y, s1) - auc(y, s2)
    v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        v.append(auc(y[i], s1[i]) - auc(y[i], s2[i]))
    v = np.array(v)
    p = float(2 * min((v <= 0).mean(), (v >= 0).mean()))
    return {"diff": float(d), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p_boot": p}


def summarize(name, y, s, ref=None):
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "auroc": auc(y, s), "ci95": boot_ci(y, s),
           "mannwhitney_p": float(stats.mannwhitneyu(s[y == 1], s[y == 0]).pvalue)}
    if ref is not None:
        out["vs_primary"] = paired_diff(y, s, ref)
    print(name, round(out["auroc"], 4), out["ci95"], flush=True)
    return out


def cmh(flag, y, strata):
    """Cochran-Mantel-Haenszel chi-square (1 d.f.) and common odds ratio over strata."""
    num = var = 0.0; rn = rd = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = ((flag == 1) & (y == 1) & m).sum(); b = ((flag == 1) & (y == 0) & m).sum()
        c = ((flag == 0) & (y == 1) & m).sum(); d = ((flag == 0) & (y == 0) & m).sum()
        n = a + b + c + d
        if n < 2:
            continue
        ea = (a + b) * (a + c) / n
        va = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        num += a - ea; var += va
        rn += a * d / n; rd += b * c / n
    chi = (abs(num) - 0.5) ** 2 / var
    return {"chi2": float(chi), "p": float(stats.chi2.sf(chi, 1)), "common_odds_ratio": float(rn / rd)}


def main() -> None:
    res: dict = {}
    es = pd.read_csv(PROC / "esm2_novelty_scores.csv")
    d = es[["protein_id", "raw_knn_distance", "is_positive"]].rename(columns={"raw_knn_distance": "esm2_L33"})
    d["y"] = d.is_positive.astype(str).eq("True").astype(int)
    for f, c, n in (("composition_novelty.csv", "composition_novelty", "composition"),
                    ("prostt5_query_novelty.csv", "structural_novelty", "prostt5_structure"),
                    ("coding_structure.csv", "cpbb", "cpbb"), ("coding_structure.csv", "aa_entropy_k3", "aa_entropy_k3")):
        x = pd.read_csv(PROC / f)[["protein_id", c]].rename(columns={c: n})
        d = d.merge(x, on="protein_id", how="left")
    g = pd.read_csv(ROOT / "results" / "genos-m_novelty_scores.csv").rename(columns={"genos_m_novelty": "genos_m"})
    d = d.merge(g, on="protein_id", how="left")
    lab = pd.read_csv(PROC / "panel_protein_labels.csv", usecols=["protein_id", "dark_at_t0"])
    ctx = genomic_context_novelty(lab, 5, 4)
    d["context"] = d.protein_id.map(ctx)

    # ---- 1a. 34,138-query set, primary = layer-33 ESM-2 (the Layer-4 axis) ----
    S: dict = {"set": "34,138 dark queries; primary scorer = ESM-2 last layer (L33)"}
    y_all = d.y.to_numpy()
    S["esm2_L33"] = summarize("esm2_L33", y_all, d.esm2_L33.to_numpy())
    for n in ("composition", "prostt5_structure", "genos_m", "context", "cpbb", "aa_entropy_k3"):
        m = d[n].notna().to_numpy()
        S[n] = summarize(n, y_all[m], d[n].to_numpy()[m], ref=d.esm2_L33.to_numpy()[m])
        S[n]["note"] = "orientation: higher value scored as more novel (cpbb and aa_entropy_k3: higher = more coding-like)"
    res["per_signal_34138"] = S

    # ---- 1b. 14,138-query set, primary = layer-22 ESM-2 ----
    im = pd.read_csv(ROOT / "results" / "immune_dark_query_flags.csv")
    im["y"] = im.is_positive.astype(int)
    T: dict = {"set": "14,138 dark queries; primary scorer = ESM-2 layer 22 (headline layer)"}
    y = im.y.to_numpy()
    T["esm2_L22_novelty"] = summarize("esm2_L22", y, im.layer1_novelty.to_numpy())
    T["immune_nonself_score"] = summarize("immune_score", y, im.layer3_nonself_score.to_numpy(),
                                          ref=im.layer1_novelty.to_numpy())
    # layer-22 raw distance, to avoid the tied novelty = 1 values
    qry = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    from darkmatter.scoring.knn import knn_distance
    d22 = knn_distance(l2_normalize(qry["L22"].astype(np.float64)), l2_normalize(ref["L22"].astype(np.float64)), k=5)
    T["esm2_L22_raw_distance"] = summarize("esm2_L22_raw", qry["is_positive"].astype(int), d22)
    pid22 = [str(i) for i in qry["ids"]]
    sub = d.set_index("protein_id").loc[pid22]
    for n in ("composition", "prostt5_structure", "cpbb"):
        m = sub[n].notna().to_numpy()
        T[n] = summarize(n + "_14k", sub.y.to_numpy()[m], sub[n].to_numpy()[m], ref=d22[m])
    res["per_signal_14138"] = T

    # ---- 2. Layer 3 flag ----
    flag = im.layer3_flag.astype(int).to_numpy()
    tab = np.array([[((flag == 1) & (y == 1)).sum(), ((flag == 0) & (y == 1)).sum()],
                    [((flag == 1) & (y == 0)).sum(), ((flag == 0) & (y == 0)).sum()]])
    orr, p = stats.fisher_exact(tab)
    decile = pd.qcut(stats.rankdata(im.layer1_novelty.to_numpy(), method="ordinal"), 10, labels=False)
    res["immune_flag"] = {
        "table_rows_positive_then_still_dark_cols_flagged_then_not": tab.tolist(),
        "flag_rate_positive": float(tab[0, 0] / tab[0].sum()), "flag_rate_still_dark": float(tab[1, 0] / tab[1].sum()),
        "fisher_odds_ratio": float(orr), "fisher_p": float(p),
        "cmh_stratified_by_layer1_decile": cmh(flag, y, decile),
        "spearman_score_vs_layer1": float(stats.spearmanr(im.layer3_nonself_score, im.layer1_novelty)[0]),
    }

    # ---- 3. ProstT5 vs ESM-2 on held-out families (same proteins) ----
    pr = np.load(PROC / "prostt5_reference.npz")
    ids_p = np.array([str(i) for i in pr["ids"]]); fam_p = np.array([str(i) for i in pr["families"]])
    ix = {s: i for i, s in enumerate(np.array([str(i) for i in ref["ids"]]))}
    rows = np.array([ix[s] for s in ids_p])
    sizes = pd.Series(fam_p).groupby(fam_p).transform("size").to_numpy()
    k5 = sizes >= 5

    def fam_aucs(v, center):
        v = v[k5].astype(np.float64)
        if center:
            v = v - v.mean(0, keepdims=True)
        vn = l2_normalize(v); f = fam_p[k5]
        idx, sim = _topm_neighbors(vn, 500)
        out = {}
        for fam in np.unique(f):
            m = f == fam
            nov = _knn_dist_excluding(idx[m], sim[m], m, 5); kn = _knn_dist_excluding(idx[~m], sim[~m], m, 5)
            out[fam] = auc(np.r_[np.ones(len(nov)), np.zeros(len(kn))].astype(int), np.r_[nov, kn])
        return pd.Series(out)

    a_pr = fam_aucs(pr["emb"], True)
    a_es = fam_aucs(ref["L22"][rows], False)
    both = a_pr.index.intersection(a_es.index)
    w = stats.wilcoxon(a_es[both], a_pr[both])
    res["prostt5_vs_esm2_heldout"] = {
        "n_families": int(len(both)), "n_proteins": int(k5.sum()),
        "prostt5_centered_mean": float(a_pr[both].mean()), "esm2_L22_mean": float(a_es[both].mean()),
        "mean_difference_esm2_minus_prostt5": float((a_es[both] - a_pr[both]).mean()),
        "wilcoxon_p": float(w.pvalue), "frac_families_esm2_better": float((a_es[both] > a_pr[both]).mean())}
    print("prostt5 vs esm2", res["prostt5_vs_esm2_heldout"], flush=True)

    # ---- 4. convergence ----
    axes = {"evt": "esm2_L33", "context": "context", "composition": "composition",
            "genos-m": "genos_m", "prostt5": "prostt5_structure"}
    m = d[list(axes.values())].notna().all(axis=1).to_numpy()
    X = d.loc[m, list(axes.values())].to_numpy(); yy = y_all[m]; N = len(yy)
    # same rule as run_convergence: an axis is "high" if its value is >= its own 0.90 quantile
    top = np.column_stack([X[:, j] >= np.quantile(X[:, j], 0.90) for j in range(X.shape[1])])
    rho = stats.spearmanr(X).statistic
    share = top.mean(axis=0)  # ties can make an axis's "top decile" larger than 10%; the null must match
    rho_g = 2 * np.sin(np.pi * rho / 6)  # Spearman -> Pearson correlation of the Gaussian copula
    names = list(axes)
    base = yy.mean()
    L = np.linalg.cholesky(rho_g + 1e-9 * np.eye(len(rho_g)))
    sims = 400
    # sample the null once per sim for all axes, then count each intersection
    combos = [c for r_ in (2, 3, 4, 5) for c in combinations(range(len(names)), r_)]
    null = {c: [] for c in combos}
    thr = stats.norm.ppf(1 - share)
    for _ in range(sims):
        Z = rng.standard_normal((N, len(names))) @ L.T
        t = Z >= thr
        for c in combos:
            null[c].append(int(t[:, list(c)].all(axis=1).sum()))
    conv = []
    for c in combos:
        obs_mask = top[:, list(c)].all(axis=1); n = int(obs_mask.sum())
        k = int(yy[obs_mask].sum())
        nl = np.array(null[c])
        entry = {"axes": [names[i] for i in c], "observed_n": n,
                 "expected_n_independent": float(N * np.prod(share[list(c)])),
                 "copula_null_mean": float(nl.mean()), "copula_null_p95": float(np.percentile(nl, 95)),
                 "copula_null_p_ge_obs": float(((nl >= n).sum() + 1) / (sims + 1)),
                 "positives": k, "positive_rate": float(k / n) if n else None,
                 "lift": float(k / n / base) if n else None}
        if n:
            entry["hypergeom_p_enrichment"] = float(stats.hypergeom.sf(k - 1, N, int(yy.sum()), n))
            entry["hypergeom_p_depletion"] = float(stats.hypergeom.cdf(k, N, int(yy.sum()), n))
        conv.append(entry)
    res["convergence"] = {"n_queries_all_signals": int(N), "base_rate": float(base), "spearman": rho.tolist(),
                          "axes": names, "top_set_share_per_axis": share.tolist(), "n_copula_sims": sims, "intersections": conv}
    (OUT / "corroboration_stats.json").write_text(json.dumps(res, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
