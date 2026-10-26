#!/usr/bin/env python
"""Scoring analyses requested by the simulated TCBB review (track-2-research-extension).

1. Lineage-disjoint calibration. The earlier 80/20 splits were random over proteins. Here the
   reference and the test proteins come from different genomes (genome-disjoint) or different
   phyla (phylum-disjoint), 20 repeats each, and flag rates of known test proteins are split by
   how many members of their Pfam family the reference holds (>= 5, 1-4, 0).
2. A density-normalised score, s(x) = d5(x) / mean over x's 5 nearest reference proteins r of
   d5(r), the reference proteins' own leave-one-out distances (a local-outlier-factor style ratio).
   It is calibrated with the same Generalized Pareto procedure and evaluated on every task.
3. More baselines on the withheld-family task: distance to the k-th neighbour (deep kNN), and a
   kernel log-density over the 500 nearest reference proteins (temperature 0.05).
4. Harder withheld-family variants: known proteins restricted to families of the same Pfam clan,
   and a pooled AUROC that weights families by size.
5. Combined pipeline: share of dark queries with no phmmer hit at all, and the withheld-family
   AUROC of a rank average of phmmer and the embedding score.
6. Transfer of the density-normalised score to the 60 independent genomes, by family coverage.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/revision_scoring.py
Writes results/revision/scoring.json.
"""
from __future__ import annotations

import gzip
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
from reembed_eval import _knn_dist_excluding, _topm_neighbors  # noqa: E402


def auroc(y, s):
    """Tie-aware AUROC (average ranks). The sequence-search scores have many ties (no hit), which an
    ordinal-rank AUROC would break by input order."""
    y = np.asarray(y); s = np.asarray(s, float)
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=0.0, posinf=1e300, neginf=-1e300)
    r = stats.rankdata(s); n1 = int((y == 1).sum()); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = np.array([0.01, 0.05, 0.10])
K = 5


def topk(Q, R, k, exclude_self=False, chunk=2048):
    """indices and similarities of the k nearest reference rows for each query row."""
    idx = np.empty((len(Q), k), np.int64); sim = np.empty((len(Q), k))
    for a in range(0, len(Q), chunk):
        S = Q[a:a + chunk] @ R.T
        if exclude_self:
            S[np.arange(len(S)), np.arange(a, a + len(S))] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, 1)
        idx[a:a + chunk] = np.take_along_axis(part, o, 1); sim[a:a + chunk] = np.take_along_axis(ps, o, 1)
    return idx, sim


def scores_against(Q, R, loo_R=None, exclude_self=False):
    """mean-of-5 distance and, if reference LOO distances are given, the density-normalised ratio."""
    idx, sim = topk(Q, R, K, exclude_self)
    d = (1 - sim).mean(1)
    if loo_R is None:
        return d, None
    return d, d / np.clip(loo_R[idx].mean(1), 1e-9, None)


def fit_and_flag(cal_scores, test_scores):
    t = fit_gpd_tail(cal_scores, 0.90)
    p = tail_pvalue(t, test_scores, ref_distances=cal_scores)
    return p


def lineage_calibration(X, fams, genomes, phyla, res):
    out = {}
    for mode, groups in (("random_protein", None), ("genome_disjoint", genomes), ("phylum_disjoint", phyla)):
        rows = []
        for rep in range(20):
            r = np.random.default_rng(3000 + rep)
            if groups is None:
                test = r.random(len(X)) < 0.2
            else:
                ug = np.unique(groups)
                tg = set(r.choice(ug, size=max(1, len(ug) // 5), replace=False))
                test = np.isin(groups, list(tg))
            ref, te = ~test, test
            R = X[ref]
            ref_idx = np.where(ref)[0]
            loo = knn_distance(R, R, k=K, exclude_self=True)
            d_te, n_te = scores_against(X[te], R, loo)
            # LOO density ratio of the reference itself
            ridx, rsim = topk(R, R, K, exclude_self=True)
            n_ref = loo / np.clip(loo[ridx].mean(1), 1e-9, None)
            p_d = fit_and_flag(loo, d_te)
            p_n = fit_and_flag(n_ref, n_te)
            fam_count = pd.Series(fams[ref]).value_counts()
            cover = pd.Series(fams[te]).map(fam_count).fillna(0).to_numpy()
            strata = {"all": np.ones(te.sum(), bool), "ge5": cover >= 5, "1to4": (cover >= 1) & (cover < 5), "0": cover == 0}
            row = {}
            for sname, m in strata.items():
                if m.sum():
                    row[sname] = {"n": int(m.sum()),
                                  "distance": [(p_d[m] <= a).mean() for a in ALPHAS],
                                  "normalised": [(p_n[m] <= a).mean() for a in ALPHAS]}
            rows.append(row)
        summ = {}
        for s in ("all", "ge5", "1to4", "0"):
            got = [r_[s] for r_ in rows if s in r_]
            if got:
                summ[s] = {"mean_n": float(np.mean([g["n"] for g in got])),
                           "distance": np.mean([g["distance"] for g in got], 0).tolist(),
                           "normalised": np.mean([g["normalised"] for g in got], 0).tolist()}
        out[mode] = summ
        print(mode, {s: [round(x, 3) for x in v["distance"]] + ["|"] + [round(x, 3) for x in v["normalised"]] for s, v in summ.items()}, flush=True)
    res["lineage_calibration"] = {"alpha": ALPHAS.tolist(), **out}


def heldout_variants(X, fams, clan_of, res):
    idx, sim = _topm_neighbors(X, 500)
    loo = 1 - sim[:, :K].mean(1)
    fam_arr = fams
    u = np.unique(fam_arr)
    out = {k: [] for k in ("knn5", "knn1", "kth5", "kde", "normalised", "knn5_hard", "normalised_hard")}
    sizes, pooled_y, pooled_s = [], [], []
    for F in u:
        is_f = fam_arr == F
        y = np.r_[np.ones(is_f.sum()), np.zeros((~is_f).sum())]
        order = np.r_[np.where(is_f)[0], np.where(~is_f)[0]]
        bad = is_f[idx.clip(min=0)] | (idx < 0)
        s_m = np.where(bad, -np.inf, sim)
        o = np.argsort(-s_m, axis=1)
        s_sorted = np.take_along_axis(s_m, o, 1); i_sorted = np.take_along_axis(idx, o, 1)
        d5 = 1 - s_sorted[:, :K].mean(1)
        d1 = 1 - s_sorted[:, 0]
        dk = 1 - s_sorted[:, K - 1]
        with np.errstate(over="ignore"):
            kde = -np.log(np.exp(np.where(np.isfinite(s_sorted), s_sorted, -np.inf) / 0.05).sum(1) + 1e-300)
        # density ratio: neighbours' own LOO distances (computed on the full reference; the hidden family
        # is excluded from the neighbour set, which is what matters for the ratio's numerator)
        nr = d5 / np.clip(loo[i_sorted[:, :K].clip(min=0)].mean(1), 1e-9, None)
        for name, s in (("knn5", d5), ("knn1", d1), ("kth5", dk), ("kde", kde), ("normalised", nr)):
            out[name].append(auroc(y, s[order]))
        c = clan_of.get(F.split(".")[0])
        if c:
            same = np.array([clan_of.get(f.split(".")[0]) == c for f in fam_arr]) & ~is_f
            if same.sum() >= 5:
                o2 = np.r_[np.where(is_f)[0], np.where(same)[0]]
                y2 = np.r_[np.ones(is_f.sum()), np.zeros(same.sum())]
                out["knn5_hard"].append(auroc(y2, d5[o2])); out["normalised_hard"].append(auroc(y2, nr[o2]))
        sizes.append(is_f.sum()); pooled_y.append(y); pooled_s.append(d5[order])
    res["heldout_variants"] = {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "n_families": int(np.sum(~np.isnan(v)))}
                               for k, v in out.items() if v}
    w = np.array(sizes, float)
    res["heldout_variants"]["knn5_size_weighted_mean"] = float(np.average(out["knn5"], weights=w))
    print({k: (round(v["mean"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in res["heldout_variants"].items()}, flush=True)
    return idx, sim, loo


def combined(res):
    ph = pd.read_csv(ROOT / "results" / "baselines" / "darkquery_phmmer_besthit.csv.gz")
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    q = samp[samp.category.isin(["positive", "dark_negative"])]
    hit = q.protein_id.isin(set(ph["query"]))
    strong = q.protein_id.isin(set(ph.loc[ph.bits >= 30, "query"]))
    res["coverage"] = {"n_dark_queries": int(len(q)), "share_with_any_phmmer_hit_E10": float(hit.mean()),
                       "share_with_phmmer_hit_ge30_bits": float(strong.mean()),
                       "share_with_no_phmmer_hit": float(1 - hit.mean())}
    print("coverage", res["coverage"], flush=True)


def heldout_combined(X, fams, ids, res):
    ph = pd.read_csv(ROOT / "results" / "baselines" / "heldout_phmmer_hits.csv.gz")
    fam_of = dict(zip(ids, fams)); ph["tfam"] = ph.target.map(fam_of)
    idx, sim = _topm_neighbors(X, 500)
    a_c, a_e, a_p = [], [], []
    for F in np.unique(fams):
        is_f = fams == F
        bad = is_f[idx.clip(min=0)] | (idx < 0)
        d5 = 1 - np.sort(np.where(bad, -np.inf, sim), axis=1)[:, ::-1][:, :K].mean(1)
        best = ph[ph.tfam != F].groupby("query").bits.max().reindex(ids).fillna(-1e6).to_numpy()
        p_nov = -best
        order = np.r_[np.where(is_f)[0], np.where(~is_f)[0]]
        y = np.r_[np.ones(is_f.sum()), np.zeros((~is_f).sum())]
        rc = (stats.rankdata(d5) + stats.rankdata(p_nov)) / 2
        a_c.append(auroc(y, rc[order])); a_e.append(auroc(y, d5[order])); a_p.append(auroc(y, p_nov[order]))
    res["heldout_combined"] = {"embedding": float(np.mean(a_e)), "phmmer": float(np.mean(a_p)), "rank_average": float(np.mean(a_c)),
                               "rank_average_median": float(np.median(a_c))}
    print("combined", res["heldout_combined"], flush=True)


def independent_transfer(res):
    ind = ROOT / "data" / "processed" / "independent_R232"
    z = np.load(ind / "esm2_L22_L33.npz")
    lab = pd.read_csv(ind / "protein_labels.csv").set_index("protein_id")
    ids = [str(i) for i in z["ids"]]
    fam = lab.pfam35_families.reindex(ids).fillna("")
    single = fam.map(lambda s: len([x for x in s.split(";") if x]) == 1).to_numpy()
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    rfam = pd.Series([str(f).split(".")[0] for f in ref["families"]])
    R = l2_normalize(ref["L22"].astype(np.float64))
    loo = knn_distance(R, R, k=K, exclude_self=True)
    ridx, _ = topk(R, R, K, exclude_self=True)
    n_ref = loo / np.clip(loo[ridx].mean(1), 1e-9, None)
    Q = l2_normalize(z["L22"].astype(np.float64)[single])
    d, nr = scores_against(Q, R, loo)
    pd_, pn = fit_and_flag(loo, d), fit_and_flag(n_ref, nr)
    cover = fam[single].map(rfam.value_counts()).fillna(0).to_numpy()
    strata = {"all": np.ones(len(d), bool), "ge5": cover >= 5, "1to4": (cover >= 1) & (cover < 5), "0": cover == 0}
    res["independent_transfer"] = {"alpha": ALPHAS.tolist(), **{
        s: {"n": int(m.sum()), "distance": [float((pd_[m] <= a).mean()) for a in ALPHAS],
            "normalised": [float((pn[m] <= a).mean()) for a in ALPHAS]} for s, m in strata.items()}}
    print("independent", {s: v for s, v in res["independent_transfer"].items() if s != "alpha"}, flush=True)


def main():
    res = {}
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    ids_all = np.array([str(i) for i in ref["ids"]]); fam_all = np.array([str(f) for f in ref["families"]])
    keep = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy() >= 5
    ids, fams = ids_all[keep], fam_all[keep]
    X = l2_normalize(ref["L22"].astype(np.float64)[keep])
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    genomes = samp.genome_accession.reindex(ids).to_numpy()
    phylum = pd.read_csv(PROC / "genome_panel.csv").set_index("accession").phylum
    phyla = pd.Series(genomes).map(phylum).fillna("unknown").to_numpy()
    clan_of = {}
    with gzip.open(ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.clans.tsv.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[1]:
                clan_of[p[0]] = p[1]
    heldout_variants(X, fams, clan_of, res)
    heldout_combined(X, fams, list(ids), res)
    combined(res)
    lineage_calibration(X, fams, genomes, phyla, res)
    independent_transfer(res)
    (OUT / "scoring.json").write_text(json.dumps(res, indent=2, default=float))
    print("written", OUT / "scoring.json")


if __name__ == "__main__":
    main()
