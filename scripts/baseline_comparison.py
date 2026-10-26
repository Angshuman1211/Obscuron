#!/usr/bin/env python
"""Compare the extreme-value kNN score with standard baselines under one protocol (TCBB revision).

Held-out-family task (903 Pfam families, 11,527 proteins, ESM-2 layer 22), every method scored
with the family hidden from the reference:
  knn5        mean cosine distance to 5 nearest reference proteins (the score in the paper)
  knn1        distance to the single nearest reference protein (HiFi-NN-style nearest neighbour)
  mahal       Mahalanobis distance to the reference without the family (PCA-128, shrinkage 0.1)
  hmmer       minus the best Pfam 35.0 bit-score margin over gathering threshold, own profile ignored
  phmmer      minus the best phmmer bit score to a reference protein outside the family
Calibration, 20 random 80/20 splits (same seeds as scripts/reviewer_scorer_analysis.py):
  gpd         Generalized Pareto tail p-value (the paper)
  conformal   split-conformal p-value (1 + #{calibration >= d}) / (n + 1) on the same distance
Dark-query task (34,138 queries, 29,862-protein reference): AUROC of each method for
later-characterised against still-dark proteins, with bootstrap intervals.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline_comparison.py
Writes results/baselines/comparison.json.
"""
from __future__ import annotations

import json
import sys
import time
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

PROC = ROOT / "data" / "processed" / "gtdb_R207"
B = ROOT / "results" / "baselines"
B.mkdir(parents=True, exist_ok=True)
ALPHAS = np.array([0.001, 0.01, 0.02, 0.05, 0.10, 0.20])
rng = np.random.default_rng(42)
BIG = 1e6


def auc(y, s):
    r = stats.rankdata(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(y, s, n=500):
    v = [auc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(per_family):
    a = np.array([v for v in per_family if not np.isnan(v)])
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n_families": int(len(a)),
            "frac_below_0.7": float((a < 0.7).mean())}


class Mahalanobis:
    def __init__(self, X, dims=128, shrink=0.1):
        self.mu0 = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu0, full_matrices=False)
        self.P = vt[:dims].T
        self.Z = (X - self.mu0) @ self.P
        self.shrink = shrink

    def score(self, idx_query, mask_ref):
        Zr = self.Z[mask_ref]
        mu = Zr.mean(0)
        S = np.cov(Zr, rowvar=False)
        S = (1 - self.shrink) * S + self.shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        Si = np.linalg.inv(S)
        D = self.Z[idx_query] - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, Si, D))


def heldout(res):
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    ids_all = np.array([str(i) for i in ref["ids"]])
    fam_all = np.array([str(f) for f in ref["families"]])
    keep = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy() >= 5
    ids, fams = ids_all[keep], fam_all[keep]
    X = l2_normalize(ref["L22"].astype(np.float64)[keep])
    families = np.unique(fams)
    timing = {}

    t = time.time(); idx, sim = _topm_neighbors(X, 500); timing["knn_index_s"] = time.time() - t
    mah = Mahalanobis(X)

    pfam_path, ph_path = B / "heldout_pfam35_hits.csv.gz", B / "heldout_phmmer_hits.csv.gz"
    hm = ph = None
    if pfam_path.exists():
        h = pd.read_csv(pfam_path)
        h["margin"] = h.bits - h.ga
        h["fam"] = h.pfam.str.split(".").str[0]
        hm = h.groupby(["protein_id", "fam"]).margin.max().unstack()   # proteins x families
        hm = hm.reindex(ids)
    if ph_path.exists():
        p = pd.read_csv(ph_path)
        fam_of = dict(zip(ids_all, fam_all))
        p["tfam"] = p.target.map(fam_of)
        ph = p
    out = {k: [] for k in ("knn5", "knn1", "mahal", "hmmer", "phmmer")}
    for F in families:
        is_f = fams == F
        y = np.r_[np.ones(is_f.sum()), np.zeros((~is_f).sum())]
        nov = _knn_dist_excluding(idx[is_f], sim[is_f], is_f, 5); kn = _knn_dist_excluding(idx[~is_f], sim[~is_f], is_f, 5)
        out["knn5"].append(auc(y, np.r_[nov, kn]))
        nov = _knn_dist_excluding(idx[is_f], sim[is_f], is_f, 1); kn = _knn_dist_excluding(idx[~is_f], sim[~is_f], is_f, 1)
        out["knn1"].append(auc(y, np.r_[nov, kn]))
        q = np.r_[np.where(is_f)[0], np.where(~is_f)[0]]
        out["mahal"].append(auc(y, mah.score(q, ~is_f)))
        if hm is not None:
            fam_id = F.split(".")[0]
            sub = hm.drop(columns=[fam_id], errors="ignore")
            best = sub.max(axis=1).fillna(-BIG).to_numpy()
            s = -best
            out["hmmer"].append(auc(y, np.r_[s[is_f], s[~is_f]]))
        if ph is not None:
            pp = ph[ph.tfam != F]
            best = pp.groupby("query").bits.max().reindex(ids).fillna(-BIG).to_numpy()
            s = -best
            out["phmmer"].append(auc(y, np.r_[s[is_f], s[~is_f]]))
    res["heldout"] = {k: summarize(v) for k, v in out.items() if v}
    res["heldout_timing"] = timing
    print({k: round(v["mean"], 4) for k, v in res["heldout"].items()}, flush=True)

    # calibration: GPD tail against split conformal, same 20 splits
    gpd, conf, floor = [], [], []
    for rep in range(20):
        r = np.random.default_rng(1000 + rep)
        perm = r.permutation(len(X)); nt = len(X) // 5
        te, ca = perm[:nt], perm[nt:]
        loo = knn_distance(X[ca], X[ca], k=5, exclude_self=True)
        dte = knn_distance(X[te], X[ca], k=5)
        tl = fit_gpd_tail(loo, 0.90)
        pg = tail_pvalue(tl, dte, ref_distances=loo)
        srt = np.sort(loo)
        pc = (1 + (len(srt) - np.searchsorted(srt, dte, side="left"))) / (len(srt) + 1)
        gpd.append((pg[:, None] <= ALPHAS[None, :]).mean(0))
        conf.append((pc[:, None] <= ALPHAS[None, :]).mean(0))
        floor.append(1 / (len(srt) + 1))
    res["calibration"] = {"alpha": ALPHAS.tolist(),
                          "gpd_mean": np.mean(gpd, 0).tolist(), "gpd_sd": np.std(gpd, 0).tolist(),
                          "conformal_mean": np.mean(conf, 0).tolist(), "conformal_sd": np.std(conf, 0).tolist(),
                          "conformal_smallest_attainable_p": float(np.mean(floor))}


def dark(res):
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    cat = samp.set_index("protein_id").category
    r1 = np.load(PROC / "esm2_reembed_L33_22.npz"); q1 = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
    rem = np.load(PROC / "esm2_L22_L33_remaining.npz")
    allids = np.array([str(i) for i in np.r_[r1["ids"], q1["ids"], rem["ids"]]])
    E = np.concatenate([r1["L22"], q1["L22"], rem["L22"]]).astype(np.float64)
    c = cat.reindex(allids).to_numpy()
    isr, isq = c == "characterised_at_t0", np.isin(c, ["positive", "dark_negative"])
    R, Q = l2_normalize(E[isr]), l2_normalize(E[isq])
    qid = allids[isq]; y = (c[isq] == "positive").astype(int)
    scores = {}
    t = time.time(); scores["knn5"] = knn_distance(Q, R, k=5); res.setdefault("dark_timing", {})["knn5_s"] = time.time() - t
    scores["knn1"] = knn_distance(Q, R, k=1)
    mah = Mahalanobis(np.vstack([R, Q]))
    scores["mahal"] = mah.score(np.arange(len(R), len(R) + len(Q)), np.r_[np.ones(len(R), bool), np.zeros(len(Q), bool)])
    aud = ROOT / "results" / "reviewer" / "label_audit_hits.csv"
    if aud.exists():
        m35 = pd.read_csv(aud).set_index("protein_id").m35.reindex(qid).fillna(-BIG).to_numpy()
        scores["hmmer"] = -m35
    phq = B / "darkquery_phmmer_besthit.csv.gz"
    if phq.exists():
        bits = pd.read_csv(phq).groupby("query").bits.max().reindex(qid).fillna(-BIG).to_numpy()
        scores["phmmer"] = -bits
    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv").set_index("protein_id").length_aa
    scores["length"] = lens.reindex(qid).to_numpy(float)
    res["dark_query"] = {"n": int(len(y)), "base_rate": float(y.mean()),
                         **{k: {"auroc": auc(y, v), "ci95": boot(y, v)} for k, v in scores.items()}}
    print({k: round(v["auroc"], 3) for k, v in res["dark_query"].items() if isinstance(v, dict)}, flush=True)


def main():
    res = {}
    heldout(res)
    dark(res)
    (B / "comparison.json").write_text(json.dumps(res, indent=2))
    print("written", B / "comparison.json")


if __name__ == "__main__":
    main()
