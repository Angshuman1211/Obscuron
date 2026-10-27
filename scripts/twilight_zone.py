#!/usr/bin/env python
"""Where does the embedding score beat sequence search? (TCBB revision)

Withheld-family task (903 Pfam families with >= 5 members, ESM-2 layer 22), as in
scripts/revision_scoring.py: for each family F the family is hidden from the reference and its
members are the positives (novel), all other proteins the negatives (known). Every protein also
has a best phmmer bit score to the reference outside F (results/baselines/heldout_phmmer_hits.csv.gz,
self-hits excluded), which is the sequence-search score. The two hard cases for sequence search:

  * hard positives: members of the withheld family whose best hit outside it is strong
    (>= 30 bits), for example to a sister family; search calls them known;
  * hard negatives: known proteins whose best hit to the rest of the reference is weak
    (< 30 bits or none), remote members of their family; search calls them novel.

For each method (kNN-5 distance, nearest-neighbour distance, phmmer, rank average of phmmer and
kNN-5) we report per-family AUROC on all proteins and on these subsets, and, for positives binned
by their best-hit score, the share ranked above the 95th percentile of the negatives (detection at
5% false flags; ties count half).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/twilight_zone.py
Writes results/revision/twilight_zone.json.
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
from darkmatter.scoring.knn import l2_normalize  # noqa: E402
from reembed_eval import _topm_neighbors  # noqa: E402
from revision_scoring import PROC, auroc  # noqa: E402

STRONG = 30.0  # bits
BINS = [(-np.inf, 1e-9, "no hit"), (1e-9, 30, "< 30 bits"), (30, 50, "30-50 bits"), (50, 100, "50-100 bits"),
        (100, np.inf, ">= 100 bits")]
METHODS = ("knn5", "knn1", "phmmer", "rank_average")


def knn_excluding(idx, sim, is_f, k):
    bad = is_f[idx.clip(min=0)] | (idx < 0)
    return 1 - np.sort(np.where(bad, -np.inf, sim), axis=1)[:, ::-1][:, :k].mean(1)


def percentile_vs(neg, s):
    """share of negatives each score exceeds, ties counted half"""
    neg = np.sort(neg)
    return (np.searchsorted(neg, s, "left") + np.searchsorted(neg, s, "right")) / 2 / len(neg)


def analyse(X, ids, fams) -> dict:
    """the twilight-zone comparison for one embedding X (rows aligned with ids and fams)"""
    idx, sim = _topm_neighbors(X, 500)
    ph = pd.read_csv(ROOT / "results" / "baselines" / "heldout_phmmer_hits.csv.gz")
    ph["tfam"] = ph.target.map(dict(zip(ids, fams)))

    subsets = ("all", "hard_positives", "hard_negatives", "both_hard")
    auc = {s: {m: [] for m in METHODS} for s in subsets}
    per_pos = []
    n_hard = {"hard_positives": 0, "hard_negatives_mean": []}
    for F in np.unique(fams):
        is_f = fams == F
        best = ph[ph.tfam != F].groupby("query").bits.max().reindex(ids).to_numpy()
        bits = np.nan_to_num(best, nan=0.0)
        s = {"knn5": knn_excluding(idx, sim, is_f, 5), "knn1": knn_excluding(idx, sim, is_f, 1),
             "phmmer": -np.nan_to_num(best, nan=-1e6)}
        s["rank_average"] = (stats.rankdata(s["knn5"]) + stats.rankdata(s["phmmer"])) / 2
        hp, hn = is_f & (bits >= STRONG), ~is_f & (bits < STRONG)
        n_hard["hard_positives"] += int(hp.sum()); n_hard["hard_negatives_mean"].append(int(hn.sum()))
        masks = {"all": (is_f, ~is_f), "hard_positives": (hp, ~is_f), "hard_negatives": (is_f, hn), "both_hard": (hp, hn)}
        for sub, (pm, nm) in masks.items():
            if pm.sum() and nm.sum():
                y = np.r_[np.ones(pm.sum()), np.zeros(nm.sum())]
                for m in METHODS:
                    auc[sub][m].append(auroc(y, np.r_[s[m][pm], s[m][nm]]))
        pct = {m: percentile_vs(s[m][~is_f], s[m][is_f]) for m in METHODS}
        for j, b in enumerate(bits[is_f]):
            per_pos.append({"family": F, "bits": float(b), **{m: float(pct[m][j]) for m in METHODS}})

    pp = pd.DataFrame(per_pos)
    by_bin = {}
    for lo, hi, name in BINS:
        sel = pp[(pp.bits > lo) & (pp.bits <= hi)] if name != "no hit" else pp[pp.bits == 0]
        by_bin[name] = {"n_positives": int(len(sel)),
                        **{m: {"detected_at_5pct": float((sel[m] >= 0.95).mean()), "mean_percentile": float(sel[m].mean())}
                           for m in METHODS}}
    res = {"strong_hit_bits": STRONG, "n_families": int(len(np.unique(fams))),
           "n_hard_positives": n_hard["hard_positives"],
           "hard_negatives_per_family_median": float(np.median(n_hard["hard_negatives_mean"])),
           "auroc": {sub: {m: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "n_families": int(np.sum(~np.isnan(v)))}
                           for m, v in d.items()} for sub, d in auc.items()},
           "detection_by_best_hit": by_bin}
    return res


def main():
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    ids_all = np.array([str(i) for i in ref["ids"]]); fam_all = np.array([str(f) for f in ref["families"]])
    keep = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy() >= 5
    ids, fams = ids_all[keep], fam_all[keep]
    res = analyse(l2_normalize(ref["L22"].astype(np.float64)[keep]), ids, fams)
    out = ROOT / "results" / "revision" / "twilight_zone.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
