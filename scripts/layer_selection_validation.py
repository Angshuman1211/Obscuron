#!/usr/bin/env python
"""Independent check of the layer-22 choice (Review II, comment 4).

The headline layer was chosen on a 100-sequence / 20-family diagnostic, using an
embedding-separation gap, before the full benchmark was scored (P2-D7, P2-D8). That
leaves two fair questions: is layer 22 simply the best layer on the full benchmark by
construction, and does the choice hold on families that took no part in choosing it?

This script reads the all-layer embeddings written by scripts/layer_sweep_esm2.py and
answers both with the same held-out-family AUROC as the headline:

  1. AUROC of every layer 0..33 on all 903 families.
  2. The 20 families of the original diagnostic removed: layer 22 vs layer 33 on the rest.
  3. Development/test protocol: families are split at random (and, separately, by Pfam
     clan so related families never straddle the split); the layer is chosen as the
     argmax on the development families only and scored on the test families. Repeated
     over 50 random splits. Reported: which layer is chosen, its test AUROC, the test
     AUROC of layer 22 and of layer 33, and the regret of fixing layer 22.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/layer_selection_validation.py
Writes results/reviewer/layer_selection.json.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.scoring.knn import l2_normalize  # noqa: E402
from reembed_eval import _knn_dist_excluding, _topm_neighbors, auroc  # noqa: E402

PROC = ROOT / "data" / "processed" / "gtdb_R207"
SWEEP = PROC / "esm2_layer_sweep"
OUT = ROOT / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
K = 5
N_LAYERS = 34


def per_family_auroc(v: np.ndarray, fams: np.ndarray) -> dict[str, float]:
    vn = l2_normalize(v.astype(np.float64))
    idx, sim = _topm_neighbors(vn, 500)
    out = {}
    for f in np.unique(fams):
        is_f = fams == f
        novel = _knn_dist_excluding(idx[is_f], sim[is_f], is_f, K)
        known = _knn_dist_excluding(idx[~is_f], sim[~is_f], is_f, K)
        a = auroc(np.r_[np.ones(len(novel)), np.zeros(len(known))], np.r_[novel, known])
        if not np.isnan(a):
            out[str(f)] = float(a)
    return out


def main() -> None:
    fams = np.load(SWEEP / "families.npy")
    cache = OUT / "layer_sweep_per_family_auroc.csv"
    if cache.exists():
        table = pd.read_csv(cache, index_col=0)
    else:
        cols = {}
        for li in range(N_LAYERS):
            v = np.load(SWEEP / f"L{li:02d}.npy")
            cols[li] = pd.Series(per_family_auroc(v, fams))
            print(f"layer {li}: mean {cols[li].mean():.4f}", flush=True)
        table = pd.DataFrame(cols)
        table.to_csv(cache)
    table.columns = [int(c) for c in table.columns]
    res: dict = {"n_families": int(len(table))}
    res["all_families"] = {str(l): {"mean": float(table[l].mean()), "median": float(table[l].median())}
                           for l in table.columns}
    res["best_layer_all_families"] = int(table.mean().idxmax())

    # 2. drop the 20 diagnostic families
    sel = set(json.load(open(PROC / "genos-m_separation_full.json"))["families"].keys())
    rest = table.loc[[f for f in table.index if f not in sel]]
    res["without_original_20"] = {
        "n_selection_families_present": int(len(table) - len(rest)), "n_families": int(len(rest)),
        "layer22_mean": float(rest[22].mean()), "layer33_mean": float(rest[33].mean()),
        "layer22_median": float(rest[22].median()), "layer33_median": float(rest[33].median()),
        "best_layer": int(rest.mean().idxmax()),
    }

    # clan map for a clan-blocked split
    clan = {}
    with gzip.open(ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.clans.tsv.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            clan[p[0]] = p[1] if p[1] else p[0]
    groups = pd.Series({f: clan.get(f.split(".")[0], f.split(".")[0]) for f in table.index})
    res["n_clans_or_singletons"] = int(groups.nunique())

    def run(kind: str, n_rep: int = 50) -> dict:
        rows = []
        ug = groups.unique()
        for rep in range(n_rep):
            r = np.random.default_rng(2000 + rep)
            if kind == "family":
                dev_mask = pd.Series(r.random(len(table)) < 1 / 3, index=table.index)
            else:
                dev_groups = set(r.choice(ug, size=len(ug) // 3, replace=False))
                dev_mask = groups.isin(dev_groups)
            dev, test = table[dev_mask.values], table[~dev_mask.values]
            pick = int(dev.mean().idxmax())
            rows.append({"picked": pick, "test_picked": float(test[pick].mean()),
                         "test_22": float(test[22].mean()), "test_33": float(test[33].mean()),
                         "test_best_possible": float(test.mean().max()),
                         "n_dev": int(dev_mask.sum()), "n_test": int((~dev_mask).sum())})
        d = pd.DataFrame(rows)
        return {
            "n_splits": n_rep, "n_dev_mean": float(d.n_dev.mean()), "n_test_mean": float(d.n_test.mean()),
            "picked_layer_counts": {str(k): int(v) for k, v in d.picked.value_counts().sort_index().items()},
            "test_auroc_picked": float(d.test_picked.mean()), "test_auroc_layer22": float(d.test_22.mean()),
            "test_auroc_layer33": float(d.test_33.mean()),
            "test_auroc_oracle_layer": float(d.test_best_possible.mean()),
            "regret_of_fixing_22_vs_picked": float((d.test_picked - d.test_22).mean()),
        }

    res["dev_test_random_family_split"] = run("family")
    res["dev_test_clan_blocked_split"] = run("clan")

    # paired bootstrap over families: layer 22 minus layer 33
    r = np.random.default_rng(7)
    diff = (table[22] - table[33]).to_numpy()
    boots = [diff[r.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    res["layer22_minus_layer33_family_bootstrap"] = {
        "mean_diff": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "frac_families_layer22_better": float((diff > 0).mean())}
    (OUT / "layer_selection.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "all_families"}, indent=1))


if __name__ == "__main__":
    main()
