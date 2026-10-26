#!/usr/bin/env python
"""Extends the P3-D7 detector-count sweep (1k/5k/10k/20k, still climbing at 20k)
to find where held-out-family AUROC actually plateaus. This is exploratory
measurement, not a config change -- config/immune.yaml's frozen n_detectors=5000
stays untouched; whether/how to update it is Track 1's call, made with this data
in hand, per the anti-fishing rule (P2-D5/P3-D5: hyperparameters are frozen
pre-result and changed only by a reviewed edit stating a reason, never chased
post-hoc). Reuses the exact same protocol, families, and seed as
scripts/immune_sweep.py's detector-count sweep so the new points are directly
appendable to results/immune_sweep_detector_count.csv.

    uv run python scripts/immune_detector_saturation.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from darkmatter.immune.negative_selection import VDetector, fit_pca
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
RESULTS = ROOT / "results"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return np.nan if npos == 0 or nneg == 0 else (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def heldout_family_auroc(X, fams, *, pca_dims, seed, eval_families, eval_n_detectors, **detector_kwargs):
    pca = fit_pca(X, pca_dims) if pca_dims else (None, None)
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(fams)))
    eval_fams = uniq if len(uniq) <= eval_families else rng.choice(uniq, eval_families, replace=False)
    aurocs = []
    for i, fam in enumerate(eval_fams):
        is_f = fams == fam
        rest = X[~is_f]
        perm = rng.permutation(len(rest))
        n_known = min(len(rest) // 5, max(50, 5 * int(is_f.sum())))
        known, build_f = rest[perm[:n_known]], rest[perm[n_known:]]
        novel = X[is_f]
        vf = VDetector(n_detectors=eval_n_detectors, pca_dims=pca_dims, seed=seed + i + 1, **detector_kwargs)
        vf.pca_mean_, vf.pca_comp_ = pca
        vf.fit(build_f)
        a = auroc(np.r_[np.ones(len(novel)), np.zeros(len(known))], np.r_[vf.score(novel), vf.score(known)])
        if not np.isnan(a):
            aurocs.append(a)
    if not aurocs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aurocs)), float(np.median(aurocs)), len(aurocs)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config" / "immune.yaml"))
    D, E = cfg["detector"], cfg["evaluation"]
    base_kwargs = dict(margin_factor=D["margin_factor"], floor_factor=D["floor_factor"],
                       expand=D["expand"], sampling=D.get("sampling", "box"),
                       jitter_scale=D.get("jitter_scale", 3.0), max_candidate_mult=D["max_candidate_mult"])
    seed = D["seed"]

    d = np.load(PROC / "esm2_reembed_L33_22.npz")
    X = d["L22"].astype(np.float64)
    fams = np.array([str(x) for x in d["families"]])
    sizes = dict(zip(*np.unique(fams, return_counts=True)))
    keep = np.array([sizes[f] >= E["min_members"] for f in fams])
    X, fams = X[keep], fams[keep]
    print(f"{len(X)} reference proteins in {len(set(fams))} families", flush=True)

    counts = [20000, 30000, 40000, 50000]  # 20000 repeated as a consistency check on the earlier run
    rows = []
    t0 = time.time()
    for n in counts:
        m, med, nf = heldout_family_auroc(X, fams, pca_dims=D["pca_dims"], seed=seed,
                                          eval_families=30, eval_n_detectors=n, **base_kwargs)
        rows.append({"n_detectors": n, "auroc_mean": m, "auroc_median": med, "n_families": nf})
        print(f"  detectors={n:6d}  AUROC mean={m:.4f} median={med:.4f} (n={nf}) [{time.time()-t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "immune_sweep_detector_saturation.csv", index=False)
    print(f"\nwrote {RESULTS / 'immune_sweep_detector_saturation.csv'}")


if __name__ == "__main__":
    main()
