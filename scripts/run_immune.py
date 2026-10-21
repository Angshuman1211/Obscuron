#!/usr/bin/env python
"""Layer 3 — calibrated negative selection over the layer-22 reference embeddings.

Builds a V-detector repertoire on "self" (characterised reference), calibrates the
decision to a self-tolerance bound alpha, then evaluates: (1) self-tolerance holds?
(2) held-out-family AUROC (P2-D4 protocol, rebuild per family), (3) convergence with
the Layer-1 kNN signal (Spearman + top-K flag overlap). Runs on the existing
esm2_reembed_L33_22.npz -- no GPU, no re-embed. See P3-D1..D5.

    uv run python scripts/run_immune.py            # full (config/immune.yaml)
    uv run python scripts/run_immune.py --smoke    # fast end-to-end check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import rankdata, spearmanr

from darkmatter.immune.negative_selection import VDetector, _l2, fit_pca

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)  # average ranks -> tie-safe (most points are uncovered, score 0)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return np.nan if npos == 0 or nneg == 0 else (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def knn_dist(Q: np.ndarray, R: np.ndarray, k: int, chunk: int = 1024) -> np.ndarray:
    """Mean cosine distance to the k nearest of R -- the Layer-1 novelty proxy."""
    qn, rn = _l2(Q), _l2(R)
    out = np.empty(len(Q))
    for a in range(0, len(Q), chunk):
        d = 1.0 - qn[a:a + chunk] @ rn.T
        d.sort(axis=1)
        out[a:a + chunk] = d[:, :k].mean(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "immune.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    D, C, E = cfg["detector"], cfg["calibration"], cfg["evaluation"]
    if args.smoke:
        D = {**D, "n_detectors": 500}
        E = {**E, "eval_families": 8, "eval_n_detectors": 300}

    d = np.load(PROC / "esm2_reembed_L33_22.npz")
    X = d[f"L{cfg['arm']['layer']}"].astype(np.float64)
    fams = np.array([str(x) for x in d["families"]])

    sizes = dict(zip(*np.unique(fams, return_counts=True)))
    keep = np.array([sizes[f] >= E["min_members"] for f in fams])
    X, fams = X[keep], fams[keep]
    uniq = np.array(sorted(set(fams)))
    print(f"{len(X)} reference proteins in {len(uniq)} families (>= {E['min_members']} members)", flush=True)

    rng = np.random.default_rng(D["seed"])

    def make_detector(seed: int, n: int) -> VDetector:
        return VDetector(n_detectors=n, margin_factor=D["margin_factor"], floor_factor=D["floor_factor"],
                         expand=D["expand"], sampling=D.get("sampling", "jitter"),
                         jitter_scale=D.get("jitter_scale", 3.0), pca_dims=D["pca_dims"],
                         max_candidate_mult=D["max_candidate_mult"], seed=seed)

    # --- 1. full-repertoire fit + self-tolerance calibration ---
    idx = rng.permutation(len(X))
    n_hold = int(C["heldout_self_frac"] * len(X))
    hold, build = X[idx[:n_hold]], X[idx[n_hold:]]
    vd = make_detector(D["seed"], D["n_detectors"])
    t0 = time.time()
    vd.fit(build)
    vd.calibrate(hold, C["alpha"])
    self_flag_rate = float(vd.flag(hold).mean())
    print(f"repertoire: {vd.n_kept_} detectors in {time.time() - t0:.0f}s; "
          f"self-tolerance target alpha={C['alpha']} -> held-out-self flag rate {self_flag_rate:.3f}", flush=True)

    # --- 2. held-out-family AUROC + Layer-1 convergence (rebuild per family, shared PCA) ---
    pca = fit_pca(X, D["pca_dims"]) if D["pca_dims"] else (None, None)
    eval_fams = uniq if len(uniq) <= E["eval_families"] else rng.choice(uniq, E["eval_families"], replace=False)
    aurocs, ns_all, knn_all = [], [], []
    t0 = time.time()
    for i, fam in enumerate(eval_fams):
        is_f = fams == fam
        rest = X[~is_f]
        perm = rng.permutation(len(rest))
        n_known = min(len(rest) // 5, max(50, 5 * int(is_f.sum())))
        known, build_f = rest[perm[:n_known]], rest[perm[n_known:]]
        novel = X[is_f]

        vf = make_detector(D["seed"] + i + 1, E["eval_n_detectors"])
        vf.pca_mean_, vf.pca_comp_ = pca
        vf.fit(build_f)
        ns_novel, ns_known = vf.score(novel), vf.score(known)
        a = auroc(np.r_[np.ones(len(novel)), np.zeros(len(known))], np.r_[ns_novel, ns_known])
        if not np.isnan(a):
            aurocs.append(a)
        ns_all.append(ns_novel)
        knn_all.append(knn_dist(novel, build_f, E["knn_k"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(eval_fams)} families ({time.time() - t0:.0f}s)", flush=True)

    ns_all, knn_all = np.concatenate(ns_all), np.concatenate(knn_all)
    rho = float(spearmanr(ns_all, knn_all).correlation)
    overlaps = {}
    for K in E["topk_overlap"]:
        k = min(K, len(ns_all))
        a = set(np.argsort(-ns_all)[:k])
        b = set(np.argsort(-knn_all)[:k])
        overlaps[str(K)] = len(a & b) / len(a | b)

    res = {
        "n_detectors": vd.n_kept_, "pca_dims": D["pca_dims"], "alpha": C["alpha"],
        "self_flag_rate": self_flag_rate,
        "heldout_family_auroc_mean": float(np.mean(aurocs)),
        "heldout_family_auroc_median": float(np.median(aurocs)),
        "n_families": len(aurocs),
        "convergence_spearman_vs_knn": rho,
        "topk_jaccard_vs_knn": overlaps,
    }
    (PROC / "immune_results.json").write_text(json.dumps(res, indent=2))

    print("\n=== Layer 3: calibrated negative selection (immune self/non-self) ===")
    print(f"detectors {vd.n_kept_}   pca_dims {D['pca_dims']}   alpha {C['alpha']}")
    print(f"self-tolerance (held-out-self flag rate): {self_flag_rate:.3f}  (target <= {C['alpha']})")
    print(f"held-out-family AUROC: mean {np.mean(aurocs):.4f}  median {np.median(aurocs):.4f}  (n={len(aurocs)})")
    print(f"convergence with Layer-1 kNN: Spearman rho = {rho:.3f}  (n={len(ns_all)} novel members)")
    print("top-K flag Jaccard (NS vs kNN): " + "  ".join(f"K={K}:{v:.3f}" for K, v in overlaps.items()))
    print(f"\nLayer-1 reference: full-scale held-out-family AUROC = 0.962 (P2-D9)")
    print(f"wrote {PROC / 'immune_results.json'}")


if __name__ == "__main__":
    main()
