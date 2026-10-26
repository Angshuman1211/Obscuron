#!/usr/bin/env python
"""Offline proxy: does layer + mean-centering improve the held-out-family AUROC
(the scorer's headline metric), using the saved layer-diagnostic vectors?

Runs the P2-D4 protocol -- withhold each family, score its members (novel) vs the
retained knowns (leave-one-out) by k=5 kNN novelty -- for every layer x {raw,
reference-centered}, on the 100-seq matched set. Shows whether the layer-22 +
centering representation beats the last-layer/raw one the current ESM-2 scorer
(full-panel AUROC 0.906) uses. Directional proxy on 20 families (small); the
definitive test is a full-panel re-embed, but this needs no GPU and answers
"is it worth it". Centering uses the REFERENCE (non-F) mean each fold -> leakage-safe.

    uv run python scripts/test_layer_scorer.py --model esm2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data" / "processed" / "gtdb_R207"
K = 5


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def knn_novelty(Q: np.ndarray, R: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    dist = 1.0 - Q @ R.T
    if exclude_self:
        np.fill_diagonal(dist, np.inf)
    dist = np.sort(dist, axis=1)
    kk = min(k, dist.shape[1] - (1 if exclude_self else 0))
    return dist[:, :kk].mean(axis=1)


def heldout_auroc(vectors: np.ndarray, labels: np.ndarray, center: bool) -> float:
    scores = []
    for fam in np.unique(labels):
        is_f = labels == fam
        ref_raw = vectors[~is_f]
        mu = ref_raw.mean(axis=0, keepdims=True) if center else 0.0

        def prep(x):
            x = x - mu
            return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

        ref = prep(ref_raw)
        novel = knn_novelty(prep(vectors[is_f]), ref, K, exclude_self=False)
        known = knn_novelty(ref, ref, K, exclude_self=True)
        y = np.r_[np.ones(len(novel)), np.zeros(len(known))]
        a = auroc(y, np.r_[novel, known])
        if not np.isnan(a):
            scores.append(a)
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["esm2", "genos-m"], default="esm2")
    args = ap.parse_args()

    d = np.load(PROC / f"{args.model}_layer_diagnostic.npz")
    layers, labels = d["layers"], d["labels"]
    print(f"{args.model}: {layers.shape[0]} layers, {layers.shape[1]} seqs, "
          f"{len(np.unique(labels))} families (k={K})\n")
    print("layer | AUROC(raw) | AUROC(centered)")
    print("------+-----------+----------------")
    rows = []
    for li in range(layers.shape[0]):
        v = layers[li]
        if not np.isfinite(v).all():
            print(f"{li:5d} |   (NaN)    |     (NaN)")
            continue
        a_raw, a_cen = heldout_auroc(v, labels, False), heldout_auroc(v, labels, True)
        rows.append((li, a_raw, a_cen))
        print(f"{li:5d} | {a_raw:9.4f} | {a_cen:14.4f}")

    last = rows[-1]
    best = max(rows, key=lambda r: r[2])
    print(f"\nbaseline  (last layer {last[0]}, raw):     AUROC {last[1]:.4f}")
    print(f"proposed  (layer {best[0]}, centered):     AUROC {best[2]:.4f}")
    print(f"full-panel ESM-2 scorer reference (last-layer, raw): 0.906")


if __name__ == "__main__":
    main()
