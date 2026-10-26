#!/usr/bin/env python
"""Track-2 Phase-3 hand-off for the Layer-3 immune self/non-self detector
(docs/Track1_phase3_immune_design.md). Track 1 built and froze the method
(darkmatter/immune/, config/immune.yaml, P3-D1..D6); this script does the
three things the design doc hands to Track 2:

  1. Parameter sweeps -- detector count, margin_factor, PCA dimension
     (deliverable 2 hand-off) and self-tolerance alpha (deliverable 3
     hand-off, reported as a coverage/error trade-off curve) -- each graded
     by held-out-family AUROC or held-out-self flag rate.
  2. Scaling flagging to the FULL dark-query set (34,138 queries: 30,000
     dark_negative + 4,138 positive) -- P3-D6 explicitly left this untested
     ("needs layer-22 dark embeddings -- a follow-up, not required for the
     narrative"). Also runs the Layer-1 EVT scorer on the same layer-22
     embeddings so the two methods' full-scale convergence (Spearman +
     top-K Jaccard) and flag-rate-vs-positive enrichment are measured on
     the real query population, not just the ~700-member held-out-family
     subset P3-D6 used.
  3. Writes comparison tables (results/) + figures (data/processed/gtdb_R207/figures/)
     for review (deliverable 4 hand-off).

Requires (built by Track 2, this phase):
    uv run python scripts/reembed_eval.py --model esm2 --layers 33,22 --fp32
    uv run python scripts/reembed_dark_queries.py --layers 33,22 --fp32

    uv run python scripts/immune_sweep.py            # full (config/immune.yaml base)
    uv run python scripts/immune_sweep.py --smoke    # fast end-to-end check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

from darkmatter.experiment_log import log_experiment
from darkmatter.immune.negative_selection import VDetector, _l2, fit_pca
from darkmatter.scoring.knn import l2_normalize
from darkmatter.scoring.scorer import NoveltyScorer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
FIG_DIR = PROC / "figures"
RESULTS = ROOT / "results"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return np.nan if npos == 0 or nneg == 0 else (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def heldout_family_auroc(X: np.ndarray, fams: np.ndarray, *, pca_dims: int, seed: int,
                         eval_families: int, eval_n_detectors: int, **detector_kwargs) -> tuple[float, float, int]:
    """Rebuild a repertoire per held-out family (shared PCA), same protocol as
    scripts/run_immune.py's evaluation loop, generalized to sweep any VDetector
    hyperparameter via **detector_kwargs."""
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


def load_reference(min_members: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(PROC / "esm2_reembed_L33_22.npz")
    X = d["L22"].astype(np.float64)
    fams = np.array([str(x) for x in d["families"]])
    sizes = dict(zip(*np.unique(fams, return_counts=True)))
    keep = np.array([sizes[f] >= min_members for f in fams])
    return X[keep], fams[keep]


def load_queries() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(PROC / "esm2_reembed_queries_L33_22.npz")
    return d["L22"].astype(np.float64), d["is_positive"].astype(bool), np.array([str(x) for x in d["ids"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "immune.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    D, C, E = cfg["detector"], cfg["calibration"], cfg["evaluation"]
    base_kwargs = dict(margin_factor=D["margin_factor"], floor_factor=D["floor_factor"],
                       expand=D["expand"], sampling=D.get("sampling", "box"),
                       jitter_scale=D.get("jitter_scale", 3.0), max_candidate_mult=D["max_candidate_mult"])
    seed = D["seed"]
    sweep_eval_families = 8 if args.smoke else 30

    X_ref, fams = load_reference(E["min_members"])
    print(f"{len(X_ref)} reference proteins in {len(set(fams))} families (>= {E['min_members']} members)", flush=True)
    X_q, is_pos, q_ids = load_queries()
    if args.smoke:
        keep = np.r_[np.where(is_pos)[0][:100], np.where(~is_pos)[0][:200]]
        X_q, is_pos, q_ids = X_q[keep], is_pos[keep], q_ids[keep]
    print(f"{len(X_q)} dark queries ({int(is_pos.sum())} positive)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # ---------- 1a. detector-count sweep (held-out-family AUROC) ----------
    counts = [500, 1500, 5000] if args.smoke else [1000, 5000, 10000, 20000]
    rows = []
    t0 = time.time()
    for n in counts:
        m, med, nf = heldout_family_auroc(X_ref, fams, pca_dims=D["pca_dims"], seed=seed,
                                          eval_families=sweep_eval_families, eval_n_detectors=n, **base_kwargs)
        rows.append({"n_detectors": n, "auroc_mean": m, "auroc_median": med, "n_families": nf})
        print(f"  detectors={n:6d}  AUROC mean={m:.4f} median={med:.4f} (n={nf}) [{time.time()-t0:.0f}s]", flush=True)
    detector_sweep = pd.DataFrame(rows)
    detector_sweep.to_csv(RESULTS / "immune_sweep_detector_count.csv", index=False)

    # ---------- 1b. margin_factor sweep ----------
    margins = [1.0, 2.0] if args.smoke else [0.5, 1.0, 1.5, 2.0]
    rows = []
    for mf in margins:
        kwargs = {**base_kwargs, "margin_factor": mf}
        m, med, nf = heldout_family_auroc(X_ref, fams, pca_dims=D["pca_dims"], seed=seed,
                                          eval_families=sweep_eval_families, eval_n_detectors=E["eval_n_detectors"], **kwargs)
        rows.append({"margin_factor": mf, "auroc_mean": m, "auroc_median": med, "n_families": nf})
        print(f"  margin_factor={mf:.2f}  AUROC mean={m:.4f} median={med:.4f} (n={nf})", flush=True)
    margin_sweep = pd.DataFrame(rows)
    margin_sweep.to_csv(RESULTS / "immune_sweep_margin.csv", index=False)

    # ---------- 1c. PCA-dimension sweep ----------
    pca_dims_list = [50] if args.smoke else [0, 50, 100]
    rows = []
    for pd_ in pca_dims_list:
        m, med, nf = heldout_family_auroc(X_ref, fams, pca_dims=pd_, seed=seed,
                                          eval_families=sweep_eval_families, eval_n_detectors=E["eval_n_detectors"], **base_kwargs)
        rows.append({"pca_dims": pd_ if pd_ else X_ref.shape[1], "auroc_mean": m, "auroc_median": med, "n_families": nf})
        print(f"  pca_dims={pd_ or X_ref.shape[1]:5d}  AUROC mean={m:.4f} median={med:.4f} (n={nf})", flush=True)
    pca_sweep = pd.DataFrame(rows)
    pca_sweep.to_csv(RESULTS / "immune_sweep_pca_dims.csv", index=False)

    # ---------- 2. full repertoire fit (base config) for alpha sweep + full-scale flagging ----------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_ref))
    n_hold = int(C["heldout_self_frac"] * len(X_ref))
    hold, build = X_ref[idx[:n_hold]], X_ref[idx[n_hold:]]
    vd = VDetector(n_detectors=D["n_detectors"], pca_dims=D["pca_dims"], seed=seed, **base_kwargs)
    t0 = time.time()
    vd.fit(build)
    self_scores = vd.score(hold)
    query_scores = vd.score(X_q)
    print(f"full repertoire: {vd.n_kept_} detectors, fit+score in {time.time()-t0:.0f}s", flush=True)

    # ---------- 2a. alpha sweep -- self-tolerance coverage/error trade-off ----------
    alphas = np.array(sorted(set(C["alpha_sensitivity"] + [C["alpha"]])))
    rows = []
    for a in alphas:
        thr = float(np.quantile(self_scores, 1.0 - a))
        self_flag = float((self_scores > thr).mean())
        dark_flag = float((query_scores > thr).mean())
        pos_flag = float((query_scores[is_pos] > thr).mean())
        neg_flag = float((query_scores[~is_pos] > thr).mean())
        rows.append({"alpha": a, "threshold": thr, "self_flag_rate": self_flag,
                    "dark_query_flag_rate": dark_flag, "positive_flag_rate": pos_flag,
                    "dark_negative_flag_rate": neg_flag})
    alpha_sweep = pd.DataFrame(rows)
    alpha_sweep.to_csv(RESULTS / "immune_sweep_alpha.csv", index=False)
    print("\nalpha | self_flag | dark_flag | positive_flag | dark_negative_flag")
    for _, r in alpha_sweep.iterrows():
        print(f"{r['alpha']:.2f}  | {r['self_flag_rate']:.3f}     | {r['dark_query_flag_rate']:.3f}     "
              f"| {r['positive_flag_rate']:.3f}         | {r['dark_negative_flag_rate']:.3f}")

    # ---------- 3. Layer-1 EVT scorer at layer-22, same reference/queries ----------
    scorer_cfg = yaml.safe_load(open(ROOT / "config" / "scorer.yaml"))
    scorer = NoveltyScorer(scorer_cfg)
    scorer.fit(l2_normalize(X_ref))
    layer1 = scorer.score(l2_normalize(X_q))
    layer1_novelty = layer1["novelty"].to_numpy()

    # ---------- 4. full-scale convergence + per-query flags ----------
    vd.calibrate(hold, C["alpha"])
    query_flags = vd.flag(X_q)
    rho = float(spearmanr(query_scores, layer1_novelty).correlation)
    overlaps = {}
    for K in E["topk_overlap"]:
        k = min(K, len(query_scores))
        a_set = set(np.argsort(-query_scores)[:k])
        b_set = set(np.argsort(-layer1_novelty)[:k])
        overlaps[str(K)] = len(a_set & b_set) / len(a_set | b_set)

    flags_df = pd.DataFrame({
        "protein_id": q_ids, "is_positive": is_pos,
        "layer1_novelty": layer1_novelty, "layer3_nonself_score": query_scores,
        "layer3_flag": query_flags,
    })
    flags_df.to_csv(RESULTS / "immune_dark_query_flags.csv", index=False)

    summary = {
        "n_detectors": vd.n_kept_, "pca_dims": D["pca_dims"], "alpha": C["alpha"],
        "self_flag_rate_at_alpha": float((self_scores > vd.threshold_).mean()),
        "n_reference": len(X_ref), "n_queries": len(X_q), "n_positive": int(is_pos.sum()),
        "full_scale_flag_rate_dark_query": float(query_flags.mean()),
        "full_scale_flag_rate_positive": float(query_flags[is_pos].mean()),
        "full_scale_flag_rate_dark_negative": float(query_flags[~is_pos].mean()),
        "full_scale_convergence_spearman": rho,
        "full_scale_topk_jaccard": overlaps,
    }
    (RESULTS / "immune_scale_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== full-scale dark-query flagging (Layer 3) ===")
    print(f"flag rate: dark queries {summary['full_scale_flag_rate_dark_query']:.4f}  "
          f"positive {summary['full_scale_flag_rate_positive']:.4f}  "
          f"dark_negative {summary['full_scale_flag_rate_dark_negative']:.4f}")
    print(f"convergence with Layer-1 (full scale, n={len(X_q)}): Spearman rho = {rho:.3f}")
    print("top-K jaccard: " + "  ".join(f"K={k}:{v:.3f}" for k, v in overlaps.items()))

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(alpha_sweep["alpha"], alpha_sweep["self_flag_rate"], "o-", label="self flag rate (error)")
    axes[0].plot(alpha_sweep["alpha"], alpha_sweep["dark_query_flag_rate"], "s-", label="dark-query flag rate (coverage)")
    axes[0].plot([0, alpha_sweep["alpha"].max()], [0, alpha_sweep["alpha"].max()], "k--", lw=0.7, label="y=x")
    axes[0].set_xlabel("alpha (self-tolerance target)")
    axes[0].set_ylabel("flag rate")
    axes[0].set_title("Self-tolerance coverage/error trade-off")
    axes[0].legend(fontsize=8)

    axes[1].bar([str(n) for n in detector_sweep["n_detectors"]], detector_sweep["auroc_mean"])
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.7)
    axes[1].set_xlabel("n_detectors")
    axes[1].set_ylabel("held-out-family AUROC (mean)")
    axes[1].set_title("Detector-count sweep")

    axes[2].scatter(layer1_novelty, query_scores, s=2, alpha=0.15, c=np.where(is_pos, "tab:red", "tab:blue"))
    axes[2].set_xlabel("Layer-1 EVT novelty (layer 22)")
    axes[2].set_ylabel("Layer-3 non-self score")
    axes[2].set_title(f"Full-scale convergence (rho={rho:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "immune_sweep_summary.png", dpi=150)
    plt.close(fig)

    log_experiment(
        title="immune layer-3 sweep + full-scale dark-query flagging (esm2, layer 22)",
        config=f"n_detectors sweep {counts}, margin sweep {margins}, pca sweep {pca_dims_list}, "
              f"alpha sweep {list(alphas)}, base config from {args.config}",
        result=(f"detector-count AUROC {detector_sweep['auroc_mean'].tolist()}; "
                f"margin AUROC {margin_sweep['auroc_mean'].tolist()}; "
                f"pca AUROC {pca_sweep['auroc_mean'].tolist()}; "
                f"full-scale ({len(X_q)} queries) flag rate dark={summary['full_scale_flag_rate_dark_query']:.4f} "
                f"positive={summary['full_scale_flag_rate_positive']:.4f} "
                f"dark_negative={summary['full_scale_flag_rate_dark_negative']:.4f}; "
                f"convergence with Layer-1 rho={rho:.3f}, top-K jaccard={overlaps}"),
        next_step="report to Track 1 against P3-D6's acceptance sanity + fold into manuscript (Phase 4)",
    )
    print(f"\nwrote results/immune_sweep_*.csv, results/immune_scale_summary.json, "
          f"results/immune_dark_query_flags.csv, {FIG_DIR / 'immune_sweep_summary.png'}")


if __name__ == "__main__":
    main()
