#!/usr/bin/env python
"""Layer 5 checks asked for in the Review II feedback (comment 7).

What the original Layer-5 numbers are: codon-position base bias (CPBB) has no fitted
parameters, so there is no training set and no held-out set. "Real vs shuffle 0.94" asks
how often a real dark gene scores higher than a surrogate of itself. That is a control
comparison on the same genes, not a prediction on independent data, and the report should
say so. This script adds what a reader needs to judge it:

  1. Null construction, repeated with several seeds, with a bootstrap CI.
  2. Real vs TRUE intergenic: the plain AUROC, then a length-matched version (CPBB is a
     chi-square-like statistic and depends on length; intergenic segments are shorter).
  3. The same comparison inside each of the 44 genomes that have intergenic controls.
  4. Representativeness of those 44 genomes against the 502-genome panel (phylum, coding GC).
  5. Controls: characterised genes against the same nulls, and CPBB separately for positives
     and still-dark genes.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/reviewer_coding_structure.py
Writes results/reviewer/coding_structure.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

from darkmatter.statistical import codon_position_bias, markov1_null, shuffle_seq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot(pos, neg, n=500):
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_nt(df: pd.DataFrame) -> dict[str, str]:
    wanted = set(df["protein_id"]); out = {}
    for acc in df["genome_accession"].unique():
        p = PROC / "panel_nucleotides" / f"{acc}_protein.fna"
        if p.exists():
            for rec in SeqIO.parse(p, "fasta"):
                if rec.id in wanted:
                    out[rec.id] = str(rec.seq)
    return out


def main() -> None:
    res: dict = {}
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    cs = pd.read_csv(PROC / "coding_structure.csv")
    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv")[["protein_id", "length_aa"]]
    d = samp.merge(cs, on="protein_id").merge(lens, on="protein_id", how="left")
    d["nt_len"] = d.length_aa * 3 + 3
    ig = pd.read_csv(PROC / "intergenic_controls.csv")

    # ---- 1. nulls with several seeds on a 4,000-gene sample ----
    sub = d.sample(4000, random_state=42)
    nt = load_nt(sub)
    sub = sub[sub.protein_id.isin(nt)]
    real = np.array([codon_position_bias(nt[p]) for p in sub.protein_id])
    runs = []
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        sh = np.array([codon_position_bias(shuffle_seq(nt[p], r)) for p in sub.protein_id])
        mk = np.array([codon_position_bias(markov1_null(nt[p], r)) for p in sub.protein_id])
        runs.append({"seed": 100 + seed, "real_vs_shuffle": auroc(real, sh), "real_vs_markov1": auroc(real, mk),
                     "paired_wins_shuffle": float(np.nanmean(real > sh)), "paired_wins_markov1": float(np.nanmean(real > mk))})
    res["null_repeats"] = {
        "n_genes": int(len(sub)), "runs": runs,
        "real_vs_shuffle_mean": float(np.mean([x["real_vs_shuffle"] for x in runs])),
        "real_vs_markov1_mean": float(np.mean([x["real_vs_markov1"] for x in runs])),
        "bootstrap_ci_real_vs_shuffle_last_seed": boot(real[~np.isnan(real)], sh[~np.isnan(sh)]),
        "bootstrap_ci_real_vs_markov1_last_seed": boot(real[~np.isnan(real)], mk[~np.isnan(mk)])}
    # CPBB of real genes by length, and of the nulls, to show the length dependence
    qs = np.quantile(sub.nt_len, [0, .25, .5, .75, 1])
    res["cpbb_by_length_quartile"] = [
        {"nt_len_range": [float(qs[i]), float(qs[i + 1])],
         "real_median": float(np.nanmedian(real[(sub.nt_len >= qs[i]) & (sub.nt_len <= qs[i + 1])])),
         "shuffle_median": float(np.nanmedian(sh[((sub.nt_len >= qs[i]) & (sub.nt_len <= qs[i + 1])).to_numpy()]))}
        for i in range(4)]

    # ---- 2. real vs true intergenic: plain and length matched ----
    dq = d[d.category.isin(["positive", "dark_negative"])].dropna(subset=["cpbb", "nt_len"])
    res["intergenic_plain"] = {"n_genes": int(len(dq)), "n_segments": int(len(ig)),
                               "auroc": auroc(dq.cpbb.to_numpy(), ig.cpbb.to_numpy()),
                               "median_gene_nt": float(dq.nt_len.median()), "median_segment_nt": float(ig.length.median())}
    edges = np.quantile(dq.nt_len, np.linspace(0, 1, 11)); edges[0] = 0; edges[-1] = 1e9
    bins = []
    for i in range(10):
        g = dq[(dq.nt_len > edges[i]) & (dq.nt_len <= edges[i + 1])].cpbb.to_numpy()
        s = ig[(ig.length > edges[i]) & (ig.length <= edges[i + 1])].cpbb.to_numpy()
        if len(g) > 20 and len(s) > 20:
            bins.append({"nt_range": [float(edges[i]), float(edges[i + 1])], "n_genes": int(len(g)),
                         "n_segments": int(len(s)), "auroc": auroc(g, s)})
    w = np.array([min(b["n_genes"], b["n_segments"]) for b in bins], float)
    res["intergenic_length_matched"] = {
        "bins": bins, "weighted_mean_auroc": float(np.average([b["auroc"] for b in bins], weights=w)),
        "min_bin_auroc": float(min(b["auroc"] for b in bins)), "max_bin_auroc": float(max(b["auroc"] for b in bins))}
    res["intergenic_plain"]["bootstrap_ci95"] = boot(dq.cpbb.to_numpy(), ig.cpbb.to_numpy(), 300)

    # ---- 3. within-genome ----
    per = []
    for acc, seg in ig.groupby("genome_accession"):
        g = dq[dq.genome_accession == acc].cpbb.to_numpy()
        if len(g) >= 20 and len(seg) >= 50:
            per.append({"genome": acc, "n_genes": int(len(g)), "n_segments": int(len(seg)),
                        "auroc": auroc(g, seg.cpbb.to_numpy())})
    pa = np.array([p["auroc"] for p in per])
    res["intergenic_within_genome"] = {
        "n_genomes_evaluated": int(len(per)), "median": float(np.median(pa)), "q1": float(np.percentile(pa, 25)),
        "q3": float(np.percentile(pa, 75)), "min": float(pa.min()), "max": float(pa.max()),
        "n_genomes_above_0.5": int((pa > 0.5).sum())}

    # ---- 4. representativeness of the 44 genomes ----
    panel = pd.read_csv(PROC / "genome_panel.csv")
    ig_genomes = set(ig.genome_accession.unique())
    panel["in44"] = panel.accession.isin(ig_genomes)
    qgen = d[d.category.isin(["positive", "dark_negative"])].groupby("genome_accession").size()
    ph = panel.groupby("phylum").agg(n_panel=("accession", "size"), n_44=("in44", "sum")).sort_values("n_panel", ascending=False)
    top = ph.head(8)
    res["representativeness"] = {
        "n_panel_genomes": int(len(panel)), "n_44": int(panel.in44.sum()),
        "n_phyla_panel": int(panel.phylum.nunique()), "n_phyla_44": int(panel[panel.in44].phylum.nunique()),
        "top_phyla_panel_vs_44": {k: [int(v.n_panel), int(v.n_44)] for k, v in top.iterrows()},
        "fisher_like_chi2_p_phylum": float(stats.chi2_contingency(
            np.vstack([ph.n_panel - ph.n_44, ph.n_44]).T[ph.n_panel >= 5].T)[1]),
        "share_of_dark_queries_in_44_genomes": float(qgen[qgen.index.isin(ig_genomes)].sum() / qgen.sum())}
    gc = {}
    for acc in panel.accession:
        p = PROC / "panel_nucleotides" / f"{acc}_protein.fna"
        if not p.exists():
            continue
        g = t = 0
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith(">"):
                    s = line.strip().upper()
                    g += s.count("G") + s.count("C"); t += len(s)
        gc[acc] = g / max(t, 1)
    gcs = pd.Series(gc)
    in44 = gcs[gcs.index.isin(ig_genomes)]; out44 = gcs[~gcs.index.isin(ig_genomes)]
    res["representativeness"]["coding_gc"] = {
        "panel_median": float(gcs.median()), "panel_iqr": [float(gcs.quantile(.25)), float(gcs.quantile(.75))],
        "in44_median": float(in44.median()), "in44_iqr": [float(in44.quantile(.25)), float(in44.quantile(.75))],
        "ks_pvalue_44_vs_rest": float(stats.ks_2samp(in44, out44).pvalue)}
    # does the within-genome AUROC depend on GC?
    if per:
        pg = pd.DataFrame(per).set_index("genome"); pg["gc"] = gcs.reindex(pg.index)
        res["intergenic_within_genome"]["spearman_auroc_vs_gc"] = float(stats.spearmanr(pg.auroc, pg.gc, nan_policy="omit")[0])

    # ---- 5. controls ----
    ctrl = d[d.category == "characterised_at_t0"].sample(3000, random_state=42) if (d.category == "characterised_at_t0").any() else None
    if ctrl is None:
        full = samp[samp.category == "characterised_at_t0"].sample(3000, random_state=42)
        nt_c = load_nt(full)
        cpbb_c = np.array([codon_position_bias(nt_c[p]) for p in full.protein_id if p in nt_c])
        r = np.random.default_rng(7)
        sh_c = np.array([codon_position_bias(shuffle_seq(nt_c[p], r)) for p in full.protein_id if p in nt_c])
        res["characterised_control"] = {"n": int(len(cpbb_c)), "real_median": float(np.nanmedian(cpbb_c)),
                                        "real_vs_shuffle": auroc(cpbb_c, sh_c),
                                        "dark_vs_characterised_cpbb_auroc": auroc(dq.cpbb.to_numpy(), cpbb_c),
                                        "dark_median": float(dq.cpbb.median())}
    pos = dq[dq.category == "positive"].cpbb.to_numpy(); neg = dq[dq.category == "dark_negative"].cpbb.to_numpy()
    res["cpbb_positive_vs_still_dark"] = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "auroc": auroc(pos, neg),
                                          "ci95": boot(pos, neg, 300),
                                          "mannwhitney_p": float(stats.mannwhitneyu(pos, neg).pvalue),
                                          "median_pos": float(np.median(pos)), "median_neg": float(np.median(neg))}
    ent_p = d[d.category == "positive"].aa_entropy_k3.dropna().to_numpy()
    ent_n = d[d.category == "dark_negative"].aa_entropy_k3.dropna().to_numpy()
    res["aa_entropy_positive_vs_still_dark"] = {"auroc": auroc(ent_p, ent_n), "ci95": boot(ent_p, ent_n, 300)}
    (OUT / "coding_structure.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
