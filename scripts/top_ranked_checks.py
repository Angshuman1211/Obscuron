#!/usr/bin/env python
"""Are the top-ranked dark proteins artefacts, and what does a filtered ranking look like?

For the 34,138 dark queries ranked by the layer-22 score (results/case_studies/dark_ranked_layer22.csv),
compare the top 1% with the rest on gene-call and sequence properties:
  length; Prodigal partial flag (gene runs off a contig end); codon-position base bias (CPBB, a
  reading-frame signal); amino-acid composition entropy; share of disorder-promoting residues
  (A, R, G, Q, S, P, E, K); AntiFam hits (known spurious ORFs; needs scripts/cross_database_scan.py).

A candidate filter is then applied whose thresholds come from CHARACTERISED proteins only
(3,000 drawn from the scorer reference), never from the dark set:
  complete gene (partial flag 00); 100 to 1,000 residues; composition entropy >= the 5th percentile
  of characterised proteins; disorder-promoting share <= their 95th percentile; CPBB >= their
  5th percentile; no AntiFam hit.
The survivors, in score order, are written for structure prediction.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/top_ranked_checks.py
Writes results/revision/top_ranked_checks.json and results/revision/filtered_ranking.csv.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

from darkmatter.statistical import codon_position_bias

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "revision"
DISORDER = set("ARGQSPEK")


def seq_props(seq):
    c = Counter(seq); n = len(seq)
    p = np.array([v / n for v in c.values()])
    return float(-(p * np.log2(p)).sum()), sum(c[a] for a in DISORDER) / n


def read(df):
    """protein sequence, partial flag and nucleotide CPBB for the proteins in df."""
    wanted = set(df.protein_id); rows = {}
    for acc in df.genome_accession.unique():
        for r in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
            if r.id in wanted:
                partial = r.description.split("partial=")[1][:2] if "partial=" in r.description else "??"
                rows[r.id] = {"seq": str(r.seq).rstrip("*"), "partial": partial}
        nt = PROC / "panel_nucleotides" / f"{acc}_protein.fna"
        if nt.exists():
            for r in SeqIO.parse(nt, "fasta"):
                if r.id in rows:
                    rows[r.id]["cpbb"] = codon_position_bias(str(r.seq))
    out = []
    for pid, v in rows.items():
        ent, dis = seq_props(v["seq"])
        out.append({"protein_id": pid, "length": len(v["seq"]), "partial": v["partial"], "entropy": ent,
                    "disorder": dis, "cpbb": v.get("cpbb", np.nan)})
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    t = pd.read_csv(ROOT / "results" / "case_studies" / "dark_ranked_layer22.csv").rename(columns={"pid": "protein_id"})
    t = t.merge(samp[["protein_id", "genome_accession"]], on="protein_id")
    feats = read(t)
    t = t.drop(columns=[c for c in ("cpbb", "length") if c in t.columns]).merge(feats, on="protein_id")
    xdb = ROOT / "results" / "revision" / "cross_database_hits.csv.gz"
    hits = pd.read_csv(xdb) if xdb.exists() else pd.DataFrame(columns=["protein_id", "library"])
    for lib in ("antifam", "ncbifam", "nmpfams"):
        t[lib] = t.protein_id.isin(set(hits.loc[hits.library == lib, "protein_id"]))
    t["complete"] = t.partial == "00"

    char = samp[samp.category == "characterised_at_t0"].sample(3000, random_state=42)
    cf = read(char)
    thr = {"entropy_p5": float(cf.entropy.quantile(0.05)), "disorder_p95": float(cf.disorder.quantile(0.95)),
           "cpbb_p5": float(cf.cpbb.quantile(0.05)), "characterised_partial_share": float((cf.partial != "00").mean())}

    top = t["rank"] <= int(round(0.01 * len(t)))
    res = {"n": int(len(t)), "n_top1pct": int(top.sum()), "thresholds_from_characterised": thr, "top1pct_vs_rest": {}}
    for col in ("length", "entropy", "disorder", "cpbb"):
        a, b = t.loc[top, col].dropna(), t.loc[~top, col].dropna()
        res["top1pct_vs_rest"][col] = {"top_median": float(a.median()), "rest_median": float(b.median()),
                                       "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue)}
    for col in ("complete", "antifam", "ncbifam", "nmpfams"):
        res["top1pct_vs_rest"][col + "_share"] = {"top": float(t.loc[top, col].mean()), "rest": float(t.loc[~top, col].mean())}
    res["top1pct_vs_rest"]["short_lt_100_share"] = {"top": float((t.loc[top, "length"] < 100).mean()),
                                                    "rest": float((t.loc[~top, "length"] < 100).mean())}

    keep = (t.complete & t.length.between(100, 1000) & (t.entropy >= thr["entropy_p5"]) &
            (t.disorder <= thr["disorder_p95"]) & (t.cpbb >= thr["cpbb_p5"]) & ~t.antifam)
    res["filter"] = {"pass_all": float(keep.mean()), "pass_top1pct": float(keep[top].mean()),
                     "failed_by": {"incomplete_gene": float((~t.complete[top]).mean()),
                                   "length": float((~t.length[top].between(100, 1000)).mean()),
                                   "low_complexity": float((t.entropy[top] < thr["entropy_p5"]).mean()),
                                   "disorder": float((t.disorder[top] > thr["disorder_p95"]).mean()),
                                   "weak_frame_signal": float((t.cpbb[top] < thr["cpbb_p5"]).mean()),
                                   "antifam": float(t.antifam[top].mean())}}
    filt = t[keep].sort_values("rank")
    filt[["protein_id", "rank", "d22", "cat", "length", "entropy", "disorder", "cpbb", "phylum", "ncbifam", "nmpfams"]]\
        .to_csv(OUT / "filtered_ranking.csv", index=False)
    res["filtered_top10"] = filt.head(10)[["protein_id", "rank", "length", "phylum"]].to_dict("records")
    (OUT / "top_ranked_checks.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "filtered_top10"}, indent=1, default=str))


if __name__ == "__main__":
    main()
