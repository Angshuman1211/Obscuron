#!/usr/bin/env python
"""What do the most novel dark proteins look like? (TCBB revision, case-study context)

For the 34,138 dark queries ranked by the layer-22 score against the 29,862-protein reference,
compare the top 1% with the rest on three sequence properties that need no model:
  - amino-acid composition entropy (bits; low = low complexity),
  - fraction of charged residues (D, E, K, R, H),
  - fraction of disorder-promoting residues (A, R, G, Q, S, P, E, K; Dunker et al. 2001),
and report Mann-Whitney p-values and the Spearman correlation of each with the score.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/top_ranked_composition.py --ranked results/case_studies/dark_ranked_layer22.csv
Writes results/case_studies/top_ranked_composition.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "case_studies"
CHARGED, DISORDER = set("DEKRH"), set("ARGQSPEK")


def props(seq: str):
    c = Counter(seq); n = len(seq)
    p = np.array([v / n for v in c.values()])
    return {"entropy": float(-(p * np.log2(p)).sum()),
            "charged": sum(c[a] for a in CHARGED) / n,
            "disorder_promoting": sum(c[a] for a in DISORDER) / n}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ranked", required=True); args = ap.parse_args()
    t = pd.read_csv(args.ranked)
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    wanted = set(t.pid); seqs = {}
    for acc in samp.loc[list(wanted), "genome_accession"].unique():
        for r in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
            if r.id in wanted:
                seqs[r.id] = str(r.seq).rstrip("*")
    P = pd.DataFrame([{"pid": k, **props(v)} for k, v in seqs.items()])
    t = t.merge(P, on="pid")
    top = t["rank"] <= int(round(0.01 * len(t)))
    res = {"n": int(len(t)), "n_top": int(top.sum())}
    for col in ("entropy", "charged", "disorder_promoting"):
        a, b = t.loc[top, col], t.loc[~top, col]
        res[col] = {"top1pct_median": float(a.median()), "rest_median": float(b.median()),
                    "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue),
                    "spearman_with_distance": float(stats.spearmanr(t.d22, t[col])[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "top_ranked_composition.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
