#!/usr/bin/env python
"""Is low structural confidence specific to the top of the ranking? (TCBB revision)

Compares the ESMFold and Foldseek results of the filtered top of the ranking (top --n proteins of
100-300 residues that pass the filter of scripts/top_ranked_checks.py) with a random draw of
still-dark proteins of the same length range from ranks 15,000 to 20,000 (seed 42), both produced
by scripts/case_studies.py:

    case_studies.py --filtered results/revision/filtered_ranking.csv --n 30 --out filtered_30.json
    case_studies.py --mid-ranked --n 30 --out mid_ranked_30.json

Reports per group the pLDDT distribution, the share folded with confidence (pLDDT >= 70), and the
share with an AlphaFold DB (afdb50) or PDB (pdb100) hit at E <= 1e-3; groups are compared with a
two-sided Mann-Whitney test (pLDDT, length) and Fisher's exact test (shares).

    .venv/Scripts/python.exe scripts/structure_comparison.py [--filtered filtered_30.json] [--mid mid_ranked_30.json]
Writes results/revision/structure_comparison.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results" / "case_studies"
E_MAX = 1e-3


def load(name: str) -> pd.DataFrame:
    rows = []
    for pid, v in json.loads((CASES / name).read_text()).items():
        row = {"pid": pid, "rank": v["rank_layer22"], "length": v["length"], "plddt": v["mean_plddt"]}
        for db in ("afdb50", "pdb100"):
            top = (v["foldseek"].get(db) or {}).get("top") or []
            row[f"{db}_hits"] = (v["foldseek"].get(db) or {}).get("n_hits", 0)
            row[f"{db}_best_e"] = min((float(a["eval"]) for a in top), default=np.inf)
        rows.append(row)
    return pd.DataFrame(rows)


def describe(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "ranks": [int(d["rank"].min()), int(d["rank"].max())],
            "length_median": float(d.length.median()),
            "plddt_median": float(d.plddt.median()), "plddt_iqr": [float(d.plddt.quantile(.25)), float(d.plddt.quantile(.75))],
            "plddt_range": [float(d.plddt.min()), float(d.plddt.max())],
            "confident_share": float((d.plddt >= 70).mean()),
            "afdb_hit_share": float((d.afdb50_best_e <= E_MAX).mean()),
            "pdb_hit_share": float((d.pdb100_best_e <= E_MAX).mean()),
            "any_afdb_alignment_share": float((d.afdb50_hits > 0).mean())}


def fisher(a: pd.Series, b: pd.Series) -> float:
    return float(stats.fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", default="filtered_30.json")
    ap.add_argument("--mid", default="mid_ranked_30.json")
    args = ap.parse_args()
    top, mid = load(args.filtered), load(args.mid)
    res = {"e_value_cutoff": E_MAX, "filtered_top": describe(top), "mid_ranked": describe(mid), "tests": {
        "plddt_mannwhitney_p": float(stats.mannwhitneyu(top.plddt, mid.plddt).pvalue),
        "length_mannwhitney_p": float(stats.mannwhitneyu(top.length, mid.length).pvalue),
        "confident_fisher_p": fisher(top.plddt >= 70, mid.plddt >= 70),
        "afdb_hit_fisher_p": fisher(top.afdb50_best_e <= E_MAX, mid.afdb50_best_e <= E_MAX),
        "pdb_hit_fisher_p": fisher(top.pdb100_best_e <= E_MAX, mid.pdb100_best_e <= E_MAX)}}
    # within the filtered top: do the few confident folds have structural neighbours?
    res["filtered_top_hits"] = top[top.afdb50_best_e <= E_MAX][["pid", "rank", "length", "plddt", "afdb50_best_e"]]\
        .to_dict("records")
    out = ROOT / "results" / "revision" / "structure_comparison.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
