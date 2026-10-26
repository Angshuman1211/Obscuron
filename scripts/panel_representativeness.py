#!/usr/bin/env python
"""Panel design checks (TCBB revision, reviewer concern on the phylum-capped sampling).

1. Representative dark fraction: the panel caps every phylum at 2 to 30 genomes, which
   over-weights rare phyla. Re-weight each phylum's dark fraction by that phylum's share of the
   GTDB R207 species representatives and compare with the unweighted panel figure.
2. Genome quality of the panel from the R207 metadata: CheckM completeness and contamination,
   and the share of genomes that are metagenome-assembled.
3. Truncation: how many proteins exceed the 1,024-residue limit applied before embedding.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/panel_representativeness.py
Writes results/revision/panel_representativeness.json.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "gtdb_R207"
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "revision"


def r207_meta() -> pd.DataFrame:
    cols = ["accession", "gtdb_representative", "gtdb_taxonomy", "checkm_completeness", "checkm_contamination",
            "ncbi_genome_category"]
    frames = []
    for dom in ("bac120", "ar53"):
        with tarfile.open(RAW / f"{dom}_metadata_r207.tar.gz") as t:
            m = [x for x in t.getmembers() if x.name.endswith(".tsv")][0]
            frames.append(pd.read_csv(t.extractfile(m), sep="\t", usecols=lambda c: c in cols, low_memory=False))
    d = pd.concat(frames)
    d["phylum"] = d.gtdb_taxonomy.str.extract(r"p__([^;]+)")[0]
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = r207_meta()
    reps = meta[meta.gtdb_representative.astype(str).str.lower().isin(["t", "true"])]
    share = reps.phylum.value_counts(normalize=True)
    panel = pd.read_csv(PROC / "genome_panel.csv")
    lab = pd.read_csv(PROC / "panel_protein_labels.csv", usecols=["genome_accession", "dark_at_t0"])
    g = lab.groupby("genome_accession").dark_at_t0.agg(["sum", "count"]).reset_index()
    g = g.merge(panel[["accession", "phylum"]], left_on="genome_accession", right_on="accession")
    ph = g.groupby("phylum")[["sum", "count"]].sum()
    ph["dark_fraction"] = ph["sum"] / ph["count"]
    ph["r207_share"] = ph.index.map(share).fillna(0)
    weighted = float(np.average(ph.dark_fraction, weights=ph.r207_share))
    unweighted = float(lab.dark_at_t0.mean())
    top = ph.sort_values("r207_share", ascending=False).head(6)

    pm = meta.set_index("accession").reindex(panel.accession)
    qual = {"completeness_median": float(pm.checkm_completeness.median()), "completeness_min": float(pm.checkm_completeness.min()),
            "completeness_p5": float(pm.checkm_completeness.quantile(0.05)),
            "contamination_median": float(pm.checkm_contamination.median()), "contamination_max": float(pm.checkm_contamination.max()),
            "share_metagenome_derived": float(pm.ncbi_genome_category.astype(str).str.contains("metagenome", case=False).mean()),
            "share_single_cell": float(pm.ncbi_genome_category.astype(str).str.contains("single cell", case=False).mean())}

    res = {"dark_fraction_panel_unweighted": unweighted, "dark_fraction_weighted_by_r207_phylum_share": weighted,
           "n_phyla_panel": int(len(ph)), "r207_share_covered_by_panel_phyla": float(ph.r207_share.sum()),
           "largest_phyla": {k: {"r207_share": float(v.r207_share), "panel_dark_fraction": float(v.dark_fraction)} for k, v in top.iterrows()},
           "panel_genome_quality": qual}

    lens = pd.read_csv(ROOT / "results" / "scorer_esm2_ranked.csv", usecols=["protein_id", "length_aa"])
    res["truncation"] = {"dark_queries_over_1024": int((lens.length_aa > 1024).sum()), "dark_queries": int(len(lens)),
                         "share": float((lens.length_aa > 1024).mean())}
    (OUT / "panel_representativeness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
