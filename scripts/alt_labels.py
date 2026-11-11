#!/usr/bin/env python
"""Do family resources other than Pfam agree with the score? (TCBB revision)

The retrospective labels come from Pfam alone. Two independent family resources were scanned for
all 34,138 dark queries (scripts/cross_database_scan.py): NCBIfam/TIGRFAM at trusted cut-offs and
the NMPFamsDB catalogue of metagenomic families (E <= 1e-5). A dark protein recognised by either
resource has relatives that someone has already grouped into a family; one recognised by neither is
dark beyond Pfam. If the score measures novelty rather than something Pfam-specific, proteins it
ranks high should more often be recognised by neither.

For each score (distance and density-normalised, layer 22; phmmer best hit; length) this reports
the AUROC for "recognised by no external resource" against "recognised by at least one", for each
resource separately and for both, and the share recognised by decile of the score. A tie-aware AUROC
is used. Higher scores mean more novel for every method (phmmer: minus the best bit score; length:
minus the length, since short proteins are less often in families).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/alt_labels.py
Writes results/revision/alt_labels.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from revision_scoring import auroc  # noqa: E402

rng = np.random.default_rng(42)


def boot(y, s, n=500):
    v = [auroc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    d = pd.read_csv(ROOT / "results" / "revision" / "dark_scores.csv.gz")
    hits = pd.read_csv(ROOT / "results" / "revision" / "cross_database_hits.csv.gz")
    for lib in ("ncbifam", "nmpfams"):
        d[lib] = d.protein_id.isin(set(hits.loc[hits.library == lib, "protein_id"]))
    d["any_external"] = d.ncbifam | d.nmpfams
    ph = pd.read_csv(ROOT / "results" / "baselines" / "darkquery_phmmer_besthit.csv.gz").groupby("query").bits.max()
    d["phmmer"] = -ph.reindex(d.protein_id).fillna(0).to_numpy()
    lens = pd.read_csv(ROOT / "results" / "case_studies" / "dark_ranked_layer22.csv").set_index("pid")["len"]
    d["length"] = -lens.reindex(d.protein_id).to_numpy()
    scores = ("distance", "normalised", "phmmer", "length")
    res = {"n": int(len(d)), "recognised_share": {lib: float(d[lib].mean()) for lib in ("ncbifam", "nmpfams", "any_external")}}
    for label in ("ncbifam", "nmpfams", "any_external"):
        y = (~d[label]).to_numpy().astype(int)  # 1 = recognised by none of this resource: novel
        res[label] = {}
        for s in scores:
            x = d[s].to_numpy(float)
            res[label][s] = {"auroc": auroc(y, x), "ci95": boot(y, x)}
        # same, within still-dark proteins only, so the Pfam positives cannot drive it
        m = d.positive.to_numpy() == 0
        res[label]["still_dark_only"] = {s: auroc(y[m], d[s].to_numpy(float)[m]) for s in scores}
    dec = {}
    for s in ("distance", "normalised"):
        q = pd.qcut(d[s].rank(method="first"), 10, labels=False)
        dec[s] = d.groupby(q).any_external.mean().round(4).tolist()  # decile 0 = least novel
    res["recognised_share_by_decile"] = dec
    out = ROOT / "results" / "revision" / "alt_labels.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
