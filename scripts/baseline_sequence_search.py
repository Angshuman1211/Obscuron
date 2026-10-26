#!/usr/bin/env python
"""Sequence-search baselines for the TCBB comparison (HMMER profile search and phmmer).

The incumbent way to decide that a protein is novel is that sequence or profile search finds
nothing close. This script records, for the proteins of both evaluation tasks, the raw search
results that the baselines need, so that `scripts/baseline_comparison.py` can score them under
the same protocol as the embedding score:

  1. Pfam 35.0 profile search (hmmsearch, every family, E <= 10) for the 11,527 held-out-family
     proteins: per protein, the best sequence bit score and that family's gathering threshold,
     for every family it hits. Hiding a family then means ignoring that family's own profile.
  2. phmmer (BLAST-like single-sequence search) of the same 11,527 proteins against the 15,360
     single-hit reference: the top 300 hits per query with bit scores, so the best hit outside a
     hidden family can be read off.
  3. phmmer of the 34,138 dark queries against the 29,862-protein scorer reference: best hit.

Outputs go to results/baselines/ as compressed CSV.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline_sequence_search.py [--cpus 12]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)
PFAM35 = ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.hmm.gz"
ALPHA = Alphabet.amino()


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def load(ids: list[str]) -> DigitalSequenceBlock:
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    wanted = set(ids)
    seqs = {}
    for acc in samp.loc[list(wanted), "genome_accession"].unique():
        with SequenceFile(PROC / "panel_proteins" / f"{acc}_protein.faa", digital=True, alphabet=ALPHA) as sf:
            for s in sf:
                if dec(s.name) in wanted:
                    seqs[dec(s.name)] = s
    missing = wanted - set(seqs)
    assert not missing, f"{len(missing)} sequences not found"
    return DigitalSequenceBlock(ALPHA, [seqs[i] for i in ids])


def pfam_hits(block, cpus):
    rows, t0 = [], time.time()
    with pyhmmer.plan7.HMMFile(PFAM35) as hf:
        for n, top in enumerate(pyhmmer.hmmsearch(hf, block, E=10.0, cpus=cpus), 1):
            ga = top.query.cutoffs.gathering
            acc = dec(top.query.accession)
            for h in top:
                rows.append((dec(h.name), acc, round(h.score, 2), round(ga[0], 2) if ga else np.nan))
            if n % 4000 == 0:
                print(f"  pfam: {n} profiles, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows, columns=["protein_id", "pfam", "bits", "ga"])


def phmmer_hits(queries, targets, cpus, keep):
    rows, t0 = [], time.time()
    for n, top in enumerate(pyhmmer.hmmer.phmmer(queries, targets, cpus=cpus, E=10.0), 1):
        q = dec(top.query.name)
        kept = 0
        for h in top:
            t = dec(h.name)
            if t == q:
                continue
            rows.append((q, t, round(h.score, 2)))
            kept += 1
            if kept >= keep:
                break
        if n % 2000 == 0:
            print(f"  phmmer: {n}/{len(queries)} queries, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows, columns=["query", "target", "bits"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=12)
    args = ap.parse_args()

    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    ids_all = [str(i) for i in ref["ids"]]
    fam_all = np.array([str(f) for f in ref["families"]])
    keep = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy() >= 5
    eval_ids = [i for i, k in zip(ids_all, keep) if k]

    if not (OUT / "heldout_pfam35_hits.csv.gz").exists():
        print(f"Pfam 35.0 search for {len(eval_ids)} held-out-family proteins", flush=True)
        pfam_hits(load(eval_ids), args.cpus).to_csv(OUT / "heldout_pfam35_hits.csv.gz", index=False)

    if not (OUT / "heldout_phmmer_hits.csv.gz").exists():
        print("phmmer: held-out-family proteins against the 15,360 single-hit reference", flush=True)
        phmmer_hits(load(eval_ids), load(ids_all), args.cpus, 300).to_csv(OUT / "heldout_phmmer_hits.csv.gz", index=False)

    if not (OUT / "darkquery_phmmer_besthit.csv.gz").exists():
        samp = pd.read_csv(PROC / "embedding_sample.csv")
        q_ids = samp[samp.category.isin(["positive", "dark_negative"])].protein_id.tolist()
        r_ids = samp[samp.category == "characterised_at_t0"].protein_id.tolist()
        print(f"phmmer: {len(q_ids)} dark queries against {len(r_ids)} reference proteins", flush=True)
        phmmer_hits(load(q_ids), load(r_ids), args.cpus, 1).to_csv(OUT / "darkquery_phmmer_besthit.csv.gz", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
