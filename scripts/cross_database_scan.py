#!/usr/bin/env python
"""Scan the 34,138 dark queries against three further HMM libraries (TCBB revision).

  antifam  AntiFam, profiles of known spurious ORFs; curated gathering thresholds
  ncbifam  NCBIfam / TIGRFAM (current release); trusted cutoffs where defined, else E <= 1e-5
  nmpfams  NMPFamsDB novel metagenomic families (Baltoumas et al. 2023); no curated cut-offs,
           so a sequence E-value <= 1e-5 is used

The libraries are current releases, so these scans describe how the proteins look today, not at
T0; they are not used to relabel the benchmark. Fetch the libraries first with
scripts/fetch_external_hmms.py.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/cross_database_scan.py [--cpus 12] [--only antifam,ncbifam]
Writes results/revision/cross_database_hits.csv.gz (protein_id, library, best hit, bits, evalue).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
EXT = ROOT / "data" / "raw" / "external"
OUT = ROOT / "results" / "revision"
ALPHA = Alphabet.amino()
EVALUE = 1e-5


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def queries() -> DigitalSequenceBlock:
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    q = samp[samp.category.isin(["positive", "dark_negative"])]
    wanted = set(q.protein_id); seqs = []
    for acc in q.genome_accession.unique():
        with SequenceFile(PROC / "panel_proteins" / f"{acc}_protein.faa", digital=True, alphabet=ALPHA) as sf:
            seqs.extend(s for s in sf if dec(s.name) in wanted)
    return DigitalSequenceBlock(ALPHA, seqs)


def hmm_files(lib: str) -> list[Path]:
    root = EXT / lib / "unpacked"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".hmm", ".lib"))
    if not files:  # some archives hold one profile per extension-less file
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.stat().st_size > 0 and p.read_bytes()[:6] == b"HMMER3")
    return files


def scan(lib: str, block, cpus: int) -> pd.DataFrame:
    best: dict[str, tuple] = {}
    t0 = time.time(); n = 0
    for f in hmm_files(lib):
        with pyhmmer.plan7.HMMFile(f) as hf:
            for top in pyhmmer.hmmsearch(hf, block, E=EVALUE, cpus=cpus):
                hmm = top.query
                cut = None
                if lib == "antifam" and hmm.cutoffs.gathering_available():
                    cut = hmm.cutoffs.gathering[0]
                elif lib == "ncbifam" and hmm.cutoffs.trusted_available():
                    cut = hmm.cutoffs.trusted[0]
                name = dec(hmm.accession or hmm.name)
                for h in top:
                    if cut is not None and h.score < cut:
                        continue
                    if cut is None and h.evalue > EVALUE:
                        continue
                    pid = dec(h.name)
                    if pid not in best or h.score > best[pid][1]:
                        best[pid] = (name, float(h.score), float(h.evalue))
                n += 1
                if n % 5000 == 0:
                    print(f"  {lib}: {n} profiles, {time.time() - t0:.0f}s", flush=True)
    print(f"{lib}: {n} profiles, {len(best)} proteins hit, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame([(p, lib, *v) for p, v in best.items()], columns=["protein_id", "library", "hit", "bits", "evalue"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--only", default="antifam,ncbifam,nmpfams")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cross_database_hits.csv.gz"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["protein_id", "library", "hit", "bits", "evalue"])
    block = queries()
    print(f"{len(block)} dark queries", flush=True)
    for lib in args.only.split(","):
        if lib in set(done.library) or not hmm_files(lib):
            print(f"skip {lib} ({'done' if lib in set(done.library) else 'not downloaded'})", flush=True)
            continue
        done = pd.concat([done, scan(lib, block, args.cpus)], ignore_index=True)
        done.to_csv(path, index=False)
    print("written", path)


if __name__ == "__main__":
    main()
