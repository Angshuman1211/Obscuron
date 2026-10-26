#!/usr/bin/env python
"""Independent test set for the TCBB evaluation: genomes that did not exist in GTDB R207.

Selection (frozen before any scoring): GTDB R232 species representatives whose genome accession
is absent from R207 and whose species name does not occur in R207 at all, with CheckM2
completeness >= 90 and contamination <= 5. Up to two genomes per phylum are drawn at random
(seed 42) until 60 genomes are chosen. None of these genomes or species is in the 502-genome panel.

For each genome: the nucleotide assembly is downloaded from the NCBI Datasets API, genes are
called with pyrodigal (the Prodigal algorithm GTDB uses, single-genome training mode), and the
proteins are scanned against Pfam 35.0 and 37.0 at the gathering threshold with the same rules as
the main benchmark (dark at T0 = no Pfam 35.0 hit; positive = dark and a hit to a Pfam 37.0
family absent from Pfam 35.0).

Outputs: data/processed/independent_R232/{genomes.csv, protein_labels.csv, proteins/*.faa}

    PYTHONPATH=src .venv/Scripts/python.exe scripts/build_independent_set.py [--cpus 12]
"""
from __future__ import annotations

import argparse
import io
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyrodigal
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "independent_R232"
PROT = OUT / "proteins"
N_GENOMES = 60
API = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download?include_annotation_type=GENOME_FASTA"


def species(tax: str) -> str:
    return tax.split(";s__")[-1] if ";s__" in tax else ""


def read_r207() -> pd.DataFrame:
    frames = []
    for dom in ("bac120", "ar53"):
        with tarfile.open(RAW / "gtdb_R207" / f"{dom}_metadata_r207.tar.gz") as t:
            member = [m for m in t.getmembers() if m.name.endswith(".tsv")][0]
            frames.append(pd.read_csv(t.extractfile(member), sep="\t", usecols=["accession", "gtdb_taxonomy"], low_memory=False))
    return pd.concat(frames)


def select() -> pd.DataFrame:
    cols = ["accession", "gtdb_representative", "gtdb_taxonomy", "checkm2_completeness", "checkm2_contamination",
            "ncbi_genbank_assembly_accession", "genome_size"]
    r232 = pd.concat([pd.read_csv(RAW / "gtdb_R232" / f"{d}_metadata_r232.tsv.gz", sep="\t", usecols=cols, low_memory=False)
                      for d in ("bac120", "ar53")])
    r207 = read_r207()
    old_acc = set(r207.accession.str.split("_", n=1).str[1].str.split(".").str[0])
    old_sp = set(r207.gtdb_taxonomy.map(species))
    panel = set(pd.read_csv(ROOT / "data" / "processed" / "gtdb_R207" / "genome_panel.csv").accession)
    c = r232[r232.gtdb_representative.astype(str).str.lower().isin(["t", "true"])].copy()
    c["raw"] = c.accession.str.split("_", n=1).str[1]
    c["species"] = c.gtdb_taxonomy.map(species)
    c["phylum"] = c.gtdb_taxonomy.str.extract(r"p__([^;]+)")[0]
    c = c[~c.raw.str.split(".").str[0].isin(old_acc) & ~c.species.isin(old_sp) & ~c.accession.isin(panel)]
    c = c[(c.checkm2_completeness >= 90) & (c.checkm2_contamination <= 5)]
    rng = np.random.default_rng(42)
    picked = []
    for _, g in c.groupby("phylum"):
        picked.append(g.sample(min(2, len(g)), random_state=int(rng.integers(1 << 30))))
    pool = pd.concat(picked)
    sel = pool.sample(min(N_GENOMES, len(pool)), random_state=42).sort_values("accession")
    print(f"candidates {len(c)} in {c.phylum.nunique()} phyla; selected {len(sel)} from {sel.phylum.nunique()} phyla", flush=True)
    return sel


def fetch_and_call(acc_raw: str, out_faa: Path) -> int:
    r = requests.get(API.format(acc=acc_raw), timeout=(10, 300))
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fna = [n for n in z.namelist() if n.endswith((".fna", ".fa", ".fasta"))]
    if not fna:
        raise IOError(f"no nucleotide FASTA for {acc_raw}")
    contigs, name, buf = [], None, []
    for line in io.TextIOWrapper(z.open(fna[0]), encoding="ascii"):
        if line.startswith(">"):
            if name:
                contigs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        contigs.append((name, "".join(buf)))
    finder = pyrodigal.GeneFinder()
    finder.train(*(s.encode() for _, s in contigs))
    n = 0
    with open(out_faa, "w") as fh:
        for cname, seq in contigs:
            for i, gene in enumerate(finder.find_genes(seq.encode()), 1):
                prot = gene.translate().rstrip("*")
                if len(prot) >= 30:
                    fh.write(f">{cname}_{i} # {gene.begin} # {gene.end} # {gene.strand}\n{prot}\n")
                    n += 1
    return n


def label(cpus: int) -> pd.DataFrame:
    from pyhmmer.easel import Alphabet
    from darkmatter.data.hmmscan import load_protein_sequences_multi, scan_against_pfam
    from darkmatter.data.pfam_diff import label_protein, new_families_since

    faas = sorted(PROT.glob("*.faa"))
    block = load_protein_sequences_multi(faas, Alphabet.amino())
    print(f"scanning {len(block)} proteins against Pfam 35.0 and 37.0", flush=True)
    t0 = time.time()
    h35 = scan_against_pfam(block, RAW / "pfam_35.0" / "Pfam-A.hmm.gz", cpus=cpus)
    print(f"  Pfam 35.0 done in {time.time() - t0:.0f}s", flush=True)
    h37 = scan_against_pfam(block, RAW / "pfam_37.0" / "Pfam-A.hmm", cpus=cpus)
    print(f"  Pfam 37.0 done in {time.time() - t0:.0f}s", flush=True)
    new = new_families_since(RAW / "pfam_35.0" / "Pfam-A.clans.tsv.gz", RAW / "pfam_37.0" / "Pfam-A.clans.tsv.gz")
    gen_of = {}
    for f in faas:
        for line in open(f):
            if line.startswith(">"):
                gen_of[line[1:].split()[0]] = f.stem
    rows = []
    for pid in h35:
        lab = label_protein(h35[pid], h37.get(pid, []), new)
        rows.append({"genome_accession": gen_of[pid], "protein_id": pid, **lab,
                     "pfam35_families": ";".join(sorted(set(a.split(".")[0] for a in h35[pid])))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=12)
    args = ap.parse_args()
    PROT.mkdir(parents=True, exist_ok=True)

    gpath = OUT / "genomes.csv"
    if gpath.exists():
        sel = pd.read_csv(gpath)
    else:
        sel = select()
        sel.to_csv(gpath, index=False)

    counts = {}
    for _, g in sel.iterrows():
        faa = PROT / f"{g.accession}.faa"
        if faa.exists():
            continue
        for attempt in range(3):
            try:
                counts[g.accession] = fetch_and_call(g.raw, faa)
                print(f"  {g.accession}: {counts[g.accession]} proteins", flush=True)
                break
            except Exception as e:  # network hiccups: retry, then skip the genome
                print(f"  {g.accession}: attempt {attempt + 1} failed ({type(e).__name__})", flush=True)
                time.sleep(5 * (attempt + 1))
        time.sleep(0.4)  # stay under the anonymous NCBI rate limit

    lpath = OUT / "protein_labels.csv"
    if not lpath.exists():
        lab = label(args.cpus)
        lab.to_csv(lpath, index=False)
        print(f"proteins {len(lab)}; dark {lab.dark_at_t0.mean():.3%}; positives {lab.positive_proxy.sum()} "
              f"({lab.positive_proxy.mean():.3%})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
