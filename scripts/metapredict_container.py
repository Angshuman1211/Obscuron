"""Runs inside the python:3.11 container started by scripts/metapredict_check.py (metapredict has no
Windows build). Reads a FASTA file, writes one row per protein: the share of residues with a
metapredict disorder score above 0.5 and the mean score.

    python metapredict_container.py in.fasta out.csv
"""
import csv
import sys

import metapredict as meta
import numpy as np


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line
    return seqs


def main():
    seqs = read_fasta(sys.argv[1])
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "metapredict_disordered_share", "metapredict_mean", "metapredict_version"])
        for n, (pid, s) in enumerate(seqs.items(), 1):
            sc = np.asarray(meta.predict_disorder(s), dtype=float)
            w.writerow([pid, float((sc > 0.5).mean()), float(sc.mean()), meta.__version__])
            if n % 500 == 0:
                print(f"{n}/{len(seqs)}", flush=True)


if __name__ == "__main__":
    main()
