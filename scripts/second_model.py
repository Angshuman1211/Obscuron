#!/usr/bin/env python
"""Does the method hold with a second, unrelated embedding model? (TCBB revision)

Repeats the main analyses with ProtT5-XL-U50 (Rostlab; T5 encoder trained on UniRef50 after BFD),
which shares neither architecture, training data nor developers with ESM-2:
  * withheld-family AUROC for all 903 families (kNN-5, nearest neighbour, density-normalised), with
    same-clan hard negatives (scripts/revision_scoring.py: heldout_variants);
  * the rank average with phmmer (heldout_combined);
  * calibration under random, genome-disjoint and phylum-disjoint splits (lineage_calibration);
  * the twilight-zone comparison with phmmer (scripts/twilight_zone.py: analyse).

Weights: the safetensors conversion of Rostlab/prot_t5_xl_half_uniref50-enc published by Hugging
Face's conversion bot (pull request #1, commit d53b8ce), downloaded to .cache/; torch < 2.6 cannot
load the .bin file safely (CVE-2025-32434). Per-protein embedding: mean over residue positions of
the last encoder layer, sequences truncated at 1,022 residues as for ESM-2, rare residues (U, Z, O,
B) mapped to X as in the ProtT5 instructions.

Embedding is checkpointed every 500 proteins and resumes after an interruption.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/second_model.py
Writes data/processed/gtdb_R207/prott5_heldout.npz and results/revision/second_model.json.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.scoring.knn import l2_normalize  # noqa: E402
from revision_scoring import PROC, heldout_combined, heldout_variants, lineage_calibration  # noqa: E402
import twilight_zone  # noqa: E402

MODEL_DIR = ROOT / ".cache" / "prot_t5_xl_half_enc_d53b8ce"
EMB = PROC / "prott5_heldout.npz"
PARTIAL = ROOT / ".cache" / "prott5_heldout_partial.npz"
MAX_RES = 1022
TOKENS_PER_BATCH = 6000


def sequences(pids) -> dict[str, str]:
    from Bio import SeqIO
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    want, out = set(pids), {}
    for acc in samp.loc[sorted(want), "genome_accession"].unique():
        for r in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
            if r.id in want:
                out[r.id] = str(r.seq).rstrip("*")
    return out


def embed(ids) -> np.ndarray:
    if EMB.exists():
        z = np.load(EMB)
        assert [str(i) for i in z["ids"]] == list(ids)
        return z["emb"].astype(np.float32)
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    seqs = sequences(ids)
    done = {}
    if PARTIAL.exists():
        z = np.load(PARTIAL)
        done = dict(zip([str(i) for i in z["ids"]], z["emb"]))
        print(f"resuming: {len(done)} proteins already embedded", flush=True)
    tok = T5Tokenizer.from_pretrained(MODEL_DIR, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, use_safetensors=True).cuda().eval()
    todo = sorted((p for p in ids if p not in done), key=lambda p: len(seqs[p]))
    t0, since_save = time.time(), 0

    def save():
        tmp = PARTIAL.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(list(done)), emb=np.stack(list(done.values())))
        os.replace(tmp, PARTIAL)

    i = 0
    while i < len(todo):
        L = min(len(seqs[todo[i]]), MAX_RES)
        j = i
        while j < len(todo) and (j - i + 1) * min(len(seqs[todo[j]]), MAX_RES) <= TOKENS_PER_BATCH:
            j += 1
        batch = todo[i:max(j, i + 1)]
        text = [" ".join(seqs[p][:MAX_RES].translate(str.maketrans("UZOB", "XXXX"))) for p in batch]
        enc = tok(text, add_special_tokens=True, padding="longest", return_tensors="pt")
        with torch.inference_mode():
            h = model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).last_hidden_state
        for k, p in enumerate(batch):
            n = min(len(seqs[p]), MAX_RES)  # residue positions only; the final </s> is left out
            done[p] = h[k, :n].float().mean(0).cpu().numpy().astype(np.float16)
        since_save += len(batch)
        i += len(batch)
        if since_save >= 500 or i >= len(todo):
            save()
            since_save = 0
            print(f"  {len(done)}/{len(ids)} embedded, {time.time() - t0:.0f} s (length {L})", flush=True)
    emb = np.stack([done[p] for p in ids])
    np.savez_compressed(EMB, ids=np.array(ids), emb=emb, model="Rostlab/prot_t5_xl_half_uniref50-enc@d53b8ce")
    return emb.astype(np.float32)


def main():
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    ids_all = np.array([str(i) for i in ref["ids"]]); fam_all = np.array([str(f) for f in ref["families"]])
    keep = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy() >= 5
    ids, fams = ids_all[keep], fam_all[keep]
    X = l2_normalize(embed(list(ids)).astype(np.float64))
    # ProtT5 embeddings share a large common component; centring on the reference mean, as done for
    # ProstT5 in the supplementary, makes cosine distance informative
    Xc = l2_normalize(X - X.mean(0))
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    genomes = samp.genome_accession.reindex(ids).to_numpy()
    phylum = pd.read_csv(PROC / "genome_panel.csv").set_index("accession").phylum
    phyla = pd.Series(genomes).map(phylum).fillna("unknown").to_numpy()
    clan_of = {}
    with gzip.open(ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.clans.tsv.gz", "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[1]:
                clan_of[p[0]] = p[1]
    res = {"model": "Rostlab/prot_t5_xl_half_uniref50-enc (safetensors, refs/pr/1 d53b8ce)", "n_proteins": int(len(ids))}
    for name, M in (("raw", X), ("centred", Xc)):
        print(f"== {name}", flush=True)
        r = {}
        heldout_variants(M, fams, clan_of, r)
        heldout_combined(M, fams, list(ids), r)
        r["twilight_zone"] = {k: v for k, v in twilight_zone.analyse(M, ids, fams).items() if k in ("auroc", "detection_by_best_hit")}
        res[name] = r
    lineage_calibration(Xc, fams, genomes, phyla, res)  # calibration for the centred embedding
    out = ROOT / "results" / "revision" / "second_model.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print("written", out)


if __name__ == "__main__":
    main()
