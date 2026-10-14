#!/usr/bin/env python
"""Embed the held-out-family evaluation set at EVERY ESM-2 layer (mean-pooled).

The headline uses layer 22, which was picked on a 100-sequence / 20-family
diagnostic (P2-D7/P2-D8), not on the full benchmark. This sweep lets
scripts/layer_selection_validation.py repeat the choice honestly: pick the layer
on a development split of families, report it on families it never saw.

Embeds the 11,527 single-hit reference proteins that belong to a family with at
least 5 members (the proteins heldout_auroc scores). fp32, same pooling as
reembed_eval.embed_layers. Writes data/processed/gtdb_R207/esm2_layer_sweep/
(L00.npy ... L33.npy float32, ids.npy, families.npy). Layers 22 and 33 are checked
against esm2_reembed_L33_22.npz as a consistency test.

    .venv/Scripts/python.exe scripts/layer_sweep_esm2.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = PROC / "esm2_layer_sweep"
N_LAYERS = 34  # embeddings + 33 transformer layers of esm2_t33_650M


def main() -> None:
    import torch
    from Bio import SeqIO
    from darkmatter.device import inference_device
    from darkmatter.embeddings import load_embedder

    OUT.mkdir(exist_ok=True)
    # TF32 matmuls keep the fp32 exponent range (fp16 overflowed ESM-2's middle layers) and run
    # several times faster on this card; the check against the saved fp32 embeddings is printed at the end
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ref = np.load(PROC / "esm2_reembed_L33_22.npz", allow_pickle=True)
    ids_all = np.array([str(x) for x in ref["ids"]])
    fam_all = np.array([str(x) for x in ref["families"]])
    size = pd.Series(fam_all).groupby(fam_all).transform("size").to_numpy()
    keep = size >= 5
    ids, fams = ids_all[keep], fam_all[keep]

    sample = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    wanted = set(ids)
    seqs: dict[str, str] = {}
    for acc in sample.loc[ids, "genome_accession"].unique():
        p = PROC / "panel_proteins" / f"{acc}_protein.faa"
        for rec in SeqIO.parse(p, "fasta"):
            if rec.id in wanted:
                seqs[rec.id] = str(rec.seq)
    assert len(seqs) == len(ids), (len(seqs), len(ids))
    print(f"{len(ids)} sequences, {len(set(fams))} families", flush=True)

    embedder = load_embedder("esm2")
    embedder.model = embedder.model.float()
    model, tok = embedder.model, embedder.tokenizer
    device = inference_device(embedder.plan, model)

    lengths = np.array([len(seqs[i]) for i in ids])
    order = np.argsort(lengths, kind="stable")  # similar lengths per batch, less padding
    acc = np.zeros((N_LAYERS, len(ids), 1280), dtype=np.float32)
    t0 = time.time()
    bs_tokens = 3000  # tokens per batch, keeps the 8 GB card comfortable
    i = 0
    n_batches = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * lengths[order[j]] <= bs_tokens and j - i < 64:
            j += 1
        j = max(j, i + 1)
        rows = order[i:j]
        toks = tok([seqs[ids[r]] for r in rows], padding=True, truncation=True,
                   max_length=embedder.max_tokens, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            hs = model(**toks, output_hidden_states=True).hidden_states
        mask = toks["attention_mask"].unsqueeze(-1)
        for li in range(N_LAYERS):
            m = mask.to(hs[li].dtype)
            pooled = (hs[li] * m).sum(1) / m.sum(1).clamp(min=1)
            acc[li, rows] = pooled.float().cpu().numpy()
        del hs
        i = j
        n_batches += 1
        if n_batches % 10 == 0:
            print(f"  {i}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"embedded in {time.time() - t0:.0f}s", flush=True)

    for li in range(N_LAYERS):
        np.save(OUT / f"L{li:02d}.npy", acc[li])
    np.save(OUT / "ids.npy", ids)
    np.save(OUT / "families.npy", fams)

    for li in (22, 33):
        old = ref[f"L{li}"][keep]
        a, b = acc[li], old
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"layer {li} vs saved embedding: min cosine {cos.min():.6f}, mean {cos.mean():.6f}", flush=True)


if __name__ == "__main__":
    main()
