#!/usr/bin/env python
"""Embed the rest of the benchmark sample at ESM-2 layers 22 and 33.

The layer-22 files hold the 15,360 single-hit references and 14,138 of the 34,138 dark queries.
Review II (comments 1 and 3) pointed out that the report mixed a layer-22 AUROC with a last-layer
Precision@K on a different query set and reference. This embeds the remaining proteins (14,502
multi-hit characterised references and 20,000 dark negatives) so that both layers can be scored on
the same 34,138 queries against the same 29,862 references.

fp32 weights with TF32 matmuls; sequences sorted by length for batching. Output:
data/processed/gtdb_R207/esm2_L22_L33_remaining.npz (ids, category, L22, L33).

    .venv/Scripts/python.exe scripts/embed_full_query_reference.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"


def main() -> None:
    import torch
    from Bio import SeqIO
    from darkmatter.device import inference_device
    from darkmatter.embeddings import load_embedder

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    have = set(str(i) for i in np.load(PROC / "esm2_reembed_L33_22.npz", allow_pickle=True)["ids"])
    have |= set(str(i) for i in np.load(PROC / "esm2_reembed_queries_L33_22.npz", allow_pickle=True)["ids"])
    todo = samp[~samp.protein_id.isin(have)].reset_index(drop=True)
    print(f"{len(todo)} proteins to embed: {todo.category.value_counts().to_dict()}", flush=True)

    wanted = set(todo.protein_id); seqs = {}
    for acc in todo.genome_accession.unique():
        for rec in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
            if rec.id in wanted:
                seqs[rec.id] = str(rec.seq)
    ids = todo.protein_id.tolist()
    assert all(i in seqs for i in ids)
    lengths = np.array([len(seqs[i]) for i in ids]); order = np.argsort(lengths, kind="stable")

    embedder = load_embedder("esm2")
    embedder.model = embedder.model.float()
    model, tok = embedder.model, embedder.tokenizer
    device = inference_device(embedder.plan, model)
    out = {22: np.zeros((len(ids), 1280), np.float32), 33: np.zeros((len(ids), 1280), np.float32)}
    t0 = time.time(); i = 0; nb = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * lengths[order[j]] <= 3000 and j - i < 64:
            j += 1
        j = max(j, i + 1); rows = order[i:j]
        toks = tok([seqs[ids[r]] for r in rows], padding=True, truncation=True,
                   max_length=embedder.max_tokens, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            hs = model(**toks, output_hidden_states=True).hidden_states
        mask = toks["attention_mask"].unsqueeze(-1)
        for li in (22, 33):
            m = mask.to(hs[li].dtype)
            out[li][rows] = ((hs[li] * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
        del hs; i = j; nb += 1
        if nb % 50 == 0:
            print(f"  {i}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(PROC / "esm2_L22_L33_remaining.npz", ids=np.array(ids), category=todo.category.to_numpy().astype(str),
             L22=out[22], L33=out[33])
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
