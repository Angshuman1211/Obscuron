#!/usr/bin/env python
"""Layer 2 driver (Phase-4): structure-aware embedding via ProstT5.

Embeds the characterised reference with ProstT5's structure-informed encoder and
runs the SAME held-out-family AUROC protocol (P2-D4) as the ESM-2 and Genos-m
arms, on the SAME 300 largest families / 6,907 proteins used for the matched
comparison in P2-D11 — so the structural number is directly comparable to
ESM-2 L22 0.988 and Genos-m L9 0.739.

Optionally (--score-queries) also embeds the dark queries and emits a structural
novelty axis (kNN distance to the ProstT5 reference) for the Layer-4 convergence.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/run_layer2_prostt5.py
    HF_HUB_DISABLE_XET=1 uv run python scripts/run_layer2_prostt5.py --score-queries
    uv run python scripts/run_layer2_prostt5.py --eval-only            # from saved npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
REF_FAMS = PROC / "reference_families.csv"
SAMPLE = PROC / "embedding_sample.csv"
AA_DIR = PROC / "panel_proteins"
NPZ = PROC / "prostt5_reference.npz"
QNPZ = PROC / "prostt5_query_novelty.csv"
MAX_FAMILIES = 300
MIN_MEMBERS = 5
K = 5


# --- held-out-family AUROC (copied minimal, identical to reembed_eval for comparability) ---
def _auroc(y, s):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return np.nan if npos == 0 or nneg == 0 else (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def _knn(Q, R, k, self_offset=None, chunk=2048):
    out = []
    for a in range(0, len(Q), chunk):
        d = 1.0 - Q[a:a + chunk] @ R.T
        if self_offset is not None:
            rows = np.arange(d.shape[0]); d[rows, self_offset + a + rows] = np.inf
        d.sort(axis=1)
        kk = min(k, d.shape[1] - (1 if self_offset is not None else 0))
        out.append(d[:, :kk].mean(1))
    return np.concatenate(out)


def _heldout_auroc(vectors, families, center):
    fams = np.array(families)
    keep = pd.Series(fams).groupby(fams).transform("size").values >= MIN_MEMBERS
    v, f = vectors[keep], fams[keep]
    scores = []
    for F in np.unique(f):
        is_f = f == F
        ref_raw = v[~is_f]
        mu = ref_raw.mean(0, keepdims=True) if center else 0.0
        norm = lambda x: (x - mu) / np.clip(np.linalg.norm(x - mu, axis=1, keepdims=True), 1e-12, None)
        ref = norm(ref_raw)
        a = _auroc(np.r_[np.ones(int(is_f.sum())), np.zeros(len(ref))],
                   np.r_[_knn(norm(v[is_f]), ref, K), _knn(ref, ref, K, self_offset=0)])
        if not np.isnan(a):
            scores.append(a)
    return float(np.mean(scores)), float(np.median(scores)), len(scores)


def _load_seqs(ids_by_genome: dict[str, set]) -> dict[str, str]:
    seqs = {}
    for acc, wanted in ids_by_genome.items():
        p = AA_DIR / f"{acc}_protein.faa"
        if p.exists():
            for rec in SeqIO.parse(p, "fasta"):
                if rec.id in wanted:
                    seqs[rec.id] = str(rec.seq)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--score-queries", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-families", type=int, default=MAX_FAMILIES,
                    help="largest N families to embed; ProstT5 on MPS is slow, so bound it")
    args = ap.parse_args()

    fam = pd.read_csv(REF_FAMS)
    top = fam["family"].value_counts().head(args.max_families).index
    ref = fam[fam["family"].isin(top)].reset_index(drop=True)
    # attach genome accession from embedding_sample
    gmap = pd.read_csv(SAMPLE)[["protein_id", "genome_accession"]]
    ref = ref.merge(gmap, on="protein_id", how="left").dropna(subset=["genome_accession"])
    print(f"{len(ref)} reference proteins across {ref['family'].nunique()} families (top {MAX_FAMILIES})", flush=True)

    embedder = None
    def get_embedder():
        nonlocal embedder
        if embedder is None:
            from darkmatter.embeddings import load_embedder
            embedder = load_embedder("prostt5")
        return embedder

    if args.eval_only:
        d = np.load(NPZ)
        vecs = d["emb"]; families = [str(x) for x in d["families"]]
    else:
        emb = get_embedder()
        by_genome: dict[str, set] = {}
        for _, r in ref.iterrows():
            by_genome.setdefault(r["genome_accession"], set()).add(r["protein_id"])
        seqs = _load_seqs(by_genome)
        ref = ref[ref["protein_id"].isin(seqs)].reset_index(drop=True)
        # length-sort so each batch is length-homogeneous -> minimal padding -> big MPS speedup
        ref["_len"] = ref["protein_id"].map(lambda i: len(seqs[i]))
        ref = ref.sort_values("_len").reset_index(drop=True)
        ids = ref["protein_id"].tolist()
        families = ref["family"].tolist()
        print(f"embedding {len(ids)} reference proteins with ProstT5 (length-sorted)...", flush=True)
        t0 = time.time()
        chunks = []
        CH = 256
        for a in range(0, len(ids), CH):
            part = [seqs[i] for i in ids[a:a + CH]]
            chunks.append(emb.embed(part, batch_size=args.batch_size))
            done = min(a + CH, len(ids)); el = time.time() - t0
            print(f"  {done}/{len(ids)} ({el:.0f}s, {done/el*60:.0f} seqs/min)", flush=True)
        vecs = np.concatenate(chunks)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(NPZ, emb=vecs, ids=np.array(ids), families=np.array(families))

    print("\nProstT5 held-out-family AUROC (300 families, same eval as P2-D11):")
    for center in (False, True):
        m, med, nf = _heldout_auroc(vecs, families, center)
        print(f"  {'centered' if center else 'raw':>8}: mean {m:.4f}  median {med:.4f}  ({nf} families)")
    print("  comparison (same 300-family eval): ESM-2 L22 0.988 / Genos-m L9 0.739 (P2-D11)")

    if args.score_queries:
        emb = get_embedder()
        sample = pd.read_csv(SAMPLE)
        qry = sample[sample["category"].isin(["dark_negative", "positive"])].reset_index(drop=True)
        by_genome = {}
        for _, r in qry.iterrows():
            by_genome.setdefault(r["genome_accession"], set()).add(r["protein_id"])
        qseqs = _load_seqs(by_genome)
        qry = qry[qry["protein_id"].isin(qseqs)].reset_index(drop=True)
        # length-sort so each batch is length-homogeneous -> minimal padding,
        # the same fix that made the reference embed tractable; the earlier,
        # unsorted version of this loop took 5+ hours on 34,138 queries with
        # zero progress visibility because random-length batches each pay
        # the padding cost of their longest member, scattered throughout
        # rather than only at the end.
        qry["_len"] = qry["protein_id"].map(lambda i: len(qseqs[i]))
        qry = qry.sort_values("_len").reset_index(drop=True)
        qids = qry["protein_id"].tolist()
        print(f"\nembedding {len(qids)} dark queries with ProstT5 (structural novelty axis, length-sorted)...", flush=True)
        t0 = time.time()
        CH = 256
        # Per-chunk checkpointing: two prior attempts crashed with CUDA OOM at
        # ~96.7% through (the longest, length-sorted-to-the-end sequences) and
        # lost ALL progress since chunks only lived in memory. Now every chunk
        # is saved to disk as it completes, so a crash only costs one chunk --
        # a restart skips every chunk whose checkpoint file already exists.
        ckpt_dir = PROC / "_prostt5_query_chunks"
        ckpt_dir.mkdir(exist_ok=True)
        n_chunks = (len(qids) + CH - 1) // CH
        for ci, a in enumerate(range(0, len(qids), CH)):
            ckpt = ckpt_dir / f"chunk_{ci:05d}.npy"
            done = min(a + CH, len(qids)); el = time.time() - t0
            if ckpt.exists():
                print(f"  {done}/{len(qids)} chunk {ci+1}/{n_chunks} (cached)", flush=True)
                continue
            part = [qseqs[i] for i in qids[a:a + CH]]
            bs = args.batch_size
            for attempt in range(4):
                try:
                    vec = emb.embed(part, batch_size=bs)
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or attempt == 3:
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    print(f"    OOM on chunk {ci+1}, retrying at batch_size={bs}", flush=True)
            np.save(ckpt, vec)
            print(f"  {done}/{len(qids)} chunk {ci+1}/{n_chunks} ({el:.0f}s, {done/max(el,1e-9)*60:.0f} seqs/min)", flush=True)
        qvec = np.concatenate([np.load(ckpt_dir / f"chunk_{ci:05d}.npy") for ci in range(n_chunks)])
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        R = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        Q = qvec / np.clip(np.linalg.norm(qvec, axis=1, keepdims=True), 1e-12, None)
        nov = _knn(Q, R, K)
        pd.DataFrame({"protein_id": qids, "structural_novelty": nov}).to_csv(QNPZ, index=False)
        print(f"wrote {QNPZ}  (feed to run_convergence.py --extra-axis)")


if __name__ == "__main__":
    main()
