#!/usr/bin/env python
"""Embed the dark-query set with Genos-m (nucleotide) at the validated layer
(9, P2-D11) and score each query's novelty vs the Genos-m reference -- the
Genos-m 4th convergence axis P4-D1/P4-D4 wired via `--extra-axis` but never
built (needs a GPU that can load Genos-m; the RTX 4060 OOMs on it, P1-D1).

Reference: pass --reference-npz pointing at the *_reembed_L{layers}.npz that
scripts/reembed_eval.py --model genos-m produces (run that first). Novelty =
mean cosine distance to the k=5 nearest reference points (matches Layer-1's
own kNN geometry, P2-D2), so the axis is on the same footing as the other
three (evt, context, composition).

Checkpointed every --chunk-size sequences (P1-D10 pattern) -- this runs on a
billed cloud GPU, an interruption should not mean starting over.

    uv run python scripts/embed_genos_m_dark_queries.py \
        --reference-npz data/processed/gtdb_R207/genos-m_reembed_L12_9.npz \
        --layer 9 --n-dark-negative 10000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
SEQ_DIR, SUFFIX = "panel_nucleotides", "_protein.fna"
K = 5  # matches config/scorer.yaml's headline k (P2-D5)


def _load_checkpoint(ckpt_path: Path) -> tuple[list[np.ndarray], list[str]]:
    if not ckpt_path.exists():
        return [], []
    d = np.load(ckpt_path)
    done_ids = [str(x) for x in d["ids"]]
    print(f"resuming from checkpoint: {len(done_ids)} sequences already embedded", flush=True)
    return [d["emb"]], done_ids


def _save_checkpoint(ckpt_path: Path, acc: list[np.ndarray], done_ids: list[str]) -> None:
    tmp = ckpt_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, emb=np.concatenate(acc), ids=np.array(done_ids))
    tmp.replace(ckpt_path)


def embed_checkpointed(ids: list[str], seq_by_id: dict[str, str], layer: int,
                       batch_size: int, ckpt_path: Path, chunk_size: int) -> np.ndarray:
    from darkmatter.embeddings import load_embedder

    acc, done_ids = _load_checkpoint(ckpt_path)
    done_set = set(done_ids)
    remaining = [s for s in ids if s not in done_set]
    if not remaining:
        print("all sequences already embedded (checkpoint complete)", flush=True)
        full = np.concatenate(acc)
        pos = {s: k for k, s in enumerate(done_ids)}
        return full[[pos[s] for s in ids]]

    embedder = load_embedder("genos-m")
    embedder.layer = layer

    t0 = time.time()
    since_ckpt = 0
    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i:i + batch_size]
        seqs = [seq_by_id[s] for s in chunk]
        emb = embedder.embed(seqs, batch_size=len(seqs))
        acc.append(emb)
        done_ids.extend(chunk)
        since_ckpt += len(chunk)
        n_done = len(done_ids)
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {n_done}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
        if since_ckpt >= chunk_size:
            _save_checkpoint(ckpt_path, acc, done_ids)
            since_ckpt = 0
            print(f"  checkpoint saved at {n_done}/{len(ids)}", flush=True)
    _save_checkpoint(ckpt_path, acc, done_ids)
    print(f"embedded {len(remaining)} new seqs at layer {layer} in {time.time()-t0:.0f}s "
          f"({len(done_ids)}/{len(ids)} total)", flush=True)
    full = np.concatenate(acc)
    pos = {s: k for k, s in enumerate(done_ids)}
    return full[[pos[s] for s in ids]]


def knn_novelty(query: np.ndarray, reference: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    qn = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    rn = reference / np.clip(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12, None)
    out = np.empty(len(qn))
    for a in range(0, len(qn), chunk):
        d = 1.0 - qn[a:a + chunk] @ rn.T
        d.sort(axis=1)
        out[a:a + chunk] = d[:, :k].mean(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-npz", required=True, help="output of reembed_eval.py --model genos-m")
    ap.add_argument("--layer", type=int, default=9, help="Genos-m layer (9 = P2-D11's best)")
    ap.add_argument("--n-positive", type=int, default=0, help="subsample positives; 0 = all")
    ap.add_argument("--n-dark-negative", type=int, default=0, help="subsample dark_negative; 0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=500)
    args = ap.parse_args()

    ckpt_path = PROC / f"genos-m_dark_queries_L{args.layer}.ckpt.npz"
    out_csv = ROOT / "results" / "genos-m_novelty_scores.csv"

    sample = pd.read_csv(PROC / "embedding_sample.csv")
    pos = sample[sample["category"] == "positive"]
    neg = sample[sample["category"] == "dark_negative"]
    if args.n_positive:
        pos = pos.sample(n=min(args.n_positive, len(pos)), random_state=args.seed)
    if args.n_dark_negative:
        neg = neg.sample(n=min(args.n_dark_negative, len(neg)), random_state=args.seed)
    queries = pd.concat([pos, neg]).reset_index(drop=True)
    print(f"{len(queries)} dark queries ({(queries['category']=='positive').sum()} positive)", flush=True)

    from Bio import SeqIO
    wanted = set(queries["protein_id"])
    seq_by_id: dict[str, str] = {}
    for acc in queries["genome_accession"].unique():
        p = PROC / SEQ_DIR / f"{acc}{SUFFIX}"
        if p.exists():
            for rec in SeqIO.parse(p, "fasta"):
                if rec.id in wanted:
                    seq_by_id[rec.id] = str(rec.seq)
    queries = queries[queries["protein_id"].isin(seq_by_id)].reset_index(drop=True)
    ids = queries["protein_id"].tolist()
    print(f"{len(ids)} sequences loaded", flush=True)

    emb = embed_checkpointed(ids, seq_by_id, args.layer, args.batch_size, ckpt_path, args.chunk_size)

    ref_d = np.load(args.reference_npz)
    reference = ref_d[f"L{args.layer}"].astype(np.float64)
    print(f"scoring novelty vs {len(reference)}-point reference (k={K})...", flush=True)
    novelty_dist = knn_novelty(emb, reference, K)

    out = pd.DataFrame({"protein_id": ids, "genos_m_novelty": novelty_dist})
    out_csv.parent.mkdir(exist_ok=True)
    out.to_csv(out_csv, index=False)
    ckpt_path.unlink(missing_ok=True)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
