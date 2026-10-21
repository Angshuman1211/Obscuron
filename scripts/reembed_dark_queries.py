#!/usr/bin/env python
"""Re-embed the dark-query set (dark_negative + positive) at chosen layer(s) +
fp32, matching scripts/reembed_eval.py's reference-side protocol exactly (same
model, same layers, same pooling) so the two embedding matrices are directly
comparable. This is the Track-2 "scale flagging to the dark-query set" input
(docs/Track1_phase3_immune_design.md, deliverable 1 cross-track hand-off) --
untested in P3-D6 because only the reference had layer-22 embeddings.

Checkpoints every --chunk-size sequences (mirrors P1-D10's chunking practice
for multi-hour runs) so an interrupted run resumes instead of restarting.

    uv run python scripts/reembed_dark_queries.py --layers 33,22 --fp32
    uv run python scripts/reembed_dark_queries.py --layers 33,22 --fp32 --limit 500   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
SEQ_DIR, SUFFIX = "panel_proteins", "_protein.faa"


def _load_checkpoint(ckpt_path: Path, want_layers: list[int]) -> tuple[dict[int, list], list[str]]:
    if not ckpt_path.exists():
        return {li: [] for li in want_layers}, []
    d = np.load(ckpt_path)
    done_ids = [str(x) for x in d["ids"]]
    acc = {li: [d[f"L{li}"]] for li in want_layers}
    print(f"resuming from checkpoint: {len(done_ids)} sequences already embedded", flush=True)
    return acc, done_ids


def _save_checkpoint(ckpt_path: Path, acc: dict[int, list], done_ids: list[str]) -> None:
    tmp = ckpt_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"L{li}": np.concatenate(v) for li, v in acc.items()},
                        ids=np.array(done_ids))
    tmp.replace(ckpt_path)


def embed_layers_checkpointed(ids: list[str], seq_by_id: dict[str, str], want_layers: list[int],
                              fp32: bool, batch_size: int, ckpt_path: Path,
                              chunk_size: int) -> dict[int, np.ndarray]:
    import torch
    from darkmatter.device import inference_device
    from darkmatter.embeddings import load_embedder

    acc, done_ids = _load_checkpoint(ckpt_path, want_layers)
    done_set = set(done_ids)
    remaining = [s for s in ids if s not in done_set]
    if not remaining:
        print("all sequences already embedded (checkpoint complete)", flush=True)
        return {li: np.concatenate(v) for li, v in acc.items()}

    embedder = load_embedder("esm2")
    if fp32:
        embedder.model = embedder.model.float()
    model, tok = embedder.model, embedder.tokenizer
    device = inference_device(embedder.plan, model)

    t0 = time.time()
    since_ckpt = 0
    for i in range(0, len(remaining), batch_size):
        chunk = remaining[i:i + batch_size]
        batch = [seq_by_id[s] for s in chunk]
        toks = tok(batch, padding=True, truncation=True, max_length=embedder.max_tokens,
                   return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            hs = model(**toks, output_hidden_states=True).hidden_states
        mask = toks["attention_mask"].unsqueeze(-1)
        for li in want_layers:
            m = mask.to(hs[li].dtype)
            pooled = (hs[li] * m).sum(1) / m.sum(1).clamp(min=1)
            acc[li].append(pooled.float().cpu().numpy())
        del hs
        done_ids.extend(chunk)
        since_ckpt += len(chunk)
        n_done = len(done_ids)
        if n_done % (batch_size * 10) < batch_size:
            print(f"  embedded {n_done}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
        if since_ckpt >= chunk_size:
            _save_checkpoint(ckpt_path, acc, done_ids)
            since_ckpt = 0
            print(f"  checkpoint saved at {n_done}/{len(ids)}", flush=True)
    _save_checkpoint(ckpt_path, acc, done_ids)
    print(f"embedded {len(remaining)} new seqs at layers {want_layers} in {time.time()-t0:.0f}s "
          f"({len(done_ids)}/{len(ids)} total)", flush=True)
    # re-order to match `ids` order (checkpoint order = arrival order, not `ids` order after resume)
    pos = {s: k for k, s in enumerate(done_ids)}
    order = [pos[s] for s in ids]
    full = {li: np.concatenate(v) for li, v in acc.items()}
    return {li: full[li][order] for li in want_layers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="33,22", help="comma list; last first as the baseline")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N queries (smoke-test); 0 = all")
    ap.add_argument("--n-positive", type=int, default=0, help="subsample positives; 0 = all")
    ap.add_argument("--n-dark-negative", type=int, default=0, help="subsample dark_negative; 0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=1000, help="checkpoint every N sequences")
    args = ap.parse_args()
    want = [int(x) for x in args.layers.split(",")]
    tag = f"L{'_'.join(map(str, want))}"
    npz_path = PROC / f"esm2_reembed_queries_{tag}.npz"
    ckpt_path = PROC / f"esm2_reembed_queries_{tag}.ckpt.npz"

    sample = pd.read_csv(PROC / "embedding_sample.csv")
    pos = sample[sample["category"] == "positive"]
    neg = sample[sample["category"] == "dark_negative"]
    rng = np.random.default_rng(args.seed)
    if args.n_positive:
        pos = pos.sample(n=min(args.n_positive, len(pos)), random_state=args.seed)
    if args.n_dark_negative:
        neg = neg.sample(n=min(args.n_dark_negative, len(neg)), random_state=args.seed)
    queries = pd.concat([pos, neg]).reset_index(drop=True)
    if args.limit:
        queries = queries.iloc[: args.limit].reset_index(drop=True)
    print(f"{len(queries)} dark queries "
          f"({(queries['category'] == 'positive').sum()} positive, "
          f"{(queries['category'] == 'dark_negative').sum()} dark_negative)", flush=True)

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
    is_positive = (queries["category"] == "positive").to_numpy()
    print(f"{len(ids)} sequences loaded", flush=True)

    layers = embed_layers_checkpointed(ids, seq_by_id, want, args.fp32, args.batch_size, ckpt_path, args.chunk_size)
    np.savez_compressed(npz_path, **{f"L{li}": layers[li] for li in want},
                        ids=np.array(ids), is_positive=is_positive)
    ckpt_path.unlink(missing_ok=True)
    print(f"wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
