#!/usr/bin/env python
"""Re-embed the characterised reference at chosen layer(s) + centering, then run the
held-out-family AUROC (P2-D4) -- the definitive test of the P2-D8 representation fix
against the 0.906 baseline. One script, both arms:

  Run 1 (local): uv run python scripts/reembed_eval.py --model esm2    --layers 33,22 --fp32
  Run 2 (cloud): uv run python scripts/reembed_eval.py --model genos-m --layers 12,9

Reference = the 29,862 `characterised_at_t0` proteins in embedding_sample.csv. Families
come from hmmscan vs Pfam-35 on the PROTEIN sequences (identical across arms, cached to
reference_families.csv). ESM-2 needs --fp32 (fp16 NaNs its intermediate layers). Compares
each (layer, {raw, centered}) under one consistent eval; last-layer/raw is the internal
baseline and cross-checks Rayyan's independent 0.906. Smoke-test first with --limit 200.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
PFAM35 = ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.hmm.gz"
MODALITY = {"esm2": ("panel_proteins", "_protein.faa", False),
            "genos-m": ("panel_nucleotides", "_protein.fna", True)}
K = 5
MIN_MEMBERS = 5


# ---------- families (shared across arms, from the protein hmmscan) ----------
def reference_families(ref: pd.DataFrame, cpus: int) -> dict[str, str]:
    """protein_id -> single Pfam-35 family, for reference proteins with exactly one hit.
    Cached to reference_families.csv so it is computed once."""
    cache = PROC / "reference_families.csv"
    if cache.exists():
        fam = pd.read_csv(cache)
        print(f"loaded {len(fam)} cached reference families", flush=True)
        return dict(zip(fam["protein_id"], fam["family"]))
    from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile
    from darkmatter.data.hmmscan import scan_against_pfam

    def dec(x):  # pyhmmer names/accessions are bytes
        return x.decode() if isinstance(x, (bytes, bytearray)) else x

    alphabet = Alphabet.amino()
    wanted = set(ref["protein_id"])
    kept = []  # load ONLY the reference proteins, not all 1.34M
    for acc in ref["genome_accession"].unique():
        p = PROC / "panel_proteins" / f"{acc}_protein.faa"
        if not p.exists():
            continue
        with SequenceFile(p, digital=True, alphabet=alphabet) as sf:
            kept.extend(s for s in sf if dec(s.name) in wanted)
    print(f"hmmsearch {len(kept)} reference proteins vs Pfam-35 (GA)...", flush=True)
    t0 = time.time()
    hits = scan_against_pfam(DigitalSequenceBlock(alphabet, kept), PFAM35, cpus=cpus)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    fam = {dec(sid): dec(fams[0]) for sid, fams in hits.items() if len(fams) == 1}
    fam = {k: v for k, v in fam.items() if k in wanted}
    pd.DataFrame({"protein_id": list(fam), "family": list(fam.values())}).to_csv(cache, index=False)
    print(f"  {len(fam)} single-hit reference proteins, cached -> {cache}", flush=True)
    return fam


# ---------- embedding at specific layers ----------
def embed_layers(model_name: str, ids: list[str], seq_by_id: dict[str, str],
                 want_layers: list[int], fp32: bool, batch_size: int) -> dict[int, np.ndarray]:
    import torch
    from darkmatter.device import inference_device
    from darkmatter.embeddings import load_embedder
    preprocess = (lambda s: s)
    if MODALITY[model_name][2]:
        from darkmatter.embeddings.genos_m import _preprocess as preprocess

    embedder = load_embedder(model_name)
    if fp32:
        embedder.model = embedder.model.float()
    model, tok = embedder.model, embedder.tokenizer
    device = inference_device(embedder.plan, model)

    acc = {li: [] for li in want_layers}
    t0 = time.time()
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        batch = [preprocess(seq_by_id[s]) for s in chunk]
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
        if (i // batch_size) % 20 == 0:
            print(f"  embedded {min(i+batch_size,len(ids))}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"embedded {len(ids)} seqs at layers {want_layers} in {time.time()-t0:.0f}s", flush=True)
    return {li: np.concatenate(v) for li, v in acc.items()}


# ---------- held-out-family AUROC (P2-D4) ----------
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    np_, nn_ = int(pos.sum()), int((~pos).sum())
    return np.nan if np_ == 0 or nn_ == 0 else (ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_)


def _topm_neighbors(vn: np.ndarray, m: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For every row in vn, its top-m nearest OTHER rows (self excluded), by
    cosine similarity. Chunked over query rows so peak memory is chunk x N,
    never the full N x N matrix (this is the same memory discipline the old,
    now-removed per-family knn() loop used -- "avoids a 15k x 15k matrix" --
    just paid once for the whole reference instead of once per held-out
    family, which was the O(F * N^2) bottleneck this replaces). Returns
    (indices, similarities), each (N, m); m must comfortably exceed the
    largest family size so that filtering out one held-out family still
    leaves >= K neighbours.
    """
    n = len(vn)
    idx_out = np.empty((n, m), dtype=np.int32)
    sim_out = np.empty((n, m), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        block = vn[a:b] @ vn.T  # (b-a, n)
        rows = np.arange(b - a)
        block[rows, a + rows] = -np.inf  # exclude self
        mm = min(m, n - 1)
        top_idx = np.argpartition(block, -mm, axis=1)[:, -mm:]
        top_sim = np.take_along_axis(block, top_idx, axis=1)
        order = np.argsort(-top_sim, axis=1)
        idx_out[a:b, :mm] = np.take_along_axis(top_idx, order, axis=1)
        sim_out[a:b, :mm] = np.take_along_axis(top_sim, order, axis=1)
        if mm < m:
            idx_out[a:b, mm:] = -1
            sim_out[a:b, mm:] = -np.inf
    return idx_out, sim_out


def _knn_dist_excluding(idx: np.ndarray, sim: np.ndarray, exclude_family: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest neighbours, per row, after
    dropping any precomputed neighbour that falls in exclude_family (a bool
    mask over the full reference, indexed by idx's values)."""
    bad = exclude_family[idx.clip(min=0)] | (idx < 0)
    sim_masked = np.where(bad, -np.inf, sim)
    order = np.argsort(-sim_masked, axis=1)[:, :k]
    top = np.take_along_axis(sim_masked, order, axis=1)
    return 1.0 - top.mean(axis=1)


def heldout_auroc(vectors, families, center, seed, top_m: int = 500):
    """Held-out-family AUROC (P2-D4), same math as the original per-family loop
    but each point's neighbour search is done ONCE against the full reference
    (chunked, top-m kept) instead of once per family (was O(F * N^2) total
    work; now O(N^2) once + O(F * N) cheap filtering). top_m must exceed the
    largest family size; 500 comfortably covers this benchmark's families.

    center uses the GLOBAL mean (not the per-fold, family-excluded mean): P2-D9
    already established centering's effect on this AUROC is negligible (33:
    0.786->0.783, 22: 0.962->0.961), and removing a handful of family members
    from a >10k-row reference shifts the mean by a negligible amount anyway --
    so this is a justified approximation, not a silent behaviour change.
    """
    del seed  # unused now that nothing here is sampled; kept for call-site compatibility
    fams = np.array(families)
    keep = pd.Series(fams).groupby(fams).transform("size").values >= MIN_MEMBERS
    v, f = vectors[keep], fams[keep]

    mu = v.mean(0, keepdims=True) if center else 0.0
    vn = v - mu
    vn = vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    idx, sim = _topm_neighbors(vn, top_m)  # the one O(N^2) cost, paid once

    scores = []
    for F in np.unique(f):
        is_f = f == F
        novel = _knn_dist_excluding(idx[is_f], sim[is_f], is_f, K)
        known = _knn_dist_excluding(idx[~is_f], sim[~is_f], is_f, K)
        a = auroc(np.r_[np.ones(len(novel)), np.zeros(len(known))], np.r_[novel, known])
        if not np.isnan(a):
            scores.append(a)
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.median(scores)), len(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["esm2", "genos-m"], required=True)
    ap.add_argument("--layers", default="33,22", help="comma list; last first as the baseline")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip embedding; re-run the AUROC table from the saved *_reembed_*.npz")
    ap.add_argument("--max-families", type=int, default=0,
                    help="keep only the N largest families (smoke-test); 0 = all")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    want = [int(x) for x in args.layers.split(",")]
    seq_dir, suffix, _ = MODALITY[args.model]
    npz_path = PROC / f"{args.model}_reembed_L{'_'.join(map(str, want))}.npz"

    if args.eval_only:
        d = np.load(npz_path)
        layers = {li: d[f"L{li}"] for li in want}
        families = [str(x) for x in d["families"]]
        if args.max_families:
            # match a capped run (e.g. Rayyan's 300-largest-family Genos-m eval) so the
            # ESM-2 comparison is WITHIN the same family set, not cross-eval (P2-D9 lesson)
            top = pd.Series(families).value_counts().head(args.max_families).index
            mask = pd.Series(families).isin(top).values
            layers = {li: layers[li][mask] for li in want}
            families = [f for f, m in zip(families, mask) if m]
            print(f"capped to {args.max_families} largest families", flush=True)
        print(f"eval-only: {layers[want[0]].shape[0]} reference embeddings, "
              f"{len(set(families))} families (from {npz_path.name})", flush=True)
    else:
        sample = pd.read_csv(PROC / "embedding_sample.csv")
        ref = sample[sample["category"] == "characterised_at_t0"].reset_index(drop=True)
        fam_map = reference_families(ref, args.cpus)
        ref = ref[ref["protein_id"].isin(fam_map)].reset_index(drop=True)
        ref["family"] = ref["protein_id"].map(fam_map)
        if args.max_families:
            top = ref["family"].value_counts().head(args.max_families).index
            ref = ref[ref["family"].isin(top)].reset_index(drop=True)
        print(f"{len(ref)} reference proteins, {ref['family'].nunique()} families", flush=True)

        from Bio import SeqIO
        wanted = set(ref["protein_id"])
        seq_by_id = {}
        for acc in ref["genome_accession"].unique():
            p = PROC / seq_dir / f"{acc}{suffix}"
            if p.exists():
                for rec in SeqIO.parse(p, "fasta"):
                    if rec.id in wanted:
                        seq_by_id[rec.id] = str(rec.seq)
        ref = ref[ref["protein_id"].isin(seq_by_id)].reset_index(drop=True)
        ids = ref["protein_id"].tolist()
        families = [fam_map[s] for s in ids]
        print(f"{len(ids)} sequences loaded for {args.model}", flush=True)

        layers = embed_layers(args.model, ids, seq_by_id, want, args.fp32, args.batch_size)
        np.savez_compressed(npz_path, **{f"L{li}": layers[li] for li in want},
                            ids=np.array(ids), families=np.array(families))

    print("\nlayer | center | AUROC(mean) | AUROC(median) | n_families")
    print("------+--------+-------------+---------------+-----------")
    for li in want:
        for center in (False, True):
            m, med, nf = heldout_auroc(layers[li], families, center, args.seed)
            tag = "cen" if center else "raw"
            print(f"{li:5d} | {tag:>6} | {m:11.4f} | {med:13.4f} | {nf}")
    print(f"\nbaseline anchor: Rayyan's independent full-pipeline held-out AUROC = 0.906 (last layer, raw)")


if __name__ == "__main__":
    main()
