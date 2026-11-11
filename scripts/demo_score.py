#!/usr/bin/env python
"""Score protein sequences with the calibrated novelty scorer (Layer 1).

Embeds each sequence with ESM-2 (layer 22, mean-pooled, L2-normalised,
fp32), measures its mean cosine distance to the k nearest characterised
reference proteins, and converts that distance into a GPD tail p-value
fitted on the reference's leave-one-out distances (config/scorer.yaml:
k = 5, threshold = 90th percentile). Prints the score, the p-value, the
flag at level alpha and the Pfam families of the nearest references.

Reference: the 15,360 single-hit Pfam 35.0 proteins of the held-out-family
test (data/processed/gtdb_R207/esm2_reembed_L33_22.npz, built by
scripts/reembed_reference_immune.py; gitignored, built locally).

Usage:
    uv run python scripts/demo_score.py --fasta examples/demo_proteins.fasta
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from darkmatter.device import inference_device
from darkmatter.embeddings import load_embedder
from darkmatter.scoring.gpd import fit_gpd_tail, tail_pvalue

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "processed" / "gtdb_R207" / "esm2_reembed_L33_22.npz"


def read_fasta(path: Path) -> list[tuple[str, str, str]]:
    records, pid, desc, seq = [], None, "", []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if pid is not None:
                records.append((pid, desc, "".join(seq)))
            head = line[1:].split(maxsplit=1)
            pid, desc, seq = head[0], head[1] if len(head) > 1 else "", []
        elif line.strip():
            seq.append(line.strip())
    if pid is not None:
        records.append((pid, desc, "".join(seq)))
    return records


def l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)


def embed(seqs: list[str], layer: int) -> np.ndarray:
    emb = load_embedder("esm2")
    model = emb.model.float()
    device = inference_device(emb.plan, model)
    out = []
    for s in seqs:
        toks = emb.tokenizer([s], truncation=True, max_length=emb.max_tokens, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            hs = model(**toks, output_hidden_states=True).hidden_states[layer]
        m = toks["attention_mask"].unsqueeze(-1).to(hs.dtype)
        out.append(((hs * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def loo_knn(ref: torch.Tensor, k: int, chunk: int = 2048) -> np.ndarray:
    d = []
    for a in range(0, len(ref), chunk):
        sims = ref[a:a + chunk] @ ref.T
        idx = torch.arange(a, min(a + chunk, len(ref)), device=ref.device)
        sims[idx - a, idx] = -float("inf")
        d.append((1 - sims.topk(k, dim=1).values).mean(1).cpu())
    return torch.cat(d).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", type=Path, default=ROOT / "examples" / "demo_proteins.fasta")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    z = np.load(REF, allow_pickle=False)
    ids, fams = z["ids"], z["families"]
    ref = l2(torch.from_numpy(z[f"L{args.layer}"]).float().to(dev))
    loo = loo_knn(ref, args.k)
    tail = fit_gpd_tail(loo, 0.90)
    print("Obscuron calibrated novelty scorer (Layer 1)")
    print(f"  reference : {len(ids):,} characterised proteins (Pfam 35.0, single hit), ESM-2 layer {args.layer}")
    print(f"  tail fit  : u = {tail.u:.4f} (90th pct), xi = {tail.xi:.3f}, beta = {tail.beta:.4f}, "
          f"{tail.n_exceedances:,} exceedances")
    print(f"  device    : {dev}, reference ready in {time.time() - t0:.1f} s\n")

    recs = read_fasta(args.fasta)
    t1 = time.time()
    q = l2(torch.from_numpy(embed([s for _, _, s in recs], args.layer)).to(dev))
    sims = q @ ref.T
    for i, (pid, _, _) in enumerate(recs):
        sims[i, torch.from_numpy(ids == pid).to(dev)] = -float("inf")  # never match itself
    top = sims.topk(args.k, dim=1)
    dist = (1 - top.values).mean(1).cpu().numpy()
    p = tail_pvalue(tail, dist, loo)
    print(f"Scored {len(recs)} proteins in {time.time() - t1:.1f} s  (flag: p <= {args.alpha})\n")
    print(f"{'protein':<24}{'len':>5}{'d5':>9}{'p-value':>11}{'score':>8}  {'flag':<6}nearest Pfam families (cosine)")
    print("-" * 112)
    for i, (pid, desc, s) in enumerate(recs):
        near = ", ".join(f"{fams[j].split('.')[0]} ({v:.3f})"
                         for v, j in zip(top.values[i, :3].tolist(), top.indices[i, :3].tolist()))
        flag = "NOVEL" if p[i] <= args.alpha else "known"
        print(f"{pid:<24}{len(s):>5}{dist[i]:>9.4f}{p[i]:>11.2e}{1 - p[i]:>8.3f}  {flag:<6}{near}")
        print(f"{'':<24}{desc}")


if __name__ == "__main__":
    main()
