#!/usr/bin/env python
"""Evaluate the frozen scorer on the independent test set (genomes new in GTDB R232).

Nothing is re-tuned: ESM-2 layer 22, k = 5, threshold at the 90th percentile, and the tail is fitted
on the ORIGINAL references only. The new genomes and their species never occur in the panel.

  (a) Calibration transfer: characterised proteins (exactly one Pfam 35.0 family) from the new
      genomes, scored against the original 15,360-protein single-hit reference with the tail fitted
      on that reference's leave-one-out distances. Flag rate at each alpha, split by whether the
      protein's family has >= 5 members in the reference (the known-like case the guarantee covers).
  (b) Hidden-family test on new genomes: for every family with >= 5 new-genome members and >= 1
      reference member, hide it from the reference; its new-genome members are novel, the other
      new-genome proteins are known. Mean and median AUROC over families.
  (c) Retrospective task: new-genome proteins dark at Pfam 35.0, positives = new Pfam 37.0 family,
      scored against the original 29,862-protein scorer reference at layers 22 and 33; length baseline.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/independent_eval.py
Embeddings are cached in data/processed/independent_R232/esm2_L22_L33.npz.
Writes results/independent/evaluation.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from darkmatter.scoring.gpd import fit_gpd_tail, tail_pvalue  # noqa: E402
from darkmatter.scoring.knn import knn_distance, l2_normalize  # noqa: E402
from reembed_eval import _knn_dist_excluding, auroc  # noqa: E402

IND = ROOT / "data" / "processed" / "independent_R232"
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "independent"
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
rng = np.random.default_rng(42)


def boot(y, s, n=500):
    v = [auroc(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def embed(ids, seqs):
    import torch
    from darkmatter.device import inference_device
    from darkmatter.embeddings import load_embedder
    torch.backends.cuda.matmul.allow_tf32 = True
    e = load_embedder("esm2"); e.model = e.model.float()
    model, tok = e.model, e.tokenizer; dev = inference_device(e.plan, model)
    lens = np.array([min(len(seqs[i]), 1022) for i in ids]); order = np.argsort(lens, kind="stable")
    out = {22: np.zeros((len(ids), 1280), np.float32), 33: np.zeros((len(ids), 1280), np.float32)}
    i, t0 = 0, time.time()
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * lens[order[j]] <= 3000 and j - i < 64:
            j += 1
        j = max(j, i + 1); rows = order[i:j]
        tk = tok([seqs[ids[r]] for r in rows], padding=True, truncation=True, max_length=e.max_tokens, return_tensors="pt")
        tk = {k: v.to(dev) for k, v in tk.items()}
        with torch.inference_mode():
            hs = model(**tk, output_hidden_states=True).hidden_states
        m = tk["attention_mask"].unsqueeze(-1)
        for L in (22, 33):
            mm = m.to(hs[L].dtype)
            out[L][rows] = ((hs[L] * mm).sum(1) / mm.sum(1).clamp(min=1)).float().cpu().numpy()
        del hs; i = j
    print(f"embedded {len(ids)} in {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    lab = pd.read_csv(IND / "protein_labels.csv")
    lab["nfam"] = lab.pfam35_families.fillna("").map(lambda s: len([x for x in s.split(";") if x]))
    single = lab[lab.nfam == 1].copy(); single["family"] = single.pfam35_families
    dark = lab[lab.dark_at_t0]
    char_s = single.sample(min(12000, len(single)), random_state=42)
    dark_s = pd.concat([dark[dark.positive_proxy], dark[~dark.positive_proxy].sample(min(10000, (~dark.positive_proxy).sum()), random_state=42)])
    ids = list(char_s.protein_id) + list(dark_s.protein_id)

    cache = IND / "esm2_L22_L33.npz"
    if cache.exists():
        z = np.load(cache); assert list(z["ids"]) == ids; E = {22: z["L22"], 33: z["L33"]}
    else:
        seqs = {}
        for f in (IND / "proteins").glob("*.faa"):
            name = None
            for line in open(f):
                if line.startswith(">"):
                    name = line[1:].split()[0]; seqs[name] = []
                else:
                    seqs[name].append(line.strip())
        seqs = {k: "".join(v) for k, v in seqs.items()}
        E = embed(ids, seqs)
        np.savez(cache, ids=np.array(ids), L22=E[22], L33=E[33])
    nchar = len(char_s)
    res = {"n_genomes": int(lab.genome_accession.nunique()), "n_proteins": int(len(lab)),
           "dark_fraction": float(lab.dark_at_t0.mean()), "n_positives": int(lab.positive_proxy.sum()),
           "positive_share_of_dark": float(lab.positive_proxy.sum() / lab.dark_at_t0.sum())}

    # ---- original references ----
    ref = np.load(PROC / "esm2_reembed_L33_22.npz")
    rfam = np.array([str(f).split(".")[0] for f in ref["families"]])
    R = l2_normalize(ref["L22"].astype(np.float64))
    fam_size = pd.Series(rfam).value_counts()

    # (a) calibration transfer
    loo = knn_distance(R, R, k=5, exclude_self=True)
    tail = fit_gpd_tail(loo, 0.90)
    Xc = l2_normalize(E[22][:nchar].astype(np.float64))
    pc = tail_pvalue(tail, knn_distance(Xc, R, k=5), ref_distances=loo)
    present = char_s.family.map(fam_size).fillna(0).to_numpy()
    groups = {"all": np.ones(nchar, bool), "family_in_reference_ge5": present >= 5,
              "family_in_reference_1to4": (present >= 1) & (present < 5), "family_absent_from_reference": present == 0}
    res["calibration_transfer"] = {"alpha": ALPHAS.tolist(), **{
        g: {"n": int(m.sum()), "flag_rate": [float((pc[m] <= a).mean()) for a in ALPHAS]} for g, m in groups.items() if m.sum()}}
    print("calibration transfer:", {g: [round(x, 3) for x in v["flag_rate"]] for g, v in res["calibration_transfer"].items() if isinstance(v, dict)}, flush=True)

    # (b) hidden-family test on the new genomes
    sims = Xc @ R.T
    top = np.argpartition(-sims, 500, axis=1)[:, :500]
    tops = np.take_along_axis(sims, top, axis=1)
    o = np.argsort(-tops, axis=1); top = np.take_along_axis(top, o, 1); tops = np.take_along_axis(tops, o, 1)
    cf = char_s.family.to_numpy()
    counts = pd.Series(cf).value_counts()
    fams = [f for f, n in counts.items() if n >= 5 and fam_size.get(f, 0) >= 1]
    aucs = []
    for F in fams:
        excl = rfam == F
        d = _knn_dist_excluding(top, tops, excl, 5)
        y = (cf == F).astype(int)
        aucs.append(auroc(y, d))
    aucs = np.array([a for a in aucs if not np.isnan(a)])
    res["hidden_family_new_genomes"] = {"n_families": int(len(aucs)), "mean": float(aucs.mean()), "median": float(np.median(aucs)),
                                        "frac_below_0.7": float((aucs < 0.7).mean())}
    print("hidden-family on new genomes:", res["hidden_family_new_genomes"], flush=True)

    # (c) retrospective task on new genomes against the 29,862-protein scorer reference
    samp = pd.read_csv(PROC / "embedding_sample.csv")
    cat = samp.set_index("protein_id").category
    q1 = np.load(PROC / "esm2_reembed_queries_L33_22.npz"); rem = np.load(PROC / "esm2_L22_L33_remaining.npz")
    allids = np.array([str(i) for i in np.r_[ref["ids"], q1["ids"], rem["ids"]]])
    isr = cat.reindex(allids).to_numpy() == "characterised_at_t0"
    y = dark_s.positive_proxy.to_numpy().astype(int)
    lens = dark_s.protein_id.map(lambda p: None)
    seqlen = {}
    for f in (IND / "proteins").glob("*.faa"):
        name = None
        for line in open(f):
            if line.startswith(">"):
                name = line[1:].split()[0]; seqlen[name] = 0
            else:
                seqlen[name] += len(line.strip())
    L = dark_s.protein_id.map(seqlen).to_numpy(float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
           "length": {"auroc": float(auroc(y, L)), "ci95": boot(y, L)}}
    for layer in (22, 33):
        Rl = l2_normalize(np.concatenate([ref[f"L{layer}"], q1[f"L{layer}"], rem[f"L{layer}"]])[isr].astype(np.float64))
        d = knn_distance(l2_normalize(E[layer][nchar:].astype(np.float64)), Rl, k=5)
        order = np.argsort(-d, kind="stable")
        out[f"layer{layer}"] = {"auroc": float(auroc(y, d)), "ci95": boot(y, d),
                                "precision_at_k": {str(k): float(y[order[:k]].mean()) for k in (50, 100, 500, 1000) if k <= len(y)}}
    res["retrospective_new_genomes"] = out
    print("retrospective on new genomes:", {k: (round(v["auroc"], 3) if isinstance(v, dict) else v) for k, v in out.items()}, flush=True)
    (OUT / "evaluation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
