#!/usr/bin/env python
"""Structure case studies of the most novel dark proteins (TCBB revision).

Picks the highest-ranked still-dark proteins by the layer-22 score on the 34,138 dark queries
(against the 29,862-protein reference), restricted to 100-400 residues so ESMFold fits on an 8 GB
GPU, predicts each structure with ESMFold (facebook/esmfold_v1, pinned revision), and searches the
predicted structure against AlphaFold DB (afdb50) and PDB (pdb100) with the public Foldseek server
in 3Di+AA mode. Only public GTDB protein sequences are sent.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/case_studies.py --ranked results/case_studies/dark_ranked_layer22.csv [--n 3] [--control <protein id>]
Writes results/case_studies/{<protein>.pdb, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "case_studies"
OUT.mkdir(parents=True, exist_ok=True)
ESMFOLD = ("facebook/esmfold_v1", "75a3841ee059df2bf4d56688166c8fb459ddd97a")
# downloaded as plain files (the Windows Hugging Face cache needs symlink rights):
#   snapshot_download(*ESMFOLD, local_dir=ESMFOLD_DIR)
ESMFOLD_DIR = ROOT / ".cache" / "esmfold_v1_75a3841"
# Weights are read from model.safetensors (Hugging Face SFconvertbot conversion, PR #6 of
# facebook/esmfold_v1), not pytorch_model.bin: torch < 2.6 cannot load .bin files safely
# (CVE-2025-32434), and safetensors needs no pickle at all.
FOLDSEEK = "https://search.foldseek.com/api"


def sequence(pid: str) -> str:
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    acc = samp.loc[pid, "genome_accession"]
    from Bio import SeqIO
    for rec in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
        if rec.id == pid:
            return str(rec.seq).rstrip("*")
    raise KeyError(pid)


def fold(seqs: dict[str, str]) -> dict[str, dict]:
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    src = ESMFOLD_DIR if ESMFOLD_DIR.exists() else ESMFOLD[0]
    kw = {} if ESMFOLD_DIR.exists() else {"revision": ESMFOLD[1]}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    model = EsmForProteinFolding.from_pretrained(src, low_cpu_mem_usage=True, use_safetensors=True, **kw)
    model.esm = model.esm.half()
    model = model.cuda().eval()
    model.trunk.set_chunk_size(64)
    out = {}
    for pid, s in seqs.items():
        t0 = time.time()
        with torch.inference_mode():
            ids = tok([s], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            o = model(ids)
        pdb = model.output_to_pdb(o)[0]
        plddt = o["plddt"][0, :, 1].float().cpu().numpy()  # CA atom
        (OUT / f"{pid}.pdb").write_text(pdb)
        out[pid] = {"length": len(s), "mean_plddt": float(plddt.mean() * (100 if plddt.max() <= 1 else 1)),
                    "fold_seconds": round(time.time() - t0, 1)}
        print(pid, out[pid], flush=True)
        torch.cuda.empty_cache()
    return out


def _retry(fn, tries=5):
    """the public Foldseek server occasionally drops connections; retry with back-off"""
    for i in range(tries):
        try:
            return fn()
        except (requests.RequestException, ValueError):
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))


def plddt_from_pdb(pid: str) -> float:
    """ESMFold writes per-atom pLDDT into the B-factor column; mean over C-alpha atoms."""
    vals = [float(l[60:66]) for l in (OUT / f"{pid}.pdb").read_text().splitlines()
            if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    v = float(np.mean(vals))
    return v * 100 if v <= 1 else v


def foldseek(pid: str) -> dict:
    """search once per protein; results are cached so an interrupted run does not repeat searches"""
    cache = OUT / "foldseek" / f"{pid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    fs = _retry(lambda: _foldseek(pid))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(fs, indent=1))
    return fs


def _foldseek(pid: str) -> dict:
    pdb = (OUT / f"{pid}.pdb").read_bytes()
    r = requests.post(f"{FOLDSEEK}/ticket", files={"q": (f"{pid}.pdb", pdb)},
                      data=[("mode", "3diaa"), ("database[]", "afdb50"), ("database[]", "pdb100")], timeout=60)
    r.raise_for_status()
    tid = r.json()["id"]
    for _ in range(120):
        st = requests.get(f"{FOLDSEEK}/ticket/{tid}", timeout=30).json().get("status")
        if st == "COMPLETE":
            break
        if st == "ERROR":
            raise RuntimeError(f"foldseek error for {pid}")
        time.sleep(10)
    res = requests.get(f"{FOLDSEEK}/result/{tid}/0", timeout=60).json()
    summary = {"ticket": tid}
    for db in res.get("results", []):
        name = db.get("db")
        alns = db.get("alignments") or []
        if alns and isinstance(alns[0], list):
            alns = [a for chain in alns for a in chain]
        alns = sorted(alns, key=lambda a: float(a.get("eval", 1e9)))
        summary[name] = {"n_hits": len(alns), "top": [
            {k: a.get(k) for k in ("target", "prob", "eval", "score", "seqId", "alnLength", "qStartPos", "qEndPos", "taxName")}
            for a in alns[:5]]}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", required=True, help="CSV with pid, d22, cat, len, rank (layer-22 ranking of dark queries)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--control", default=None, help="a characterised protein run through the same steps as a positive control")
    ap.add_argument("--filtered", default=None,
                    help="filtered ranking CSV (results/revision/filtered_ranking.csv): fold its top --n proteins of 100-300 residues")
    ap.add_argument("--mid-ranked", action="store_true",
                    help="fold --n random still-dark proteins from ranks 15,000 to 20,000 as a comparison set")
    ap.add_argument("--out", default=None, help="summary file name (default depends on the mode)")
    args = ap.parse_args()
    t = pd.read_csv(args.ranked)
    pick = t[(t.cat == "dark_negative") & t.len.between(100, 400)].head(args.n)
    if args.filtered:
        f = pd.read_csv(args.filtered).rename(columns={"protein_id": "pid", "length": "len"})
        pick = f[f.len.between(100, 300)].head(args.n)
    if args.mid_ranked:
        # comparison set: random still-dark proteins of 100-300 residues from the middle of the ranking
        mid = t[(t.cat == "dark_negative") & t.len.between(100, 300) & t["rank"].between(15000, 20000)]
        pick = mid.sample(args.n, random_state=42)
    if args.control:
        pick = pd.DataFrame([{"pid": args.control, "d22": float("nan"), "cat": "control", "rank": -1, "phylum": None}])
    seqs = {p: sequence(p) for p in pick.pid}
    todo = {p: q for p, q in seqs.items() if not (OUT / f"{p}.pdb").exists()}
    folded = fold(todo) if todo else {}
    for p in seqs:  # structures already on disk are reused; pLDDT is read back from the PDB file
        folded.setdefault(p, {"length": len(seqs[p]), "mean_plddt": plddt_from_pdb(p), "fold_seconds": None})
    summary = {}
    for _, r in pick.iterrows():
        fs = foldseek(r.pid)
        summary[r.pid] = {"rank_layer22": int(r["rank"]), "distance_layer22": float(r.d22), "phylum": r.get("phylum"),
                          **folded[r.pid], "foldseek": fs}
        print(r.pid, json.dumps(fs)[:400], flush=True)
    name = args.out or ("control.json" if args.control else "filtered_summary.json" if args.filtered
                        else "mid_ranked_summary.json" if args.mid_ranked else "summary.json")
    (OUT / name).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
