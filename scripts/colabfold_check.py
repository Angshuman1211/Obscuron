#!/usr/bin/env python
"""Alignment-based structure prediction of the structure-comparison proteins (TCBB revision).

ESMFold is built on ESM-2, the model behind the score, so its low confidence at the top of the
ranking is not independent evidence. AlphaFold2 (ColabFold 1.5.5, official Docker image, five
models, no templates) predicts from multiple sequence alignments instead; the alignments are the
ones fetched by scripts/msa_depth.py (UniRef30 + ColabFoldDB, results/revision/msa/). If the
filtered top folds confidently with alignments, its low ESMFold confidence was an ESM-2 blind spot;
if it does not, it lacks the evolutionary information any current predictor needs.

Every model with mean pLDDT >= 70 is searched with Foldseek (AlphaFold DB and PDB). A confident
model without a structural hit would be a candidate new fold.

Resumable: ColabFold skips finished proteins, Foldseek results are cached per protein.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/colabfold_check.py
Writes results/revision/colabfold/ (best model and scores per protein) and
results/revision/colabfold_check.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from case_studies import foldseek  # noqa: E402

CASES = ROOT / "results" / "case_studies"
MSA = ROOT / "results" / "revision" / "msa"
WORK = ROOT / ".cache" / "colabfold"
OUT = ROOT / "results" / "revision" / "colabfold"
IMAGE = "ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2"
E_MAX = 1e-3


def jobs() -> pd.DataFrame:
    """the 60 proteins, alternating between the two sets so a partial run stays balanced"""
    sets = [(n, json.loads((CASES / f).read_text())) for n, f in
            (("filtered_top", "filtered_30.json"), ("mid_ranked", "mid_ranked_30.json"))]
    rows = []
    for i in range(30):
        for n, s in sets:
            p = list(s)[i]
            rows.append({"job": f"{len(rows):02d}_{p}", "pid": p, "set": n, "esmfold_plddt": s[p]["mean_plddt"],
                         "rank": s[p]["rank_layer22"]})
    return pd.DataFrame(rows)


def best_model(job: str) -> tuple[Path | None, float, float]:
    """rank-1 model and its mean pLDDT and pTM"""
    sc = sorted((WORK / "out").glob(f"{job}_scores_rank_001_*.json"))
    pdb = sorted((WORK / "out").glob(f"{job}_unrelaxed_rank_001_*.pdb"))
    if not sc or not pdb:
        return None, np.nan, np.nan
    d = json.loads(sc[0].read_text())
    return pdb[0], float(np.mean(d["plddt"])), float(d.get("ptm", np.nan))


def best_e(fs: dict, db: str) -> float:
    return min((float(a["eval"]) for a in (fs.get(db) or {}).get("top", [])), default=np.inf)


def main():
    (WORK / "in").mkdir(parents=True, exist_ok=True)
    (WORK / "out").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    J = jobs()
    for r in J.itertuples():
        dst = WORK / "in" / f"{r.job}.a3m"
        if not dst.exists():
            shutil.copy(MSA / f"{r.pid}.a3m", dst)
    if len(list((WORK / "out").glob("*.done.txt"))) < len(J):
        # the image lets JAX spill into host memory (unified memory) for large proteins; inside Docker
        # Desktop's memory-limited VM that allocation fails, and 100-300 residues fit on the GPU anyway
        subprocess.run(["docker", "run", "--rm", "--gpus", "all", "-e", "TF_FORCE_UNIFIED_MEMORY=0",
                        "-e", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.85", "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                        "-v", "obscuron-colabfold-cache:/cache",
                        "-v", f"{WORK}:/work", IMAGE, "bash", "-c",
                        "python -m colabfold.download >/dev/null && colabfold_batch --num-models 5 "
                        "/work/in /work/out"], check=True)
    rows = []
    for r in J.itertuples():
        pdb, plddt, ptm = best_model(r.job)
        row = {"pid": r.pid, "set": r.set, "rank": r.rank, "esmfold_plddt": r.esmfold_plddt, "af2_plddt": plddt, "af2_ptm": ptm}
        if pdb is not None:
            shutil.copy(pdb, OUT / f"{r.pid}_af2.pdb")
            if plddt >= 70:
                fs = foldseek(r.pid, OUT / f"{r.pid}_af2.pdb", OUT / "foldseek")
                row.update(afdb_best_e=best_e(fs, "afdb50"), pdb_best_e=best_e(fs, "pdb100"),
                           afdb_top=((fs.get("afdb50") or {}).get("top") or [{}])[0].get("target"),
                           pdb_top=((fs.get("pdb100") or {}).get("top") or [{}])[0].get("target"))
        rows.append(row)
    d = pd.DataFrame(rows)
    depth = pd.DataFrame(json.loads((ROOT / "results" / "revision" / "msa_depth.json").read_text())["per_protein"])
    d = d.merge(depth[["pid", "total_cov50"]], on="pid", how="left")
    a, b = d[d.set == "filtered_top"].dropna(subset=["af2_plddt"]), d[d.set == "mid_ranked"].dropna(subset=["af2_plddt"])

    def summ(x):
        conf = x.af2_plddt >= 70
        novel = conf & (x.get("afdb_best_e", pd.Series(np.inf, index=x.index)).fillna(np.inf) > E_MAX) \
            & (x.get("pdb_best_e", pd.Series(np.inf, index=x.index)).fillna(np.inf) > E_MAX)
        return {"n": int(len(x)), "af2_plddt_median": float(x.af2_plddt.median()),
                "esmfold_plddt_median": float(x.esmfold_plddt.median()), "af2_confident": int(conf.sum()),
                "af2_confident_no_structural_hit": int(novel.sum()),
                "af2_minus_esmfold_median": float((x.af2_plddt - x.esmfold_plddt).median())}
    res = {"image": IMAGE, "models": 5, "templates": False, "filtered_top": summ(a), "mid_ranked": summ(b)}
    if len(a) and len(b):
        res["af2_plddt_mannwhitney_p"] = float(stats.mannwhitneyu(a.af2_plddt, b.af2_plddt).pvalue)
        both = pd.concat([a, b])
        res["spearman_af2_plddt_msa_depth"] = float(stats.spearmanr(both.af2_plddt, both.total_cov50).statistic)
        res["spearman_af2_esmfold_plddt"] = float(stats.spearmanr(both.af2_plddt, both.esmfold_plddt).statistic)
    res["per_protein"] = d.to_dict("records")
    (ROOT / "results" / "revision" / "colabfold_check.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "per_protein"}, indent=1))
    print(d[d.af2_plddt >= 70][["pid", "set", "rank", "af2_plddt", "esmfold_plddt", "total_cov50"]].to_string())


if __name__ == "__main__":
    main()
