#!/usr/bin/env python
"""Disorder at the top of the ranking with a modern predictor, metapredict v3 (TCBB revision).

Same sets and comparisons as scripts/disorder_check.py (Uversky rule and FoldIndex), with
metapredict, a deep-learning predictor of consensus disorder that does not use ESM-2. metapredict
has no Windows build, so it runs in a python:3.11 Docker container (CPU torch), with the version
pinned. A residue is disordered when its score exceeds 0.5.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/metapredict_check.py
Writes results/revision/metapredict_check.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from disorder_check import sequences  # noqa: E402

CASES = ROOT / "results" / "case_studies"
WORK = ROOT / ".cache" / "metapredict"
IMAGE = "python:3.11"
OPTS = "-q --root-user-action=ignore --timeout 120 --retries 10"
# CPU torch from the PyTorch index first, then metapredict from source (its build fetches Cython and
# the version tooling itself); the install is retried, and the pip cache volume keeps what earlier
# attempts downloaded
PIP = (f"pip install {OPTS} --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' && "
       f"pip install {OPTS} 'metapredict==3.1.1'")


def summary(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "disordered_share_median": float(d.metapredict_disordered_share.median()),
            "mostly_disordered_share": float((d.metapredict_disordered_share > 0.5).mean()),
            "mean_score_median": float(d.metapredict_mean.median())}


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    x, y = a.metapredict_disordered_share, b.metapredict_disordered_share
    return {"mannwhitney_p": float(stats.mannwhitneyu(x, y).pvalue),
            "mostly_disordered_fisher_p": float(stats.fisher_exact([[(x > .5).sum(), (x <= .5).sum()],
                                                                    [(y > .5).sum(), (y <= .5).sum()]])[1])}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    sets = {name: json.loads((CASES / f).read_text()) for name, f in
            (("filtered_top", "filtered_30.json"), ("mid_ranked", "mid_ranked_30.json"))}
    meta = pd.DataFrame([{"pid": p, "set": n, "plddt": v["mean_plddt"]} for n, s in sets.items() for p, v in s.items()])
    t = pd.read_csv(CASES / "dark_ranked_layer22.csv")
    k = int(round(0.01 * len(t)))
    top = set(t[t["rank"] <= k].pid)
    rest = set(t[t["rank"] > k].pid.sample(3000, random_state=42))  # same sample as disorder_check.py
    out_csv = WORK / "out.csv"
    if not out_csv.exists():
        seqs = sequences(set(meta.pid) | top | rest)
        (WORK / "in.fasta").write_text("".join(f">{p}\n{s}\n" for p, s in seqs.items()))
        cmd = ["docker", "run", "--rm", "-v", f"{WORK}:/work", "-v", f"{ROOT / 'scripts'}:/scripts:ro",
               "-v", "obscuron-pip-cache:/root/.cache/pip", IMAGE, "bash", "-c",
               f"for i in 1 2 3 4 5 6; do ({PIP}) && break; echo install attempt $i failed; sleep 30; done; "
               f"python /scripts/metapredict_container.py /work/in.fasta /work/out.tmp && mv /work/out.tmp /work/out.csv"]
        subprocess.run(cmd, check=True)
    mp = pd.read_csv(out_csv)
    res = {"metapredict_version": str(mp.metapredict_version.iloc[0]), "cutoff": 0.5}

    d = meta.merge(mp, on="pid")
    a, b = d[d.set == "filtered_top"], d[d.set == "mid_ranked"]
    res["structure_sets"] = {"filtered_top": summary(a), "mid_ranked": summary(b), **compare(a, b),
                             "spearman_disorder_plddt_all60": float(stats.spearmanr(d.metapredict_disordered_share, d.plddt).statistic),
                             "filtered_top_low_plddt_not_disordered": int(((a.plddt < 50) & (a.metapredict_disordered_share < 0.3)).sum()),
                             "per_protein": d.round(4).to_dict("records")}
    w = mp[mp.pid.isin(top | rest)].copy()
    w["top"] = w.pid.isin(top)
    w["passes_filter"] = w.pid.isin(set(pd.read_csv(ROOT / "results" / "revision" / "filtered_ranking.csv").protein_id))
    res["ranking"] = {"top1pct": summary(w[w.top]), "rest_sample": summary(w[~w.top]), **compare(w[w.top], w[~w.top])}
    f = w[w.passes_filter]
    res["ranking_after_filter"] = {"top1pct": summary(f[f.top]), "rest_sample": summary(f[~f.top]), **compare(f[f.top], f[~f.top])}
    out = ROOT / "results" / "revision" / "metapredict_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "structure_sets"}, indent=1))
    print(json.dumps({k: v for k, v in res["structure_sets"].items() if k != "per_protein"}, indent=1))


if __name__ == "__main__":
    main()
