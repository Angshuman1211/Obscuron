#!/usr/bin/env python
"""Is the top of the ranking disordered? An ESM-independent check (TCBB revision).

ESMFold is built on ESM-2, the language model behind the novelty score, so its low confidence on
the top of the ranking is not independent evidence. This check uses two classic predictors that
need only the sequence and share nothing with ESM-2:

  * the charge-hydropathy rule of Uversky et al. (2000): a protein is predicted natively unfolded
    when its mean normalised Kyte-Doolittle hydropathy <H> is below (|<R>| + 1.151) / 2.785, with
    <R> the mean net charge at pH 7 (K, R = +1; D, E = -1);
  * FoldIndex (Prilusky et al. 2005), the same quantity 2.785<H> - |<R>| - 1.151 over sliding
    51-residue windows; a residue is disordered when its window scores below zero, and a protein's
    disorder is the share of such residues.

(metapredict would be the stronger predictor but has no Windows build; its compiled extension
needs a C++ tool chain.)

Applied to the two structure-comparison sets (results/case_studies/filtered_30.json and
mid_ranked_30.json), and to the top 1% of the 34,138 dark queries against 3,000 drawn from the rest
(seed 42), before and after the artefact filter of scripts/top_ranked_checks.py.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/disorder_check.py
Writes results/revision/disorder_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from top_ranked_checks import PROC  # noqa: E402

CASES = ROOT / "results" / "case_studies"
WINDOW = 51
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
CHARGE = dict(K=1, R=1, D=-1, E=-1)


def sequences(pids: set[str]) -> dict[str, str]:
    samp = pd.read_csv(PROC / "embedding_sample.csv").set_index("protein_id")
    from Bio import SeqIO
    out = {}
    for acc in samp.loc[sorted(pids), "genome_accession"].unique():
        for r in SeqIO.parse(PROC / "panel_proteins" / f"{acc}_protein.faa", "fasta"):
            if r.id in pids:
                out[r.id] = str(r.seq).rstrip("*")
    return out


def foldindex(seq: str) -> tuple[float, float]:
    """whole-sequence FoldIndex (below zero = Uversky 'natively unfolded') and windowed disorder share"""
    s = [a for a in seq if a in KD]
    h = np.array([(KD[a] + 4.5) / 9.0 for a in s])  # Kyte-Doolittle scaled to [0, 1]
    q = np.array([CHARGE.get(a, 0) for a in s], dtype=float)
    whole = 2.785 * h.mean() - abs(q.mean()) - 1.151
    if len(s) <= WINDOW:
        return float(whole), float(whole < 0)
    ch, cq = np.concatenate([[0], np.cumsum(h)]), np.concatenate([[0], np.cumsum(q)])
    win = 2.785 * (ch[WINDOW:] - ch[:-WINDOW]) / WINDOW - np.abs(cq[WINDOW:] - cq[:-WINDOW]) / WINDOW - 1.151
    half = WINDOW // 2  # score each residue by the window centred on it; ends take the nearest window
    per_res = np.concatenate([np.full(half, win[0]), win, np.full(len(s) - len(win) - half, win[-1])])
    return float(whole), float((per_res < 0).mean())


def predict(seqs: dict[str, str]) -> pd.DataFrame:
    rows = [{"pid": p, **dict(zip(("foldindex", "disorder_share"), foldindex(s)))} for p, s in seqs.items()]
    d = pd.DataFrame(rows)
    d["uversky_unfolded"] = d.foldindex < 0
    return d


def summary(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "foldindex_median": float(d.foldindex.median()),
            "uversky_unfolded_share": float(d.uversky_unfolded.mean()),
            "disorder_share_median": float(d.disorder_share.median()),
            "mostly_disordered_share": float((d.disorder_share > 0.5).mean())}


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    return {"foldindex_mannwhitney_p": float(stats.mannwhitneyu(a.foldindex, b.foldindex).pvalue),
            "disorder_share_mannwhitney_p": float(stats.mannwhitneyu(a.disorder_share, b.disorder_share).pvalue),
            "uversky_fisher_p": float(stats.fisher_exact(
                [[a.uversky_unfolded.sum(), (~a.uversky_unfolded).sum()],
                 [b.uversky_unfolded.sum(), (~b.uversky_unfolded).sum()]])[1])}


def main():
    res = {"window": WINDOW}

    # structure-comparison sets, with ESMFold pLDDT alongside
    sets = {name: json.loads((CASES / f).read_text()) for name, f in
            (("filtered_top", "filtered_30.json"), ("mid_ranked", "mid_ranked_30.json"))}
    d = predict(sequences({p for s in sets.values() for p in s}))
    meta = pd.DataFrame([{"pid": p, "set": name, "plddt": v["mean_plddt"]} for name, s in sets.items() for p, v in s.items()])
    d = meta.merge(d, on="pid")
    a, b = d[d.set == "filtered_top"], d[d.set == "mid_ranked"]
    res["structure_sets"] = {
        "filtered_top": summary(a), "mid_ranked": summary(b), **compare(a, b),
        "spearman_foldindex_plddt_all60": float(stats.spearmanr(d.foldindex, d.plddt).statistic),
        "spearman_foldindex_plddt_filtered_top": float(stats.spearmanr(a.foldindex, a.plddt).statistic),
        # low ESMFold confidence that the ESM-independent predictor does not call disordered
        "filtered_top_low_plddt_not_disordered": int(((a.plddt < 50) & (a.disorder_share < 0.3)).sum()),
        "per_protein": d.round(4).to_dict("records")}

    # whole ranking: top 1% against a sample of the rest, before and after the artefact filter
    t = pd.read_csv(CASES / "dark_ranked_layer22.csv")
    k = int(round(0.01 * len(t)))  # same top-1% definition as scripts/top_ranked_checks.py
    top = set(t[t["rank"] <= k].pid)
    rest = set(t[t["rank"] > k].pid.sample(3000, random_state=42))
    w = predict(sequences(top | rest))
    w["top"] = w.pid.isin(top)
    w["passes_filter"] = w.pid.isin(set(pd.read_csv(ROOT / "results" / "revision" / "filtered_ranking.csv").protein_id))
    res["ranking"] = {"top1pct": summary(w[w.top]), "rest_sample": summary(w[~w.top]), **compare(w[w.top], w[~w.top])}
    f = w[w.passes_filter]
    res["ranking_after_filter"] = {"top1pct": summary(f[f.top]), "rest_sample": summary(f[~f.top]),
                                   **compare(f[f.top], f[~f.top])}

    out = ROOT / "results" / "revision" / "disorder_check.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    tmp.replace(out)
    print(json.dumps({k: v for k, v in res.items() if k != "structure_sets"}, indent=1))
    print(json.dumps({k: v for k, v in res["structure_sets"].items() if k != "per_protein"}, indent=1))


if __name__ == "__main__":
    main()
