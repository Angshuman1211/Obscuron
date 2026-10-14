#!/usr/bin/env python
"""Audit of the retrospective labels (Review II, comment 2).

The benchmark labels come from two hard calls at the Pfam gathering (GA) threshold:
a protein is dark at T0 if it has no Pfam-35 hit at GA, and a positive if it is dark and
hits a Pfam-37 family that did not exist in Pfam 35. A hard threshold produces borderline
cases on both sides, so this script re-scans the 34,138 dark-query proteins (and 3,000
characterised proteins as a control) against both libraries WITHOUT the GA cut-off
(E <= 1e-3) and records, per protein, the best bit-score margin over each family's own GA.

Outputs (results/reviewer/):
  label_audit_hits.csv       one row per scanned protein: best Pfam-35 and Pfam-37 margins
  label_audit_summary.json   counts used in the report
A margin >= 0 means the sequence score reaches GA. Margins are on the sequence bit score,
so a few proteins labelled by the domain-level GA rule can sit slightly off zero; the
characterised control shows how large that disagreement is.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/label_audit.py [--cpus 12]
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyhmmer
from pyhmmer.easel import Alphabet, DigitalSequenceBlock, SequenceFile

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "gtdb_R207"
OUT = ROOT / "results" / "reviewer"
OUT.mkdir(parents=True, exist_ok=True)
PFAM35 = ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.hmm.gz"
PFAM37 = ROOT / "data" / "raw" / "pfam_37.0" / "Pfam-A.hmm"
EVALUE = 1e-3


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def load_sequences(df: pd.DataFrame, alphabet) -> DigitalSequenceBlock:
    wanted = set(df["protein_id"])
    kept = []
    for acc in df["genome_accession"].unique():
        p = PROC / "panel_proteins" / f"{acc}_protein.faa"
        with SequenceFile(p, digital=True, alphabet=alphabet) as sf:
            kept.extend(s for s in sf if dec(s.name) in wanted)
    return DigitalSequenceBlock(alphabet, kept)


def scan(seqs, hmm_path, new_ids: set[str] | None, cpus: int):
    """best[name] = dict(margin, acc, new_margin, new_acc) over all HMMs at E<=EVALUE."""
    best: dict[str, dict] = {}
    t0 = time.time()
    n = 0
    with pyhmmer.plan7.HMMFile(hmm_path) as hf:
        for top in pyhmmer.hmmsearch(hf, seqs, E=EVALUE, cpus=cpus):
            acc = dec(top.query.accession)
            ga = top.query.cutoffs.gathering
            if ga is None:
                continue
            fam = acc.split(".")[0]
            for hit in top:
                m = hit.score - ga[0]
                b = best.setdefault(dec(hit.name), {"margin": -1e9, "acc": "", "new_margin": -1e9, "new_acc": ""})
                if m > b["margin"]:
                    b["margin"], b["acc"] = m, acc
                if new_ids is not None and fam in new_ids and m > b["new_margin"]:
                    b["new_margin"], b["new_acc"] = m, acc
            n += 1
            if n % 2000 == 0:
                print(f"  {n} HMMs, {time.time() - t0:.0f}s", flush=True)
    return best


def clans(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            out[p[0]] = (p[1], p[3] if len(p) > 3 else "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--n-control", type=int, default=3000)
    args = ap.parse_args()

    samp = pd.read_csv(PROC / "embedding_sample.csv")
    q = samp[samp.category.isin(["positive", "dark_negative"])]
    ctrl = samp[samp.category == "characterised_at_t0"].sample(args.n_control, random_state=42)
    df = pd.concat([q, ctrl]).reset_index(drop=True)
    alphabet = Alphabet.amino()
    seqs = load_sequences(df, alphabet)
    print(f"{len(seqs)} sequences loaded", flush=True)

    c35 = clans(ROOT / "data" / "raw" / "pfam_35.0" / "Pfam-A.clans.tsv.gz")
    c37 = clans(ROOT / "data" / "raw" / "pfam_37.0" / "Pfam-A.clans.tsv.gz")
    new_ids = set(c37) - set(c35)

    print("Pfam-35 scan", flush=True)
    b35 = scan(seqs, PFAM35, None, args.cpus)
    print("Pfam-37 scan", flush=True)
    b37 = scan(seqs, PFAM37, new_ids, args.cpus)

    import pickle
    with open(OUT / "label_audit_raw.pkl", "wb") as fh:
        pickle.dump({"b35": b35, "b37": b37}, fh)

    rows = []
    for _, r in df.iterrows():
        a, b = b35.get(r.protein_id), b37.get(r.protein_id)
        rows.append({
            "protein_id": r.protein_id, "category": r.category,
            "m35": a["margin"] if a else np.nan, "acc35": a["acc"] if a else "",
            "m37": b["margin"] if b else np.nan, "acc37": b["acc"] if b else "",
            "m37_new": b["new_margin"] if b and b["new_margin"] > -1e8 else np.nan,
            "acc37_new": b["new_acc"] if b else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "label_audit_hits.csv", index=False)

    S: dict = {"n_scanned": int(len(out)), "evalue_cutoff": EVALUE, "n_new_families_pfam37": len(new_ids)}
    pos = out[out.category == "positive"]
    dark = out[out.category == "dark_negative"]
    chr_ = out[out.category == "characterised_at_t0"]
    S["control_characterised"] = {"n": int(len(chr_)), "frac_margin35_ge0": float((chr_.m35 >= 0).mean())}
    for name, g in (("positive", pos), ("still_dark", dark)):
        S[name] = {
            "n": int(len(g)),
            "frac_no_pfam35_hit_at_E1e-3": float(g.m35.isna().mean()),
            "frac_pfam35_margin_ge0_by_sequence_score": float((g.m35 >= 0).mean()),
            "frac_pfam35_within_minus10_of_GA": float(((g.m35 < 0) & (g.m35 >= -10)).mean()),
            "frac_pfam35_subthreshold_hit": float(((g.m35 < 0)).mean()),
            "frac_new_family_hit_margin_ge0": float((g.m37_new >= 0).mean()),
            "frac_new_family_hit_subthreshold": float(((g.m37_new < 0)).mean()),
            "frac_new_family_within_minus10_below_GA": float(((g.m37_new < 0) & (g.m37_new >= -10)).mean()),
        }
    # strength of the positive label
    m = pos.m37_new.dropna()
    S["positive_new_family_margin"] = {
        "n": int(len(m)), "median": float(m.median()), "p10": float(m.quantile(0.1)),
        "frac_margin_lt_5_bits": float((m < 5).mean()), "frac_margin_lt_10_bits": float((m < 10).mean())}
    # what the new families are
    fam = pos.acc37_new.str.split(".").str[0]
    inclan, duf = [], []
    for f in fam.dropna():
        if not f:
            continue
        clan37 = c37.get(f, ("", ""))[0]
        inclan.append(bool(clan37) and any(c35[x][0] == clan37 for x in c35 if c35[x][0]))
        duf.append(c37.get(f, ("", ""))[1].upper().startswith(("DUF", "UPF")))
    clan35_ids = {v[0] for v in c35.values() if v[0]}
    S["positive_new_families"] = {
        "n_distinct_families": int(fam.replace("", np.nan).dropna().nunique()),
        "frac_proteins_in_clan_that_already_existed_in_pfam35": float(
            np.mean([bool(c37.get(f, ("", ""))[0]) and c37[f][0] in clan35_ids for f in fam if f])),
        "frac_proteins_in_any_clan": float(np.mean([bool(c37.get(f, ("", ""))[0]) for f in fam if f])),
        "frac_proteins_duf_or_upf": float(np.mean(duf)),
    }
    # stringent label sets used for the robustness re-evaluation
    out["stringent_positive"] = (out.category == "positive") & (out.m37_new >= 10) & (out.m35.isna() | (out.m35 < -10))
    out["stringent_negative"] = (out.category == "dark_negative") & (out.m35.isna() | (out.m35 < -10)) & (
        out.m37_new.isna() | (out.m37_new < -10))
    out.to_csv(OUT / "label_audit_hits.csv", index=False)
    S["stringent"] = {"positives": int(out.stringent_positive.sum()), "negatives": int(out.stringent_negative.sum())}
    (OUT / "label_audit_summary.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
