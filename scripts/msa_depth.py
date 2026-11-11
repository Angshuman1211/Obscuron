#!/usr/bin/env python
"""How many relatives do the top-ranked proteins have in large metagenomic databases? (TCBB revision)

Separates two explanations for the low ESMFold confidence at the filtered top of the ranking:
the proteins have relatives that ESM-2 did not learn (deep alignments; an alignment-based predictor
could fold them), or they have almost none (orphans). Alignments come from the public ColabFold
MMseqs2 server (mode "env": UniRef30 plus ColabFoldDB, which adds BFD, MGnify, MetaEuk, SMAG and
other environmental collections), the same alignments AlphaFold2 would use in ColabFold. Only the
public GTDB protein sequences are sent.

Sets: the filtered top 30 and the 30 mid-ranked comparison proteins (results/case_studies/
filtered_30.json, mid_ranked_30.json), the unfiltered top three (summary.json) and the PF02545
control (control.json). For each protein: number of aligned sequences in the UniRef and the
environmental alignment, and the number covering at least half of the query.

Resumable: each batch's result archive is kept in .cache/msa/ and not requested again.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/msa_depth.py
Writes results/revision/msa_depth.json and results/revision/msa/<protein>.a3m.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from case_studies import sequence  # noqa: E402

CASES = ROOT / "results" / "case_studies"
CACHE = ROOT / ".cache" / "msa"
A3M = ROOT / "results" / "revision" / "msa"
HOST = "https://api.colabfold.com"
HEADERS = {"User-Agent": "obscuron-tcbb-revision/1.0"}
BATCH = 10


def submit_and_fetch(batch: list[tuple[str, str]], tag: str) -> bytes:
    arch = CACHE / f"{tag}.tar.gz"
    if arch.exists():
        return arch.read_bytes()
    query = "".join(f">{101 + i}\n{s}\n" for i, (_, s) in enumerate(batch))
    delay = 30
    while True:
        try:
            r = requests.post(f"{HOST}/ticket/msa", data={"q": query, "mode": "env"}, headers=HEADERS, timeout=60)
            out = r.json()
            if out.get("status") in ("RATELIMIT", "MAINTENANCE", "UNKNOWN") or "id" not in out:
                raise RuntimeError(out.get("status", "no ticket"))
            tid = out["id"]
            while out.get("status") in ("PENDING", "RUNNING"):
                time.sleep(10)
                out = requests.get(f"{HOST}/ticket/{tid}", headers=HEADERS, timeout=60).json()
            if out.get("status") != "COMPLETE":
                raise RuntimeError(out.get("status"))
            r = requests.get(f"{HOST}/result/download/{tid}", headers=HEADERS, timeout=300)
            r.raise_for_status()
            break
        except (requests.RequestException, ValueError, RuntimeError) as e:
            print(f"  {tag}: {e}; retrying in {delay} s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
    tmp = arch.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(arch)
    return r.content


def split_a3m(text: str) -> dict[int, list[tuple[str, str]]]:
    """the server concatenates one alignment per query, separated by NUL; first record is the query"""
    out = {}
    for block in text.split("\x00"):
        recs, name, seq = [], None, []
        for line in block.splitlines():
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(seq)))
                name, seq = line[1:].strip(), []
            elif line.strip():
                seq.append(line.strip())
        if name is not None:
            recs.append((name, "".join(seq)))
        if recs and recs[0][0].isdigit():
            out[int(recs[0][0])] = recs
    return out


def depth(recs: list[tuple[str, str]]) -> tuple[int, int]:
    """aligned sequences (query excluded) and those covering >= 50% of the query columns"""
    qlen = sum(c.isupper() or c == "-" for c in recs[0][1])
    n, n50 = 0, 0
    for _, s in recs[1:]:
        match = [c for c in s if c.isupper() or c == "-"]  # lower case = insertions relative to the query
        cov = sum(c != "-" for c in match) / max(qlen, 1)
        n += 1
        n50 += cov >= 0.5
    return n, n50


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    A3M.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name, f in (("filtered_top", "filtered_30.json"), ("mid_ranked", "mid_ranked_30.json"),
                    ("unfiltered_top3", "summary.json"), ("control", "control.json")):
        for p, v in json.loads((CASES / f).read_text()).items():
            sets.setdefault(p, {"set": name, "plddt": v["mean_plddt"], "rank": v.get("rank_layer22")})
    pids = list(sets)
    seqs = {p: sequence(p) for p in pids}
    rows = []
    for b in range(0, len(pids), BATCH):
        batch = [(p, seqs[p]) for p in pids[b:b + BATCH]]
        tag = f"batch{b // BATCH:02d}_" + hashlib.sha1("".join(p for p, _ in batch).encode()).hexdigest()[:10]
        print(f"batch {b // BATCH + 1}/{-(-len(pids) // BATCH)}", flush=True)
        tf = tarfile.open(fileobj=io.BytesIO(submit_and_fetch(batch, tag)))
        files = {m.name: tf.extractfile(m).read().decode() for m in tf.getmembers() if m.name.endswith(".a3m")}
        uni = split_a3m(files.get("uniref.a3m", ""))
        env = split_a3m(next((v for k, v in files.items() if k.startswith("bfd")), ""))
        for i, (p, _) in enumerate(batch):
            u, e = uni.get(101 + i, [("q", seqs[p])]), env.get(101 + i, [("q", seqs[p])])
            (A3M / f"{p}.a3m").write_text("".join(f">{n}\n{s}\n" for n, s in u + e[1:]))
            (nu, nu50), (ne, ne50) = depth(u), depth(e)
            rows.append({"pid": p, **sets[p], "length": len(seqs[p]), "uniref": nu, "uniref_cov50": nu50,
                         "env": ne, "env_cov50": ne50, "total": nu + ne, "total_cov50": nu50 + ne50})
    d = pd.DataFrame(rows)
    a, b = d[d.set == "filtered_top"], d[d.set == "mid_ranked"]

    def summ(x):
        return {"n": int(len(x)), "total_median": float(x.total.median()), "total_cov50_median": float(x.total_cov50.median()),
                "uniref_median": float(x.uniref.median()), "env_median": float(x.env.median()),
                "share_lt10_cov50": float((x.total_cov50 < 10).mean()), "share_zero": float((x.total == 0).mean())}
    res = {"server": HOST, "mode": "env", "filtered_top": summ(a), "mid_ranked": summ(b),
           "mannwhitney_p_total_cov50": float(stats.mannwhitneyu(a.total_cov50, b.total_cov50).pvalue),
           "spearman_depth_plddt_all60": float(stats.spearmanr(pd.concat([a, b]).total_cov50, pd.concat([a, b]).plddt).statistic),
           "spearman_depth_plddt_filtered_top": float(stats.spearmanr(a.total_cov50, a.plddt).statistic),
           "per_protein": d.to_dict("records")}
    out = ROOT / "results" / "revision" / "msa_depth.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "per_protein"}, indent=1))
    print(d[d.set.isin(["unfiltered_top3", "control"])][["pid", "set", "plddt", "total", "total_cov50"]].to_string())


if __name__ == "__main__":
    main()
