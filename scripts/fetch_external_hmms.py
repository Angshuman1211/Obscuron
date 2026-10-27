#!/usr/bin/env python
"""Fetch the external HMM libraries used for the cross-database checks (TCBB revision).

  ncbifam  : NCBIfam / TIGRFAM (NCBI PGAP HMMs, current release) - a second curated family resource
  antifam  : AntiFam (EBI) - profiles of known spurious open reading frames
  nmpfams  : NMPFamsDB (Baltoumas et al. 2023) - >100,000 novel protein families from metagenomes

None of these publishers lists checksums next to the file, so the SHA-256 of each download is
recorded in data/raw/external/manifest.json. Archives are unpacked with tarfile's "data" filter,
which refuses absolute paths, links outside the target and other unsafe members.

    .venv/Scripts/python.exe scripts/fetch_external_hmms.py
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "raw" / "external"
SOURCES = {
    "ncbifam": "https://ftp.ncbi.nlm.nih.gov/hmm/current/hmm_PGAP.HMM.tgz",
    "antifam": "https://ftp.ebi.ac.uk/pub/databases/Pfam/AntiFam/current/Antifam.tar.gz",
    "nmpfams": "https://bib.fleming.gr/NMPFamsDB/download_files/pHMMs_all.tgz",
}


def fetch(name: str, url: str) -> dict:
    d = EXT / name
    d.mkdir(parents=True, exist_ok=True)
    arch = d / url.rsplit("/", 1)[-1]
    if not arch.exists():
        part = arch.with_suffix(arch.suffix + ".part")
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        with requests.get(url, stream=True, timeout=(15, 600), headers=headers) as r:
            r.raise_for_status()
            mode = "ab" if done and r.status_code == 206 else "wb"  # resume only if the server honours Range
            with open(part, mode) as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        part.rename(arch)
    h = hashlib.sha256()
    with open(arch, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    out = d / "unpacked"
    if not out.exists():
        out.mkdir()
        with tarfile.open(arch) as t:
            t.extractall(out, filter="data")
    hmms = sorted(p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in (".hmm", ".lib", ".hmms"))
    return {"url": url, "file": arch.name, "bytes": arch.stat().st_size, "sha256": h.hexdigest(), "n_hmm_files": len(hmms)}


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    man = {k: fetch(k, u) for k, u in SOURCES.items()}
    (EXT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
