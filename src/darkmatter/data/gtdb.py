"""Generic GTDB release fetcher — parameterized by release tag, no hardcoded snapshot pair.

URL layout confirmed against https://data.gtdb.ecogenomic.org/releases/release232/232.0/ :
    releases/release{N}/{N}.0/bac120_taxonomy_r{N}.tsv.gz   (bacterial taxonomy, ~9MB)
    releases/release{N}/{N}.0/ar53_taxonomy_r{N}.tsv.gz     (archaeal taxonomy, ~300KB)
    releases/release{N}/{N}.0/bac120_metadata_r{N}.tsv.gz   (bacterial quality/rep-genome flags, ~275MB)
    releases/release{N}/{N}.0/ar53_metadata_r{N}.tsv.gz     (archaeal quality/rep-genome flags, ~7MB)
    releases/release{N}/{N}.0/MD5SUM.txt, FILE_DESCRIPTIONS.txt

NOTE: the metadata file extension is NOT stable across releases — R232
serves `bac120_metadata_r232.tsv.gz` but R207 serves
`bac120_metadata_r207.tar.gz` (confirmed against both live directory
listings). fetch_gtdb_release() tries `.tsv.gz` first and falls back to
`.tar.gz` on a 404.

`metadata_only=True` (default) fetches the above — all told a few hundred
MB per release, no sequence data. It does NOT fetch anything under
genomic_files_reps/ or genomic_files_all/: the representative-genome
protein FASTA alone (gtdb_proteins_aa_reps_r{N}.tar.gz) is ~123GB on
R232, and the full genome/nucleotide siblings are 170GB+ — those need a
deliberate decision (storage target, subset strategy) before pulling,
not a default flag.
"""

from __future__ import annotations

from pathlib import Path

import requests

from darkmatter.data.download import download, read_md5_list, verify_md5

BASE_URL = "https://data.gtdb.ecogenomic.org/releases"

TAXONOMY_FILES = ["bac120_taxonomy_r{n}.tsv.gz", "ar53_taxonomy_r{n}.tsv.gz"]
METADATA_BASENAMES = ["bac120_metadata_r{n}", "ar53_metadata_r{n}"]
METADATA_EXTS = [".tsv.gz", ".tar.gz"]
PROVENANCE_FILES = ["MD5SUM.txt", "FILE_DESCRIPTIONS.txt"]

# Representative-genome protein FASTA — deliberately NOT included in
# metadata_only or the default "full" fetch. ~123GB on R232. Call
# fetch_gtdb_reps_proteins() explicitly once storage/subset strategy is decided.
REPS_PROTEIN_FILE = "genomic_files_reps/gtdb_proteins_aa_reps_r{n}.tar.gz"


def _release_number(release: str) -> str:
    """'R207' -> '207'."""
    return release.lstrip("Rr")


def _fetch_files(release: str, out_dir: Path, base: str, templates: list[str]) -> list[Path]:
    fetched = []
    for template in templates:
        fname = template.format(n=_release_number(release))
        dest = out_dir / f"gtdb_{release}" / fname
        if not dest.exists():
            download(f"{base}/{fname}", dest)
        _verify(dest)
        fetched.append(dest)
    return fetched


def _verify(dest: Path) -> None:
    """Check a fetched file against the release's own MD5SUM.txt, when that list is present."""
    sums = dest.parent / "MD5SUM.txt"
    if sums.exists() and dest.name != "MD5SUM.txt":
        verify_md5(dest, read_md5_list(sums))


def _fetch_with_ext_fallback(release: str, out_dir: Path, base: str, basename_template: str, exts: list[str]) -> Path:
    n = _release_number(release)
    basename = basename_template.format(n=n)
    last_err: Exception | None = None
    for ext in exts:
        fname = basename + ext
        dest = out_dir / f"gtdb_{release}" / fname
        if dest.exists():
            _verify(dest)
            return dest
        try:
            download(f"{base}/{fname}", dest)
            _verify(dest)
            return dest
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                last_err = e
                continue
            raise
    raise last_err


def fetch_gtdb_release(release: str, out_dir: Path, metadata_only: bool = True) -> list[Path]:
    n = _release_number(release)
    base = f"{BASE_URL}/release{n}/{n}.0"
    fetched = _fetch_files(release, out_dir, base, TAXONOMY_FILES + PROVENANCE_FILES)
    for basename in METADATA_BASENAMES:
        fetched.append(_fetch_with_ext_fallback(release, out_dir, base, basename, METADATA_EXTS))
    if not metadata_only:
        fetched += _fetch_files(release, out_dir, base, [REPS_PROTEIN_FILE])
    return fetched


def fetch_gtdb_reps_proteins(release: str, out_dir: Path) -> list[Path]:
    """Explicit, separate call for the ~123GB rep-genome protein FASTA. Not part of any default fetch."""
    n = _release_number(release)
    base = f"{BASE_URL}/release{n}/{n}.0"
    return _fetch_files(release, out_dir, base, [REPS_PROTEIN_FILE])
