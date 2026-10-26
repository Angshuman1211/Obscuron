"""Generic Pfam-A release fetcher — parameterized by release version.

URL layout confirmed against https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/ :
    Pfam-A.clans.tsv.gz   (family -> clan/description list, small, few MB)
    Pfam-A.hmm.gz         (full HMM library, large, hundreds of MB+)

`metadata_only=True` fetches just Pfam-A.clans.tsv.gz — the family list
needed to diff "which families exist in release X vs release Y", without
pulling the full HMM library (only needed later for actually running
hmmscan against historical vs current families).
"""

from __future__ import annotations

from pathlib import Path

from darkmatter.data.download import download, read_md5_list, verify_md5

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases"
CHECKSUM_FILE = "md5_checksums"  # published by EBI next to each release's files

METADATA_FILE = "Pfam-A.clans.tsv.gz"
HMM_FILE = "Pfam-A.hmm.gz"


def fetch_pfam_release(release: str, out_dir: Path, metadata_only: bool = True) -> list[Path]:
    base = f"{BASE_URL}/Pfam{release}"
    filenames = [METADATA_FILE] if metadata_only else [METADATA_FILE, HMM_FILE]

    sums_path = out_dir / f"pfam_{release}" / CHECKSUM_FILE
    if not sums_path.exists():
        download(f"{base}/{CHECKSUM_FILE}", sums_path)
    expected = read_md5_list(sums_path)

    fetched = []
    for fname in filenames:
        dest = out_dir / f"pfam_{release}" / fname
        if not dest.exists():
            download(f"{base}/{fname}", dest)
        verify_md5(dest, expected)
        fetched.append(dest)
    return fetched
