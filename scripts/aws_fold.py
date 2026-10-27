#!/usr/bin/env python
"""ESMFold on a cloud GPU for the structure comparison (TCBB revision).

Runs on the EC2 instance started by scripts/aws_structure_run.py, with the same model, pinned
revision and settings as scripts/case_studies.py (ESM stem in fp16, trunk chunk size 64). Needs
torch >= 2.6, which loads the pinned pytorch_model.bin safely.

    python aws_fold.py seqs.fasta out/
Writes out/<protein>.pdb atomically (proteins already folded are skipped, so the job resumes after
an interruption), one line per protein in out/fold_log.jsonl, and out/DONE when all are folded.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

ESMFOLD = ("facebook/esmfold_v1", "75a3841ee059df2bf4d56688166c8fb459ddd97a")


def read_fasta(path: Path) -> dict[str, str]:
    seqs, pid = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            pid = line[1:].split()[0]
            seqs[pid] = ""
        elif pid:
            seqs[pid] += line.strip()
    return seqs


def main():
    fasta, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(fasta)
    todo = {p: s for p, s in seqs.items() if not (out / f"{p}.pdb").exists()}
    print(f"{len(seqs)} proteins, {len(todo)} to fold", flush=True)
    if todo:
        tok = AutoTokenizer.from_pretrained(ESMFOLD[0], revision=ESMFOLD[1])
        model = EsmForProteinFolding.from_pretrained(ESMFOLD[0], revision=ESMFOLD[1], low_cpu_mem_usage=True)
        model.esm = model.esm.half()
        model = model.cuda().eval()
        model.trunk.set_chunk_size(64)
        with open(out / "fold_log.jsonl", "a") as log:
            for pid, s in todo.items():
                t0 = time.time()
                with torch.inference_mode():
                    ids = tok([s], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                    o = model(ids)
                pdb = model.output_to_pdb(o)[0]
                tmp = out / f"{pid}.pdb.tmp"
                tmp.write_text(pdb)
                os.replace(tmp, out / f"{pid}.pdb")
                plddt = o["plddt"][0, :, 1].float().cpu().numpy()
                rec = {"pid": pid, "length": len(s), "mean_plddt": float(plddt.mean() * (100 if plddt.max() <= 1 else 1)),
                       "fold_seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
                torch.cuda.empty_cache()
    (out / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
