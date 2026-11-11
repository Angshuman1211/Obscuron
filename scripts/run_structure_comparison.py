#!/usr/bin/env python
"""Resumable driver for the structure comparison (TCBB revision).

Runs three stages in order and can be stopped at any point, by a crash, a lost Wi-Fi connection,
sleep or a shutdown, and started again with the same command:

  1. filtered_30  ESMFold + Foldseek for the top 30 proteins of the filtered ranking
  2. mid_ranked_30  the same for 30 random still-dark proteins from ranks 15,000 to 20,000
  3. compare  scripts/structure_comparison.py

Checkpoints are the result files themselves: one PDB per folded protein and one Foldseek JSON per
searched protein (written atomically by scripts/case_studies.py), and each stage's summary file. A
restart therefore skips finished stages and, inside a stage, finished proteins. A lock file stops
a second copy from running at the same time. Progress is logged to .cache/structure_run/.

    .venv/Scripts/python.exe scripts/run_structure_comparison.py               # run or resume
    .venv/Scripts/python.exe scripts/run_structure_comparison.py --status      # show progress
    .venv/Scripts/python.exe scripts/run_structure_comparison.py --install-autostart
    .venv/Scripts/python.exe scripts/run_structure_comparison.py --remove-autostart

--install-autostart puts a small .cmd file in the current user's Startup folder, so the run resumes
by itself after the laptop restarts and the user logs in. It needs no administrator rights, and the
runner deletes the file once all stages are done.
"""
from __future__ import annotations

import argparse
import json
import msvcrt
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results" / "case_studies"
RUN = ROOT / ".cache" / "structure_run"
STATE = RUN / "state.json"
LOG = RUN / "runner.log"
STARTUP = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
AUTOSTART = STARTUP / "obscuron_structure_resume.cmd"
RANKED = "results/case_studies/dark_ranked_layer22.csv"
N = 30  # proteins per set; --n changes it, and the summary files are named after it
ATTEMPTS = 5  # per stage and launch; the next launch starts the count again


def stages(n: int) -> list:
    comp = "structure_comparison.json" if n == 30 else f"structure_comparison_{n}.json"
    return [
        (f"filtered_{n}", CASES / f"filtered_{n}.json",
         ["scripts/case_studies.py", "--ranked", RANKED, "--filtered", "results/revision/filtered_ranking.csv",
          "--n", str(n), "--out", f"filtered_{n}.json"]),
        (f"mid_ranked_{n}", CASES / f"mid_ranked_{n}.json",
         ["scripts/case_studies.py", "--ranked", RANKED, "--mid-ranked", "--n", str(n), "--out", f"mid_ranked_{n}.json"]),
        ("compare" if n == 30 else f"compare_{n}", ROOT / "results" / "revision" / comp,
         ["scripts/structure_comparison.py", "--filtered", f"filtered_{n}.json", "--mid", f"mid_ranked_{n}.json",
          "--out", comp]),
    ]


STAGES = stages(N)


def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_state(state: dict):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


def progress() -> dict:
    pdbs = {p.stem for p in CASES.glob("*.pdb")}
    searched = {p.stem for p in (CASES / "foldseek").glob("*.json")}
    out = {}
    for name, done_file, _ in STAGES:
        out[name] = "done" if done_file.exists() else "pending"
    out["proteins_folded"] = len(pdbs)
    out["proteins_searched"] = len(searched)
    return out


def lock():
    """hold an exclusive lock on a file for the life of this process; the OS releases it on exit,
    crash or shutdown, so a stale lock can never block a restart"""
    fh = open(RUN / "lock", "a+")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return None
    return fh


def install_autostart():
    STARTUP.mkdir(parents=True, exist_ok=True)
    AUTOSTART.write_text(
        "@echo off\r\n"
        "rem Resumes the Obscuron structure comparison after a restart; deleted automatically when done.\r\n"
        f'cd /d "{ROOT}"\r\n'
        f'start "obscuron structure run" /min "{sys.executable}" "{ROOT / "scripts" / "run_structure_comparison.py"}"\r\n')
    log(f"autostart installed: {AUTOSTART}")


def remove_autostart():
    if AUTOSTART.exists():
        AUTOSTART.unlink()
        log(f"autostart removed: {AUTOSTART}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--install-autostart", action="store_true")
    ap.add_argument("--remove-autostart", action="store_true")
    ap.add_argument("--n", type=int, default=N, help="proteins per set (default 30)")
    args = ap.parse_args()
    global STAGES
    STAGES = stages(args.n)
    RUN.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps({"progress": progress(), "state": load_state(), "autostart": AUTOSTART.exists()}, indent=1))
        return
    if args.install_autostart:
        install_autostart()
        return
    if args.remove_autostart:
        remove_autostart()
        return

    held = lock()
    if held is None:
        print("another copy of the runner is already working; nothing to do")
        return
    state = load_state()
    state["launches"] = state.get("launches", 0) + 1
    save_state(state)
    log(f"launch {state['launches']}: {json.dumps(progress())}")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")

    for name, done_file, cmd in STAGES:
        if done_file.exists():
            continue
        for attempt in range(1, ATTEMPTS + 1):
            log(f"{name}: attempt {attempt}")
            st = state.setdefault(name, {})
            st.update(started=datetime.now().isoformat(timespec="seconds"), attempts=st.get("attempts", 0) + 1)
            save_state(state)
            with open(RUN / f"{name}.log", "a", encoding="utf-8") as out:
                rc = subprocess.call([sys.executable, *cmd], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
            if rc == 0 and done_file.exists():
                st.update(finished=datetime.now().isoformat(timespec="seconds"), last_error=None)
                save_state(state)
                log(f"{name}: done ({json.dumps(progress())})")
                break
            st["last_error"] = f"exit code {rc} (see {name}.log)"
            save_state(state)
            log(f"{name}: failed with exit code {rc}; finished proteins are kept, retrying in 2 min")
            time.sleep(120)
        else:
            log(f"{name}: gave up after {ATTEMPTS} attempts in this launch; run the command again to resume")
            sys.exit(1)

    state["all_done"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    log("all stages done")
    remove_autostart()


if __name__ == "__main__":
    main()
