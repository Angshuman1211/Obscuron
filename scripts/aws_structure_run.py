#!/usr/bin/env python
"""Fold the structure-comparison proteins on an AWS GPU instance (TCBB revision).

Moves the ESMFold step of scripts/run_structure_comparison.py to one EC2 g5.xlarge (A10G, 24 GB),
where the model fits in GPU memory. Only the proteins without a PDB file in results/case_studies/
are sent. The structures are copied back into results/case_studies/, after which the local runner
skips them and only runs the Foldseek searches and the comparison.

Every step is idempotent and its state is kept in .cache/aws_run/state.json, so the command can be
run again after a lost connection, a crash or a reboot and continues where it stopped: it reuses a
running instance, does not re-upload or re-fold finished proteins, and copies back only new files.
The instance shuts itself down (and is terminated) after MAX_HOURS even if this machine never
reconnects, and the run removes the instance, security group and key pair when it is done.

Credentials are read from the environment only (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY); they
are never written to disk.

    .venv/Scripts/python.exe scripts/aws_structure_run.py            # run or resume
    .venv/Scripts/python.exe scripts/aws_structure_run.py --status
    .venv/Scripts/python.exe scripts/aws_structure_run.py --cleanup  # terminate and delete everything
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CASES = ROOT / "results" / "case_studies"
RUN = ROOT / ".cache" / "aws_run"
STATE = RUN / "state.json"
REGION = "us-east-1"
TYPE = "g5.xlarge"
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id"
NAME = "obscuron-fold"
MAX_HOURS = 4  # dead-man switch: the instance powers off (and terminates) after this long
RANKED = ROOT / "results" / "case_studies" / "dark_ranked_layer22.csv"
SETS = [dict(filtered=ROOT / "results" / "revision" / "filtered_ranking.csv"), dict(mid_ranked=True)]
N = 30
REMOTE = "/home/ubuntu/fold"
SETUP = ("set -e; source /opt/pytorch/bin/activate 2>/dev/null || true; "
         "pip install -q 'transformers==5.14.1' accelerate >/dev/null; touch ~/setup_ok")


def log(msg: str):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def aws(*args: str, check=True) -> str:
    env = dict(os.environ, AWS_DEFAULT_REGION=REGION, MSYS_NO_PATHCONV="1")
    r = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise RuntimeError(f"aws {' '.join(args[:2])}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save(st: dict):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE)


def ssh_args(st: dict) -> list[str]:
    return ["-i", str(RUN / f"{NAME}.pem"), "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={RUN / 'known_hosts'}", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30"]


def ssh(st: dict, cmd: str, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run(["ssh", *ssh_args(st), f"ubuntu@{st['ip']}", cmd], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr.strip()[-300:]}")
    return r


def export_sequences() -> Path:
    from case_studies import select, sequence
    pids = []
    for kw in SETS:
        pids += [p for p in select(RANKED, N, **kw).pid if not (CASES / f"{p}.pdb").exists()]
    fa = RUN / "seqs.fasta"
    fa.write_text("".join(f">{p}\n{sequence(p)}\n" for p in dict.fromkeys(pids)))
    log(f"{len(pids)} proteins without a structure written to {fa.name}")
    return fa


def ensure_network(st: dict):
    if not (RUN / f"{NAME}.pem").exists():
        aws("ec2", "delete-key-pair", "--key-name", NAME, check=False)
        key = json.loads(aws("ec2", "create-key-pair", "--key-name", NAME, "--key-type", "ed25519"))["KeyMaterial"]
        (RUN / f"{NAME}.pem").write_text(key)
        os.chmod(RUN / f"{NAME}.pem", 0o600)
    if "sg" not in st:
        vpc = json.loads(aws("ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true"))["Vpcs"][0]["VpcId"]
        found = json.loads(aws("ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={NAME}",
                               f"Name=vpc-id,Values={vpc}"))["SecurityGroups"]
        st["sg"] = found[0]["GroupId"] if found else json.loads(aws(
            "ec2", "create-security-group", "--group-name", NAME, "--vpc-id", vpc,
            "--description", "ssh for the obscuron structure run"))["GroupId"]
        save(st)
    ip = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=20).read().decode().strip()
    if st.get("my_ip") != ip:  # the home IP can change after a reconnect
        aws("ec2", "authorize-security-group-ingress", "--group-id", st["sg"], "--protocol", "tcp", "--port", "22",
            "--cidr", f"{ip}/32", check=False)
        st["my_ip"] = ip
        save(st)


def ensure_instance(st: dict):
    if "instance" in st:
        d = json.loads(aws("ec2", "describe-instances", "--instance-ids", st["instance"], check=False) or "{}")
        inst = [i for r in d.get("Reservations", []) for i in r["Instances"]]
        if inst and inst[0]["State"]["Name"] in ("pending", "running"):
            if inst[0]["State"]["Name"] == "pending":
                aws("ec2", "wait", "instance-running", "--instance-ids", st["instance"])
                inst = [json.loads(aws("ec2", "describe-instances", "--instance-ids", st["instance"]))
                        ["Reservations"][0]["Instances"][0]]
            st["ip"] = inst[0].get("PublicIpAddress")
            save(st)
            return
        log("previous instance is gone; starting a new one (folded proteins are kept locally)")
        st.pop("instance", None)
        st.pop("setup_ok", None)
    ami = json.loads(aws("ssm", "get-parameter", "--name", AMI_PARAM))["Parameter"]["Value"]
    user_data = f"#!/bin/bash\nshutdown -h +{MAX_HOURS * 60}\n"
    r = json.loads(aws(
        "ec2", "run-instances", "--image-id", ami, "--instance-type", TYPE, "--count", "1",
        "--key-name", NAME, "--security-group-ids", st["sg"],
        "--instance-initiated-shutdown-behavior", "terminate", "--user-data", user_data,
        "--block-device-mappings", '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]',
        "--tag-specifications", f'ResourceType=instance,Tags=[{{Key=Name,Value={NAME}}}]'))
    st["instance"] = r["Instances"][0]["InstanceId"]
    st["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(st)
    log(f"started {TYPE} {st['instance']}")
    aws("ec2", "wait", "instance-running", "--instance-ids", st["instance"])
    ensure_instance(st)


def wait_ssh(st: dict):
    for _ in range(60):
        if ssh(st, "true", check=False).returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError("instance did not accept ssh within 10 min")


def fetch(st: dict) -> int:
    """copy back structures not yet present locally; returns how many arrived"""
    have = {p.stem for p in CASES.glob("*.pdb")}
    remote = ssh(st, f"ls {REMOTE}/out 2>/dev/null", check=False).stdout.split()
    new = [f for f in remote if f.endswith(".pdb") and f[:-4] not in have]
    if not new:
        return 0
    tmp = RUN / "incoming"
    tmp.mkdir(exist_ok=True)
    subprocess.run(["scp", *ssh_args(st), *[f"ubuntu@{st['ip']}:{REMOTE}/out/{f}" for f in new], str(tmp)],
                   check=True, capture_output=True)
    subprocess.run(["scp", *ssh_args(st), f"ubuntu@{st['ip']}:{REMOTE}/out/fold_log.jsonl", str(RUN)],
                   check=False, capture_output=True)
    for f in new:  # move one by one; a file is only in results/ once it is complete
        os.replace(tmp / f, CASES / f)
    return len(new)


def cleanup(st: dict):
    if "instance" in st:
        aws("ec2", "terminate-instances", "--instance-ids", st["instance"], check=False)
        aws("ec2", "wait", "instance-terminated", "--instance-ids", st["instance"], check=False)
        log(f"terminated {st['instance']}")
    if "sg" in st:
        aws("ec2", "delete-security-group", "--group-id", st["sg"], check=False)
    aws("ec2", "delete-key-pair", "--key-name", NAME, check=False)
    (RUN / f"{NAME}.pem").unlink(missing_ok=True)
    st.clear()
    st["cleaned_up"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    st = load()
    if args.status:
        print(json.dumps(st, indent=1))
        return
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        sys.exit("set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment first")
    if args.cleanup:
        cleanup(st)
        return

    fa = export_sequences()
    if not fa.read_text().strip():
        log("every protein already has a structure; nothing to fold")
        cleanup(st)
        return
    ensure_network(st)
    ensure_instance(st)
    wait_ssh(st)
    if not st.get("setup_ok"):
        log("installing transformers on the instance")
        ssh(st, SETUP)
        st["setup_ok"] = True
        save(st)
    ssh(st, f"mkdir -p {REMOTE}/out")
    subprocess.run(["scp", *ssh_args(st), str(fa), str(ROOT / "scripts" / "aws_fold.py"), f"ubuntu@{st['ip']}:{REMOTE}/"],
                   check=True, capture_output=True)
    ssh(st, f"rm -f {REMOTE}/out/DONE; pgrep -f aws_fold.py >/dev/null || (cd {REMOTE} && "
            f"source /opt/pytorch/bin/activate 2>/dev/null; nohup python aws_fold.py seqs.fasta out > fold.log 2>&1 &)")
    log("folding on the instance")

    while True:
        try:
            n = fetch(st)
            if n:
                log(f"copied back {n} structures")
            done = ssh(st, f"test -f {REMOTE}/out/DONE", check=False).returncode
            alive = ssh(st, "pgrep -f aws_fold.py >/dev/null", check=False).returncode
            if 255 in (done, alive):  # ssh's own exit code: the connection, not the job, failed
                raise OSError("ssh connection failed")
            if done == 0:
                fetch(st)
                break
            if alive != 0:
                tail = ssh(st, f"tail -5 {REMOTE}/fold.log", check=False).stdout
                raise RuntimeError(f"fold job stopped without finishing:\n{tail}")
        except (subprocess.CalledProcessError, OSError) as e:  # a dropped connection: wait and poll again
            log(f"connection problem ({type(e).__name__}); retrying")
        time.sleep(60)
    log("all structures copied back")
    cleanup(st)


if __name__ == "__main__":
    main()
