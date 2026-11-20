"""Execute the exact two new PACS configurations, one fold per process."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from tools.pacs_causal.queue import DATA, DOMAINS, atomic_json, query, say


ROOT = Path("/home/hooasia/lgg/projects/dicm-pacs-b8-a16-20261120")
A8 = ROOT.parent / "dicm-pacs-causal-validation-20261119"
PROJECT = Path(__file__).resolve().parents[2]
GROUPS = {"B8":0, "A16":1}
MIN_MEMORY = 30720
MIN_DISK = 22*1024**3
EXPECTED_DELTA = {"B8":{"dicm_fixed_reference": True}, "A16":{"dicm_rank":16}}
BASE_COMMIT = "cc767755fbbf18432309bad2c010c293ca7b7a20"


def folds(group, index):
    return sorted((ROOT/group/"PACS").glob(f"*_{group}_env{index}"))


def historical_fold(index):
    folder, = (A8/"P1"/"PACS").glob(f"*_P1_env{index}")
    return folder


def check_hparams(group, old, new):
    expected={**old, **EXPECTED_DELTA[group]}
    different={k:(old.get(k),new.get(k)) for k in old.keys() | new.keys() if old.get(k)!=new.get(k)}
    if new != expected:
        bad={k:(v,new.get(k)) for k,v in expected.items() if new.get(k)!=v}
        absent=sorted(new.keys()-expected.keys())
        raise ValueError(f"{group} differs from A8 outside its intended change: {bad} extra {absent}")
    return different


def check_gpu(gpu):
    free, uuid = (s.strip() for s in query("-i",str(gpu),"--query-gpu=memory.free,uuid").split(","))
    free=int(free)
    if free<=MIN_MEMORY:
        return None, f"{free} MiB free, requires >{MIN_MEMORY}"
    if shutil.disk_usage(ROOT).free < MIN_DISK:
        raise RuntimeError("Insufficient disk for rolling checkpoints and feature exports")
    for pid in query("-i",str(gpu),"--query-compute-apps=pid").splitlines():
        if not pid.strip().isdigit():
            continue
        try:
            argv=(Path("/proc")/pid.strip()/"cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if any(argv[i:i+2] == [b"--algorithm",b"DICM"] for i in range(len(argv)-1)):
            return None, f"DICM process {pid.strip()} already present on GPU{gpu}"
    return uuid, f"{free} MiB free"


def wait_gpu(gpu):
    while True:
        uuid,status=check_gpu(gpu)
        if uuid:
            return uuid,status
        say(f"GPU{gpu} waiting",status)
        time.sleep(60)


def command(group, index):
    return [sys.executable,"-B","-u","train_all.py",f"{group}_env{index}",
            str(Path(__file__).with_name(group+".yaml")),"--algorithm","DICM",
            "--dataset","PACS","--test_envs",str(index),"--steps","5001","--checkpoint_freq","100",
            "--seed","0","--trial_seed","0","--deterministic","--holdout_fraction","0.2",
            "--label_ratio","1","--model","resnet50","--data_dir",str(DATA),
            "--output_root",str(ROOT/group),"--pacs_causal_checkpoints",
            "--pacs_matched_protocol",str(ROOT/"configs"/f"A8_env{index}.json")]


def prepare():
    """Store A8 fingerprints only; no historical weights are passed to training."""
    import torch
    for index, name in enumerate(DOMAINS):
        folder=historical_fold(index)
        manifest=json.loads((folder/".github-archive"/"manifest.json").read_text())
        fold,=json.loads((folder/".github-archive"/"evaluation.json").read_text())
        cp=torch.load(folder/"checkpoints"/"iid_best.pth",map_location="cpu",weights_only=False)
        assert manifest["code"]["commit"]==BASE_COMMIT and fold["target"]==[name]
        assert fold["steps"]==5001 and fold["checkpoint_freq"]==100
        assert fold["args"]["seed"]==fold["args"]["trial_seed"]==0
        assert fold["hparams"]==cp["hparams"]
        item=dict(target=name,identity=cp["identity"],initial_sha256=cp["initial_sha256"],
                  hparams=fold["hparams"],archive_run_id=manifest["run_id"],commit=BASE_COMMIT)
        atomic_json(ROOT/"configs"/f"A8_env{index}.json",item)
        del cp
    old=json.loads((ROOT/"configs"/"A8_env0.json").read_text())["hparams"]
    for g in GROUPS:
        shown=subprocess.check_output([*command(g,0),"--show"],cwd=PROJECT,text=True,
                                      env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
        actual=yaml.safe_load(shown.split("checkpoint_freq=100\n",1)[1])
        # num_domains is resolved at dataset setup, after configuration display.
        actual.setdefault("total_num_domains",old["total_num_domains"])
        # Config.dumps prints some scientific notation in a form PyYAML displays as text.
        for key,old_value in old.items():
            if isinstance(old_value,float) and isinstance(actual.get(key),str):
                actual[key]=float(actual[key])
        say(g,"effective hparams verify",check_hparams(g,old,actual))
        (ROOT/"configs"/f"{g}_effective.txt").write_text(shown,encoding="utf-8")


def completed(folder, group, index, commit):
    audit=json.loads((ROOT/"configs"/f"A8_env{index}.json").read_text())
    outbox=folder/".github-archive"
    manifest=json.loads((outbox/"manifest.json").read_text())
    fold,=json.loads((outbox/"evaluation.json").read_text())
    assert manifest["code"]["commit"]==commit and fold["target"]==[DOMAINS[index]]
    assert fold["steps"]==5001 and fold["checkpoint_freq"]==100
    assert fold["args"]["seed"]==fold["args"]["trial_seed"]==0
    assert fold["args"]["holdout_fraction"]==.2 and fold["args"]["deterministic"]
    check_hparams(group,audit["hparams"],fold["hparams"])
    assert fold["swad_evaluation"] is not None
    assert (folder/"sample_manifest.json").is_file()
    for name in ("iid_best","swad","latest"):
        assert (folder/"checkpoints"/(name+".pth")).is_file()
    log=(folder/"log.txt").read_text()
    assert "A8 split and initial Backbone/Classifier verified" in log
    assert all(f"Reload verified for {name}.pth" in log for name in ("iid_best","swad"))
    return fold


def run(group):
    gpu=GROUPS[group]
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=PROJECT,text=True).strip()
    assert all((DATA/"PACS"/d).is_dir() for d in DOMAINS)
    with (ROOT/f"{group}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,domain in enumerate(DOMAINS):
            name=f"{group}_env{index}"
            found=folds(group,index)
            if len(found)>1:
                raise RuntimeError(f"Duplicate target fold: {name}")
            if found and (found[0]/".github-archive").is_dir():
                completed(found[0],group,index,commit)
                say(name,"already finished")
                continue
            cmd=command(group,index)
            if found:
                latest=found[0]/"checkpoints/latest.pth"
                if not latest.is_file():
                    raise RuntimeError(f"Incomplete fold has no recovery checkpoint: {found[0]}")
                cmd+=["--pacs_causal_resume",str(latest)]
            uuid,status=wait_gpu(gpu)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER="PCI_BUS_ID")
            console=ROOT/"logs"/(name+".log")
            with console.open("a",encoding="utf-8",buffering=1) as handle:
                say(name,domain,f"GPU{gpu}",status)
                p=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                running=ROOT/"logs"/f"{group}.running.json"
                atomic_json(running,dict(group=group,pid=p.pid,gpu=gpu,target=domain,index=index,
                                         commit=commit,console=str(console),command=cmd))
                if (code:=p.wait()) != 0:
                    raise RuntimeError(f"{name} exited {code}; inspect {console}")
            path,=folds(group,index)
            fold=completed(path,group,index,commit)
            say(name,domain,f"IID {fold['selection']['iid']:.4%}",
                f"SWAD {fold['selection']['SWAD']:.4%}","converged",fold["swad_evaluation"]["converged"])
            running.unlink(missing_ok=True)
        atomic_json(ROOT/"logs"/f"{group}.complete.json",dict(group=group,commit=commit,folds=4))
        say(group,"completed all four folds")


if __name__=="__main__":
    action,=sys.argv[1:]
    prepare() if action=="prepare" else run(action)
