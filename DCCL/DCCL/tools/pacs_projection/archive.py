"""Synchronize fold bundles and the completed comparison without publishing weights."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

from experiment_archive import BRANCHES, REMOTE, git, safe
from tools.pacs_projection.queue import DOMAINS, GROUPS, ROOT, atomic_json, folds, say


PREFIX=Path("pacs_causal_validation_20261119")/"b8_a16_20261120"
REF="refs/heads/"+BRANCHES["DICM"]


def local_git(repo, data, *args):
    env=dict(os.environ,GIT_TERMINAL_PROMPT="0",GIT_NO_LAZY_FETCH="1")
    return subprocess.run(["git","-C",str(repo),*args], input=data, check=True,
                          capture_output=True, timeout=30, env=env).stdout.strip()


def entry(name, oid, directory):
    return (b"040000 tree " if directory else b"100644 blob ")+oid+b"\t"+name.encode()


def tree(repo, rows):
    return local_git(repo,b"".join(x+b"\0" for x in rows),"mktree","--missing","-z")


def new_tree(repo, files):
    dirs, rows={},[]
    for p,data in files.items():
        if len(p.parts)==1:
            rows.append(entry(p.name,local_git(repo,data,"hash-object","-w","--stdin"),False))
        else:
            dirs.setdefault(p.parts[0],{})[Path(*p.parts[1:])]=data
    for name,descendants in dirs.items():
        rows.append(entry(name,new_tree(repo,descendants),True))
    return tree(repo,rows)


def listing(repo, revision):
    return [r for r in git(repo,"ls-tree","-z",revision).split(b"\0") if r]


def attach(repo, existing, parts, addition):
    old=listing(repo,existing)
    name=parts[0]
    present=[r for r in old if r.rsplit(b"\t",1)[-1]==name.encode()]
    if len(parts)==1:
        if present:
            raise RuntimeError(f"Archive destination already exists: {name}")
        return tree(repo,[*old,entry(name,addition,True)])
    if present:
        oid=present[0].split()[2].decode()
        child=attach(repo,oid,parts[1:],addition)
        surrounding=[r for r in old if r not in present]
    else:
        child=addition
        for component in reversed(parts[1:]):
            child=tree(repo,[entry(component,child,True)])
        surrounding=old
    return tree(repo,[*surrounding,entry(name,child,True)])


def verify(repo,commit,path,payload):
    subtree=git(repo,"ls-tree","-r","-z",f"{commit}:{path}").split(b"\0")
    objects={}
    for metadata in filter(None,subtree):
        header,name=metadata.split(b"\t",1)
        objects[name.decode()]=header.split()[2]
    expected={str(name):local_git(repo,content,"hash-object","--stdin") for name,content in payload.items()}
    if objects != expected:
        raise RuntimeError(f"Archive paths or content hashes differ: {path}")


def publish(path, files):
    payload=dict(files)
    integrity={str(k):sha256(v).hexdigest() for k,v in files.items()}
    payload[Path("file_integrity.json")]=(json.dumps(integrity,indent=2,sort_keys=True)+"\n").encode()
    with TemporaryDirectory(prefix="pacs-b8-a16-git-") as path_tmp:
        repo=Path(path_tmp)
        git(repo,"init","-q")
        git(repo,"remote","add","results",REMOTE)
        git(repo,"config","remote.results.promisor","true")
        git(repo,"config","remote.results.partialclonefilter","blob:none")
        git(repo,"fetch","--depth=1","--filter=blob:none","results",REF)
        parent=git(repo,"rev-parse","FETCH_HEAD").decode().strip()
        if git(repo,"ls-tree",parent,"--",str(path)):
            verify(repo,parent,path,payload)
            return parent
        bundle=new_tree(repo,payload)
        root=attach(repo,parent,path.parts,bundle)
        commit=local_git(repo,None,"-c","user.name=Codex","-c","user.email=codex@users.noreply.github.com",
                         "commit-tree",root.decode(),"-p",parent,"-m",f"Archive PACS DICM {path.name}").decode()
        git(repo,"push","results",commit+":"+REF)
        head=git(repo,"ls-remote","--exit-code",REMOTE,REF).decode().split()
        if not head or head[0]!=commit:
            raise RuntimeError("Archive head advanced during verification; retry")
        verify(repo,commit,path,payload)
        return commit


def safe_file(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>25*1024**2:
        raise ValueError(f"Invalid archive file {path.name}")
    if path.suffix in (".png",".pdf"):
        return path.read_bytes()
    return safe(path.read_text(encoding="utf-8")).encode()


def fold_files(folder):
    names=("log.txt","manifest.json","evaluation.json","results.jsonl")
    payload={Path(n):safe_file(folder/".github-archive"/n) for n in names}
    payload[Path("sample_manifest_sha256.txt")]=(sha256((folder/"sample_manifest.json").read_bytes()).hexdigest()+"\n").encode()
    return payload


def final_files():
    payload={}
    for part, ext in (("reports",{".md",".png",".pdf",".csv"}),
                      ("probes",{".json"}),
                      ("visualizations",{".png",".pdf",".csv",".json"})):
        for item in sorted((ROOT/part).rglob("*")):
            if item.suffix in ext:
                payload[item.relative_to(ROOT)]=safe_file(item)
    for group in GROUPS:
        p=ROOT/"configs"/f"{group}_effective.txt"
        payload[p.relative_to(ROOT)]=safe_file(p)
    for log in ("config-check.log","checkpoint-preflight.log","postprocess-preflight.log"):
        p=ROOT/"logs"/log
        payload[p.relative_to(ROOT)]=safe_file(p)
    return payload


def run():
    # Disable hydration on local Git tree operations, but fetch trees and verify via the repo SSH remote.
    with (ROOT/"archive.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            done=0
            for group in GROUPS:
                for i,domain in enumerate(DOMAINS):
                    paths=folds(group,i)
                    if len(paths)>1:
                        raise RuntimeError(f"Duplicate train folder {group} {domain}")
                    if not paths or not (paths[0]/".github-archive").is_dir():
                        continue
                    folder=paths[0]
                    receipt=folder/"archive_receipt.json"
                    try:
                        if not receipt.is_file():
                            destination=PREFIX/"folds"/group/domain
                            revision=publish(destination,fold_files(folder))
                            atomic_json(receipt,dict(commit=revision,archive_path=str(destination)))
                            say("GitHub fold verified",group,domain,revision)
                        done+=1
                    except (subprocess.SubprocessError,OSError,RuntimeError) as error:
                        say("Fold archive retry",group,domain,type(error).__name__)
            if done==8 and (ROOT/"logs"/"report.complete.json").is_file():
                try:
                    destination=PREFIX/"analysis"
                    revision=publish(destination,final_files())
                    atomic_json(ROOT/"logs"/"final_archive_receipt.json",dict(commit=revision,
                                                                            archive_path=str(destination)))
                    say("GitHub report verified",revision)
                    return
                except (subprocess.SubprocessError,OSError,RuntimeError) as error:
                    say("Final archive retry",type(error).__name__)
            time.sleep(60)


if __name__=="__main__":
    run()
