"""GitHub Git Data fallback for non-forced result and source publication."""
import base64
import hashlib
import json
import subprocess
import time

REPO="repos/r97993346-design/DCCL-3Source-DomainNet"
def api(path,method="GET",body=None,optional=False):
    cmd=["gh","api","--method",method,f"{REPO}/{path}"]
    if body is not None: cmd += ["--input","-"]
    p=subprocess.run(cmd,input=json.dumps(body) if body is not None else None,text=True,capture_output=True,timeout=40)
    if p.returncode:
        if optional and "(HTTP 404)" in p.stderr: return None
        raise RuntimeError(p.stderr[:240])
    return json.loads(p.stdout)

def publish(branch, files, base_commit=None):
    for _ in range(8):
        ref=api(f"git/ref/heads/{branch}",optional=True)
        parent=ref["object"]["sha"] if ref else base_commit
        base_tree=api(f"git/commits/{parent}")["tree"]["sha"] if parent else None
        old={}
        if base_tree:
            old={r["path"]:r["sha"] for r in api(f"git/trees/{base_tree}?recursive=1")["tree"] if r["type"]=="blob"}
        entries=[]
        for name,path in files.items():
            data=path.read_bytes()
            digest=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if old.get(name)==digest: continue
            blob=api("git/blobs","POST",{"content":base64.b64encode(data).decode(),"encoding":"base64"})
            entries.append(dict(path=name,mode="100644",type="blob",sha=blob["sha"]))
        if not entries and ref: return parent
        body={"tree":entries}
        if base_tree: body["base_tree"]=base_tree
        tree=api("git/trees","POST",body)["sha"]
        commit=api("git/commits","POST",{"message":"Synchronize MOTPE experiment records","tree":tree,"parents":[parent] if parent else []})["sha"]
        try:
            if ref: api(f"git/refs/heads/{branch}","PATCH",{"sha":commit,"force":False})
            else: api("git/refs","POST",{"ref":"refs/heads/"+branch,"sha":commit})
            print("GITHUB_API_SYNC_CONFIRMED",branch,commit,flush=True)
            return commit
        except RuntimeError:
            time.sleep(2)
    raise RuntimeError("GitHub ref contention")
