"""Score and export each completed IID and SWAD checkpoint with exact A8 sample IDs."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
import torch

from domainbed.lib.causal_checkpoints import load_inference, prediction, qr_basis
from tools.pacs_causal.features import STYLES, encode, project, source_samples
from tools.pacs_causal.metrics import PROBE, coverage, energy, geometry, probe_pair
from tools.pacs_causal.visualize import plots
from tools.pacs_projection.queue import A8, DOMAINS, GROUPS, ROOT, atomic_json, completed, folds, say


def hash_path(path):
    h=sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def protection(z, q, y, d, train):
    """SVD in Q coordinates using equal domain centroids per class, only in source train."""
    u=z[train].astype(np.float64)@q.astype(np.float64)
    classes, domains=y[train],d[train]
    cells=[]
    for c in np.unique(classes):
        segments=[u[(classes==c) & (domains==p)] for p in np.unique(domains)]
        assert all(len(chunk) for chunk in segments)
        cells.append(np.stack([chunk.mean(0) for chunk in segments]).mean(0))
    centers=np.stack(cells)
    differences=centers-centers.mean(0)
    _,singular,vt=np.linalg.svd(differences,full_matrices=True)
    tol=max(differences.shape)*np.finfo(np.float64).eps*singular[0]
    rank=int((singular>tol).sum())
    assert rank<=len(centers)-1
    keep,remove=vt[:rank].T,vt[rank:].T
    np.testing.assert_allclose(keep@keep.T+remove@remove.T,np.eye(q.shape[1]),atol=1e-12)
    np.testing.assert_allclose(differences@remove,0,atol=1e-7)
    return (q@remove).astype(np.float32), dict(Q_rank=q.shape[1], class_rank=rank,
             remaining_rank=remove.shape[1],class_center_singular_values=singular.tolist(),
             rank_floor=float(tol))


def class_domain_drift(z,q,q_env,y,d,val):
    centers=[]
    for c in np.unique(y):
        means=np.stack([z[val & (y==c) & (d==domain)].mean(0) for domain in np.unique(d)])
        centers.append(means-means.mean(0))
    differences=np.concatenate(centers)
    den=energy(differences @ q)
    return float(energy(differences @ q_env)/den) if den else None


def describe(group, index, variant, check, folder):
    domain=DOMAINS[index]
    model,saved=load_inference(check,"cuda")
    assert saved["kind"]==variant and saved["identity"]["test_envs"]==[index]
    assert saved["hparams"]["dicm_beta"]==.75 and saved["hparams"]["dicm_rank"]==(8 if group=="B8" else 16)
    assert saved["source"]["commit"]==json.loads((ROOT/"logs"/f"{group}.complete.json").read_text())["commit"]
    mf=folder/saved["sample_manifest"]["file"]
    assert hash_path(mf)==saved["sample_manifest"]["sha256"]
    logits=prediction(model,saved["example_input"].cuda())
    torch.testing.assert_close(logits,saved["example_logits"],rtol=1e-5,atol=1e-6)
    reload_error=float((logits-saved["example_logits"]).abs().max().item())
    wrapped=model.module if variant=="swad" else model.get_forward_model()
    assert wrapped.reference is not None
    projection=wrapped.network[1]
    fixed=projection.reference_mean.detach().cpu().numpy() if group=="B8" else None
    assert (projection.reference_mean is not None)==(group=="B8")
    assert group!="B8" or bool(projection.reference_ready.item())
    q=qr_basis(wrapped).numpy()
    np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-5)
    np.testing.assert_allclose(np.linalg.svd(q.T@saved["q"].numpy(),compute_uv=False),
                               np.ones(q.shape[1]),atol=1e-5)
    basis=projection.subspace.basis.detach().cpu().numpy().copy()

    rows,classes=source_samples(saved)
    names=("sample_id","y","domain","is_source_train","domain_train","domain_validation","tsne_indices","intervention_indices")
    with np.load(A8/"features"/domain/"features.npz", allow_pickle=False) as bank:
        reference_samples={name:bank[name] for name in names}
    mappings={"sample_id":[r["sample_id"] for r in rows],"y":[r["category"] for r in rows],
              "domain":[r["domain"] for r in rows],"is_source_train":[r["split"]=="train" for r in rows]}
    for name,values in mappings.items():
        np.testing.assert_array_equal(reference_samples[name],values)
    train=reference_samples["is_source_train"]
    val=~train
    y,d=reference_samples["y"],reference_samples["domain"]
    assert index not in d
    assert np.all(train[reference_samples["domain_train"]])
    assert np.all(val[reference_samples["domain_validation"]]) and np.all(val[reference_samples["intervention_indices"]])
    encoder=SimpleNamespace(featurizer=wrapped.network[0],pre_featurizer=wrapped.reference)
    z,ref=encode(encoder,rows,"cuda")
    pz,pref=project(z,q),project(ref,q)
    s_raw=project(z-ref,q)
    chosen_ref=ref if fixed is None else fixed
    e=z-.75*project(z-chosen_ref,q)
    with torch.no_grad():
        expected_e=projection(torch.from_numpy(z[:16]).cuda(), reference=torch.from_numpy(ref[:16]).cuda())
    np.testing.assert_allclose(e[:16],expected_e.cpu().numpy(),rtol=1e-5,atol=5e-6)
    reps=dict(z=z,z_ref=ref,Pz=pz,Pz_ref=pref,E_beta=e,S_removed=z-e,
              E_orth=z-pz,E_full_res=z-s_raw,S_raw=s_raw)
    extra={}
    reference_error=None
    if fixed is not None:
        reference_from_bank=np.stack([ref[train & (d==i)].mean(0,dtype=np.float64) for i in np.unique(d)]).mean(0)
        reference_error=float(np.max(np.abs(fixed-reference_from_bank)))
        np.testing.assert_allclose(fixed,reference_from_bank,rtol=.001,atol=.0005)
        fixed_raw=project(z-fixed,q)
        extra.update(z_ref_mean=fixed,S_raw_fixed=fixed_raw,E_beta_sample_reference=z-.75*s_raw,
                     E_full_fixed=z-fixed_raw,Pz_ref_mean=project(fixed[None],q)[0])
    interventions=reference_samples["intervention_indices"]
    snap={}
    updates=[]
    q_env,rank=protection(z,q,y,d,train)
    for style in STYLES:
        a,b=encode(encoder,[rows[int(i)] for i in interventions],"cuda",style)
        d_z=a-z[interventions]
        d_r=d_z-(b-ref[interventions])
        snap.update({"delta_z_"+style:d_z,"delta_r_"+style:d_r})
        for selector,value in (("delta_z",d_z),("delta_r",d_r)):
            for space, matrix in (("Q",q),("Q_env_after_class_protection",q_env)):
                updates.append(dict(style=style,response=selector,space=space,**coverage(value,matrix)))
    del model,wrapped,encoder,projection
    torch.cuda.empty_cache()
    cache=ROOT/"features"/group/domain/variant
    cache.mkdir(parents=True,exist_ok=True)
    temporary=cache/"pending_features.npz"
    np.savez(temporary,**reps,**extra,**reference_samples,**snap,Q=q,Basis=basis,Q_env=q_env)
    temporary.replace(cache/"features.npz")

    payload=dict(train=train,y=y,env=d,domain_train=reference_samples["domain_train"],
                                 domain_validation=reference_samples["domain_validation"])
    probes={name:probe_pair(reps[name],payload) for name in ("z","z_ref","E_beta","S_removed","Pz","Pz_ref","E_orth","E_full_res")}
    if any(not v["converged"] for pairs in probes.values() for v in pairs.values()):
        say("Probe convergence warning",group,domain,variant)
    axes=PCA(n_components=48,svd_solver="randomized",random_state=0).fit(z[train])
    geo={name:geometry(reps[name][val],y[val],d[val],axes)
         for name in ("z","E_beta","E_orth","E_full_res")}
    plot_payload=dict(representations=reps,tsne=reference_samples["tsne_indices"],env=d,y=y,
                      classes=classes,domains=list(DOMAINS),target=index,source=rows)
    plot_dir=ROOT/"visualizations"/group/domain/variant
    settings=plots(plot_payload,plot_dir)
    settings["candidate"]=group
    settings["checkpoint_kind"]=variant
    settings["E_definition"]="fixed source mean" if group=="B8" else "per-sample frozen reference"
    atomic_json(plot_dir/"plot_settings.json",settings)
    record=dict(group=group,held_out=domain,checkpoint=variant,checkpoint_sha256=hash_path(check),
             checkpoint_file=str(check),features_file=str(cache/"features.npz"),
             split_sha256=saved["identity"]["split_sha256"],source=saved["source"],
             sample_manifest_sha256=saved["sample_manifest"]["sha256"],
             reload_max_abs_logit_difference=reload_error,
             probe_protocol=PROBE,source_train=int(train.sum()),source_validation=int(val.sum()),
             probes=probes,geometry=geo,coverage=updates,protection=rank,
             within_class_domain_drift_fraction_Q_env=class_domain_drift(z,q,q_env,y,d,val),
             removed_energy_over_z=energy((z-e)[val])/energy(z[val]),
             projected_z_energy_over_z=energy(pz[val])/energy(z[val]),
             projected_ref_energy_over_z=energy(pref[val])/energy(z[val]),
             frozen_mean_recomputed_max_abs_difference=reference_error,
             fixed_mean_sha256=sha256(np.ascontiguousarray(fixed).tobytes()).hexdigest() if fixed is not None else None)
    return record


def wait_training(group):
    marker=ROOT/"logs"/f"{group}.complete.json"
    while not marker.is_file():
        panes=subprocess.run(["tmux","list-panes","-t",f"=dicm_pacs_proj_{group}","-F","#{pane_dead}"],
                             text=True,capture_output=True)
        if (panes.returncode or "0" not in panes.stdout.splitlines()) and not marker.is_file():
            raise RuntimeError(f"Training queue {group} exited without completion, see logs/{group}.queue.log")
        time.sleep(60)
    info=json.loads(marker.read_text())
    assert info["folds"]==4
    return info["commit"]


def run(group):
    commit=wait_training(group)
    # The parent runs on CPU and starts a fresh subprocess per fold for CUDA visibility.
    from tools.pacs_projection.queue import wait_gpu
    for i,domain in enumerate(DOMAINS):
        folder,=folds(group,i)
        completed(folder,group,i,commit)
        for variant in ("iid_best","swad"):
            dst=ROOT/"probes"/group/domain/(variant+".json")
            if dst.is_file():
                data=json.loads(dst.read_text())
                expected=folder/"checkpoints"/(variant+".pth")
                assert data["checkpoint_sha256"]==hash_path(expected)
                assert Path(data["features_file"]).is_file()
                continue
            uuid,status=wait_gpu(GROUPS[group])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER="PCI_BUS_ID")
            say("Extracting",group,domain,variant,status)
            subprocess.run([sys.executable,"-B","-u","-m","tools.pacs_projection.analyze",
                            group,str(i),variant],env=env,check=True)
    atomic_json(ROOT/"logs"/f"{group}.analysis_complete.json",dict(group=group,commit=commit,features=8))


if __name__=="__main__":
    if len(sys.argv)==2:
        run(sys.argv[1])
    else:
        group,index,variant=sys.argv[1:]
        i=int(index)
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        folder,=folds(group,i)
        comment=describe(group,i,variant,folder/"checkpoints"/(variant+".pth"),folder)
        dest=ROOT/"probes"/group/DOMAINS[i]/(variant+".json")
        dest.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(dest,comment)
        say("Verified feature export",group,DOMAINS[i],variant)
