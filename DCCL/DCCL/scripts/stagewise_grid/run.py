"""Thirty-one serial configurations with deterministic Pareto stage selection."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import mean

import core
p, runner = core.p, core.runner
ROOT, NAME, DATASET = core.ROOT, core.NAME, core.DATASET
S_VALUES=(.5,1.,2.)
COV_VALUES=(1.,2.,3.,5.)
RD_VALUES=(.5,1.,2.,3.)
CL_VALUES=(.01,.025,.05)
N_CASES=31
KEYS = ("s_weight","condcov_weight","rd_weight","contrastive_weight")
CODE = core.CODE

def params(s, cov, rd=0., cl=0.):
    return dict(zip(KEYS, (float(s),float(cov),float(rd),float(cl))))

def case(label, number, stage, config, parent=None):
    return dict(id=label,number=number,stage=stage,params=config,parent=parent,state="pending")

def decision(message):
    with (ROOT/"decision_log.md").open("a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ")+message+"\n")

def write_status(state, **extra):
    p.write_json(ROOT/"status.json",dict(state=state,dataset=DATASET,planned_configurations=N_CASES,
                 complete_configurations=sum(c["state"]=="complete" for c in cases),
                 **active,**extra))
    (ROOT/"sync.request").touch()

cases=[]
active={}

def save():
    p.write_json(ROOT/"cases.json",cases)
    rows=[]
    for c in cases:
        r=c.get("result",{})
        rows.append(dict(id=c["id"],stage=c["stage"],state=c["state"],parent=c["parent"],**c["params"],
                    **{k:r.get(k) for k in ("source_val","worst_fold_source_val","iid","test_in","test_out","elapsed_seconds")},
                    reused=bool(c.get("reused_from")),folds_json=json.dumps(r.get("folds",[]))))
    p.write_csv(ROOT/"trials_summary.csv",rows,["id","stage","state","parent",*KEYS,"source_val","worst_fold_source_val","iid","test_in","test_out","elapsed_seconds","reused","folds_json"])
    folds=[]
    for c in cases:
        for fold in c.get("result",{}).get("folds",[]):
            folds.append(dict(case_id=c["id"],**{k:fold.get(k) for k in ("fold","domain","seed","selected_step","source_val","iid","test_in","test_out","updates","scheduler_updates","actual_lr","elapsed_seconds","learning_rates")},
                              losses_json=json.dumps(fold.get("losses",{}))))
    p.write_csv(ROOT/"folds_summary.csv",folds,["case_id","fold","domain","seed","selected_step","source_val","iid","test_in","test_out","updates","scheduler_updates","actual_lr","elapsed_seconds","learning_rates","losses_json"])
    (ROOT/"sync.request").touch()

def dominates(a,b):
    x,y=a["result"],b["result"]
    return (x["source_val"]>=y["source_val"] and x["worst_fold_source_val"]>=y["worst_fold_source_val"]
            and (x["source_val"]>y["source_val"] or x["worst_fold_source_val"]>y["worst_fold_source_val"]))

def rank(pool):
    remaining=list(pool)
    ordered=[]
    layer=0
    while remaining:
        front=[a for a in remaining if not any(dominates(b,a) for b in remaining)]
        front.sort(key=lambda c:(-c["result"]["source_val"],-c["result"]["worst_fold_source_val"],c["id"]))
        for c in front:
            ordered.append((layer,c))
            remaining.remove(c)
        layer+=1
    return ordered

def select(pool,count):
    return [c for _,c in rank(pool)[:count]]

def resources():
    while runner.available_ram_kib() <= 10*1024**2:
        write_status("waiting_for_ram", available_ram_kib=runner.available_ram_kib())
        time.sleep(30)


def refresh(dataset,study,root,state,**kwargs):
    write_status(state,fold=kwargs.get("active"))
    resources()
runner.refresh=refresh

def retry_fold(root,dataset,fold,number,config,gpu,env,lock_fd,*,updates=p.STEPS):
    out=root/f"env{fold}"
    out.mkdir(exist_ok=True)
    attempts = sorted(out.glob("attempt_*"))

    def inspect(attempt):
        try:
            return core.read_attempt(attempt,dataset,fold,config,updates=updates)
        except (ValueError, KeyError, OSError) as exc:
            error = f"{attempt}: {type(exc).__name__}: {exc}"
            p.write_json(attempt/"failure.json",dict(error=error))
            decision(error)
            raise core.ProtocolError(error) from exc

    failures=0
    for attempt in attempts:
        exit_code=attempt/"exit_code"
        if not exit_code.exists(): continue
        if exit_code.read_text().strip()=="0":
            return inspect(attempt)
        failures+=1
    while failures<2:
        resources()
        attempt=out/f"attempt_{len(attempts)+1:03d}"
        runner.start_attempt(attempt,dataset,fold,number,root/"config.json",gpu,env,lock_fd,updates=updates)
        if (attempt/"exit_code").read_text().strip()=="0":
            return inspect(attempt)
        failures+=1
        attempts.append(attempt)
        tail=(attempt/"console.log").read_text(errors="replace").splitlines()[-12:]
        error=f"Process exit code actual={(attempt/'exit_code').read_text().strip()}, expected=0; console tail: {tail}"
        p.write_json(attempt/"failure.json",dict(error=error))
        decision(error)
        if any("actual=" in line and "expected=" in line for line in tail):
            raise core.ProtocolError(error)
        if failures<2:
            write_status("retry_wait",error=error)
            time.sleep(60)
    raise core.ProtocolError(f"Fold {fold}: two nonzero training exits; see {out}/attempt_*/failure.json")

runner.run_fold=retry_fold

def execute(c,lock):
    if c["state"]=="complete": return
    active.update(case_id=c["id"],stage=c["stage"],params=c["params"])
    c["state"]="running";save();resources()
    result=runner.finish_trial(ROOT/DATASET,DATASET,None,c["number"],c["params"],0,
                              runner.environment_for("GPU-97473323-921d-3883-586d-1a0230891c12"),lock.fileno())
    assert "error" not in result and [f["fold"] for f in result["folds"]]==list(range(4))
    result["worst_fold_source_val"]=min(f["source_val"] for f in result["folds"])
    result["full_hparams"]=core.hp(c["params"])
    c.update(state="complete",result=result)
    save();write_status("running")
    decision(f"Completed {c['id']}, mean source_val={result['source_val']}, minimum source_val={result['worst_fold_source_val']}.")

def stage_report(stage,pool,selected):
    ordered=rank(pool)
    text=[f"# Stage {stage} results","", "Candidates are ranked by Pareto layer, descending mean source_val, then descending minimum source_val.","",
          "| Candidate | Pareto layer | Mean source_val | Minimum source_val | Mean IID | Mean test_out |",
          "|---|---|---|---|---|---|"]
    for layer,c in ordered:
        r=c["result"]
        text.append(f"| {c['id']} | {layer} | {r['source_val']:.6f} | {r['worst_fold_source_val']:.6f} | {r['iid']:.6f} | {r['test_out']:.6f} |")
    text+=["","Selected candidates are "+", ".join(c["id"] for c in selected)+".","","## Parameter sensitivity",""]
    for key in KEYS:
        values=sorted({c["params"][key] for c in pool})
        for value in values:
            group=[c for c in pool if c["params"][key]==value]
            text.append(f"- {key}={value:g}, {len(group)} candidates, mean source_val={mean(c['result']['source_val'] for c in group):.6f}, mean minimum source_val={mean(c['result']['worst_fold_source_val'] for c in group):.6f}.")
    text+=["","## Changes from the preceding control",""]
    for c in pool:
        parent=c.get("parent")
        if not parent: continue
        baseline=next(x for x in cases if x["id"]==parent)
        a,b=c["result"],baseline["result"]
        delta=a["source_val"]-b["source_val"]
        worst=a["worst_fold_source_val"]-b["worst_fold_source_val"]
        text.append(f"- {c['id']} versus {parent}, mean source_val change {delta:+.6f}, minimum source_val change {worst:+.6f}.")
        for f,g in zip(a["folds"],b["folds"]):
            text.append(f"  - {f['domain']}, source_val change {f['source_val']-g['source_val']:+.6f}, IID change {f['iid']-g['iid']:+.6f}, test_out change {f['test_out']-g['test_out']:+.6f}.")
        if delta<=0 or worst<=0:
            text.append("  - At least one source-validation objective failed to improve. Review the recorded raw and weighted losses for excessive constraint or interaction effects.")
    text+=["","Target-domain metrics are recorded for research interpretation. Source-validation metrics alone determine selection.",""]
    p.write_text(ROOT/f"stage_{stage}_report.md","\n".join(text))
    selections=p.read_json(ROOT/"stage_selections.json") if (ROOT/"stage_selections.json").exists() else {}
    selections[str(stage)]=[c["id"] for c in selected]
    p.write_json(ROOT/"stage_selections.json",selections)
    (ROOT/"sync.request").touch()

def final_comparison():
    pool=[c for c in cases if c["stage"]==3 or c["id"] in p.read_json(ROOT/"stage_selections.json")["2"]]
    stage_report(3,pool,select(pool,len(pool)))
    best=select(pool,1)[0]
    baseline=cases[0]
    text=[f"# {DATASET} stagewise grid", "", f"Completed {N_CASES} configurations across four folds each.", ""]
    for label,c in (("Stage 0 independent reference",baseline),("Final source-validation leader",best)):
        r=c["result"]
        text.append(f"{label} {c['id']} {c['params']}, source mean {r['source_val']:.6f}, source minimum {r['worst_fold_source_val']:.6f}, IID {r['iid']:.6f}, test_out {r['test_out']:.6f}.")
    text.append(f"Final leader versus A00, mean source change {best['result']['source_val']-baseline['result']['source_val']:+.6f}.")
    p.write_text(ROOT/"final_comparison.md","\n".join(text)+"\n")

def ensure_cases(new):
    existing={c["id"]:c for c in cases}
    for c in new:
        if c["id"] in existing:
            assert all(existing[c["id"]][k]==c[k] for k in ("number","stage","params","parent"))
        else: cases.append(c)
    save()

def main():
    global cases
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol=dict(name=NAME,dataset=DATASET,base_training=core.hp(params(0,0)),fixed_args=p.FIXED_ARGS,
                  evaluation_cache=True,seed_strategy="DomainBed per-target hash",fold_seeds=[p.seed_for(DATASET,i) for i in range(4)],
                  updates=5001,final_update_index=5000,check=300,lr=2e-5,lr_schedule=core.p.LR_POLICY,prompt_length=4,
                  model_implementation_reference="condcov_rd_crossdomain_constant1e5_motpe_20261128",
                  stages={"0":1,"1":12,"2":12,"3":6},planned_configurations=N_CASES,planned_folds=124,
                  selection=["Pareto layer","descending mean source_val","descending minimum source_val","candidate ID for exact ties"],
                  inheritance="hyperparameters only; fresh model and fixed fold seeds for every configuration",
                  reuse="only previously validated results from this exact protocol and output directory",
                  code_commit=p.git("rev-parse","HEAD"))
    for name,value in (("protocol",protocol),("search_space",dict(stage0=[params(0,0)],stage1=[params(s,c) for s in S_VALUES for c in COV_VALUES],
                                   stage2=dict(top=3,rd=list(RD_VALUES),include_previous_rd_zero=True),stage3=dict(top=2,cl=list(CL_VALUES),include_previous_cl_zero=True)))):
        path=ROOT/(name+".json")
        if path.exists():
            old=p.read_json(path)
            if name=="protocol": value["code_commit"]=old["code_commit"]
            assert old==value
        else: p.write_json(path,value)
    with (ROOT/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=p.read_json(ROOT/"cases.json") if (ROOT/"cases.json").exists() else []
        ensure_cases([case("A00",0,0,params(0,0))])
        execute(cases[0],lock)
        stage_report(0,[cases[0]],[cases[0]])
        ensure_cases([case(f"A{i:02}",i,1,params(s,c),"A00") for i,(s,c) in enumerate(((s,c) for s in S_VALUES for c in COV_VALUES),1)])
        stage1=[c for c in cases if c["stage"]==1]
        for c in stage1: execute(c,lock)
        top3=select(stage1,3);stage_report(1,stage1,top3)
        ensure_cases([case(f"B{i:02}",12+i,2,dict(parent["params"],rd_weight=float(rd)),parent["id"])
                      for i,(parent,rd) in enumerate(((c,r) for c in top3 for r in RD_VALUES),1)])
        stage2=[c for c in cases if c["stage"]==2]
        for c in stage2: execute(c,lock)
        top2=select([*top3,*stage2],2);stage_report(2,[*top3,*stage2],top2)
        ensure_cases([case(f"C{i:02}",24+i,3,dict(parent["params"],contrastive_weight=cl),parent["id"])
                      for i,(parent,cl) in enumerate(((c,v) for c in top2 for v in CL_VALUES),1)])
        for c in cases:
            if c["stage"]==3: execute(c,lock)
        assert len(cases)==N_CASES and all(c["state"]=="complete" for c in cases)
        active.clear()
        final_comparison();save();write_status("complete")

if __name__=="__main__":
    try:
        main()
    except (core.ProtocolError, KeyError) as exc:
        decision(f"Stopped for protocol or parsing diagnosis: {exc}")
        write_status("validation_failed", error=str(exc))
        raise
