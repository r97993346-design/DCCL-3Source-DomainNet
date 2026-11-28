"""Exercise the full stage planner and restart without GPU training."""
import copy
import tempfile
from pathlib import Path
import run as grid

with tempfile.TemporaryDirectory() as temp:
    grid.ROOT=Path(temp)
    grid.cases=[]
    grid.active={}
    grid.resources=lambda:None
    visited=[]
    def simulate(c,lock):
        if c["state"]=="complete": return
        assert c["id"] not in visited
        visited.append(c["id"])
        v=.65+.03*c["params"]["s_weight"]+.005*c["params"]["condcov_weight"]-.01*c["params"]["rd_weight"]-.2*c["params"]["contrastive_weight"]
        folds=[dict(fold=i,domain=n,source_val=v-i*.01,iid=.99,test_in=.99,test_out=.99)
               for i,n in enumerate(("Art","Clipart","Product","Real World"))]
        c.update(state="complete",result=dict(source_val=sum(f["source_val"] for f in folds)/4,
                 worst_fold_source_val=min(f["source_val"] for f in folds),
                 iid=.99,test_in=.99,test_out=.99,elapsed_seconds=0,folds=folds))
        grid.save()
    grid.execute=simulate
    grid.VLCS_REPORT=Path(temp)/"vlcs_reference.json"
    class Finished(Exception):pass
    grid.time.sleep=lambda _:(_ for _ in ()).throw(Finished())
    for _ in range(2):
        try: grid.main()
        except Finished: pass
    assert len(visited)==25
    assert len({grid.core.key(c["params"]) for c in grid.cases})==25 if hasattr(grid.core,"key") else len({tuple(c["params"][k] for k in grid.KEYS) for c in grid.cases})==25
    assert [sum(c["stage"]==i for c in grid.cases) for i in range(4)]==[1,9,9,6]
    assert sum(len(c["result"]["folds"]) for c in grid.cases)==100
    chosen=grid.p.read_json(grid.ROOT/"stage_selections.json")
    assert len(chosen["1"])==3 and len(chosen["2"])==2
    assert all(next(c for c in grid.cases if c["id"]==i)["params"]["rd_weight"]==0 for i in chosen["2"])
    assert grid.select([c for c in grid.cases if c["id"] in chosen["2"] or c["stage"]==3],1)[0]["params"]["contrastive_weight"]==0
    a=dict(id="x",result=dict(source_val=.9,worst_fold_source_val=.7))
    b=dict(id="y",result=dict(source_val=.8,worst_fold_source_val=.75))
    c=dict(id="z",result=dict(source_val=.8,worst_fold_source_val=.7))
    assert [(n,x["id"]) for n,x in grid.rank([c,b,a])]==[(0,"x"),(0,"y"),(1,"z")]
    print("PASS 25 unique configurations, 100 folds, exact stage budgets, Pareto tiers and tie order, RD=0 and CL=0 eligible, no IID stopping, resume without duplicate execution")
