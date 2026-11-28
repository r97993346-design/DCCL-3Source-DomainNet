import sys
from pathlib import Path
import torch
import optuna
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from domainbed.algorithms.cipt_clip_guided_contrastive import cross_domain_targets, clip_guided_cross_domain_loss
from domainbed.lib.cipt_lr_schedule import make_cipt_lr_scheduler
labels=torch.tensor([0,0,0,1,1,1])
domains=torch.tensor([0,0,1,0,1,2])
visual=torch.arange(30,dtype=torch.float).reshape(6,5).requires_grad_()
causal=torch.arange(30,dtype=torch.float).reshape(6,5).clone().requires_grad_()
w,pos,den,sim=cross_domain_targets(visual,labels,domains,.2,.25,0)
same=labels[:,None]==labels[None,:]
cross=domains[:,None]!=domains[None,:]
eye=torch.eye(6,dtype=torch.bool)
assert torch.equal(pos,same & cross & ~eye)
assert torch.equal(den,pos | (~same & ~eye))
assert not w.requires_grad and not sim.requires_grad
assert torch.all(w[~pos]==0)
loss,stats=clip_guided_cross_domain_loss(visual,causal,labels,domains,temperature=.07,teacher_temperature=.2,teacher_mix=.25,topk=0)
loss.backward()
assert visual.grad is None and causal.grad is not None and torch.isfinite(loss)
sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=5,n_ei_candidates=24,multivariate=False)
study=optuna.create_study(directions=["maximize","maximize"],sampler=sampler,pruner=optuna.pruners.NopPruner())
for i in range(7):
    t=study.ask(); x=t.suggest_categorical("x",[.5,.75,1.]); study.tell(t,[x,1-x])
assert study.best_trials and isinstance(study.sampler,optuna.samplers.TPESampler)
print("PASS cross-domain masks, excluded same-domain positives, detached teacher, causal gradients, multiobjective TPESampler",optuna.__version__)
