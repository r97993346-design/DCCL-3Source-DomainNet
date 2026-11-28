"""Audit the actual Adam and cosine scheduler call order without changing training."""
import json
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

code = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(code), str(code/"scripts/cipt_clip_supcon_optuna")]
from protocol import cosine_lr
from domainbed.algorithms.cipt_prompt import PromptLearner
from domainbed.algorithms.cipt_env_kl import CIPTDCCL

destination = Path(sys.argv[1])
args = sys.argv[2:]
config = json.loads(Path(args[1]).read_text())
initial = 2e-5
updates = 5001
adam = scheduled = 0
optimizer_seen = None
used_lr = None
cache_checked = False

def check(field, actual, expected):
    if actual != expected:
        raise ValueError(f"{field}: actual={actual!r}, expected={expected!r}")

def check_lr(field,actual,expected):
    if abs(actual-expected) > max(1e-16,abs(expected)*1e-7):
        raise ValueError(f"{field}: actual={actual!r}, expected={expected!r}")

check("configured LR",config["lr"],initial)
check("configured scheduler",config["cipt_lr_schedule"],"cosine")
check("configured eta_min",config["cipt_lr_min"],0.)
check("GPU visibility",torch.cuda.device_count(),1)
check("GPU UUID",f"GPU-{torch.cuda.get_device_properties(0).uuid}".lower(),os.environ["CUDA_VISIBLE_DEVICES"].lower())

original_update = CIPTDCCL.update
original_prompt = PromptLearner.__init__
original_encode = CIPTDCCL.encode_eval_visual
original_step = CosineAnnealingLR.step

def prompt(self, *a, **kw):
    original_prompt(self,*a,**kw)
    check("effective prompt length",self.context.shape[0],4)
    print("PROMPT_CONTEXT_LENGTH 4",flush=True)

def encode(self,x):
    global cache_checked
    check("CLIP frozen",all(not p.requires_grad for p in self.clip_model.parameters()),True)
    raw=original_encode(self,x)
    if not cache_checked:
        with torch.no_grad():
            cached,direct=self.predict_from_eval_visual(raw),self.predict(x)
        check("cached logits equivalent",torch.allclose(cached,direct,rtol=1e-5,atol=1e-5),True)
        cache_checked=True
        print("CACHE_DIRECT_LOGITS_EQUIVALENT",float((cached-direct).abs().max()),flush=True)
    return raw

def pre(optimizer,args,kwargs):
    global optimizer_seen, used_lr
    if optimizer_seen is None: optimizer_seen=optimizer
    check("Adam optimizer",isinstance(optimizer,torch.optim.Adam) and optimizer is optimizer_seen,True)
    check("Adam/scheduler ordering",adam,scheduled)
    check("Adam update within budget",adam<updates,True)
    check("Adam groups",len(optimizer.param_groups),1)
    group=optimizer.param_groups[0]
    check("Adam weight decay",group["weight_decay"],0)
    used_lr=group["lr"]
    check_lr(f"Adam[{adam}] LR",used_lr,cosine_lr(adam,initial,updates))

def post(optimizer,args,kwargs):
    global adam
    adam+=1

def cosine_step(self,*a,**kw):
    global scheduled
    if self.last_epoch < 0:
        check("schedule periods",self.T_max,updates)
        check("minimum LR",self.eta_min,0)
        return original_step(self,*a,**kw)
    check("scheduler optimizer identity",self.optimizer is optimizer_seen,True)
    check("scheduler after Adam",adam,scheduled+1)
    result=original_step(self,*a,**kw)
    scheduled+=1
    check("scheduler epoch",self.last_epoch,scheduled)
    lr=self.optimizer.param_groups[0]["lr"]
    check_lr(f"scheduler[{scheduled}] LR",lr,cosine_lr(scheduled,initial,updates))
    trace.write(json.dumps(dict(step=adam-1,lr_used=used_lr,lr_after_scheduler=lr,
                                optimizer_updates=adam,scheduler_updates=scheduled))+"\n")
    return result

def update(self,*args,**kwargs):
    values=original_update(self,*args,**kwargs)
    weights=dict(e_cls=self.e_cls_weight,s_uniform=self.s_uniform_weight,
                 condcov=self.ind_weight,rd=self.env_loss_weight,
                 contrastive=values["contrastive_weight_eff"])
    values["loss_contrastive"]=values["loss_clip_contrast"]
    for term,weight in weights.items():
        values[f"weight_{term}"]=weight
        values[f"weighted_{term}"]=weight*values[f"loss_{term}"]
    return values

CIPTDCCL.update=update
PromptLearner.__init__=prompt
CIPTDCCL.encode_eval_visual=encode
CosineAnnealingLR.step=cosine_step
first=register_optimizer_step_pre_hook(pre)
second=register_optimizer_step_post_hook(post)
sys.argv=[str(code/"train_all.py"),*args]
with destination.open("x",buffering=1) as trace:
    runpy.run_path(sys.argv[0],run_name="__main__")
    check("Adam updates",adam,updates)
    check("cosine updates",scheduled,updates)
    check("final LR",optimizer_seen.param_groups[0]["lr"],0.)
    check("cache used",cache_checked,True)
    print(f"ADAM_UPDATES {adam} COSINE_UPDATES {scheduled} FINAL_LR 0.0",flush=True)
