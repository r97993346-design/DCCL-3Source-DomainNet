"""Checks specific to the two changed configurations, plus checkpoint reload tests."""
from pathlib import Path
import numpy as np
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import yaml

from domainbed.algorithms.dicm import DICM
from domainbed.lib import causal_checkpoints as cp
from domainbed.lib.fixed_reference import domain_balanced_reference
from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley


def settings(group):
    path = (Path(__file__).parent.parent/"pacs_causal"/"P1.yaml" if group=="A8"
            else Path(__file__).with_name(group+".yaml"))
    return yaml.safe_load(path.read_text())


def tiny(shape, classes, domains, hp):
    class Features(nn.Module):
        n_outputs=32
        def __init__(self):
            super().__init__()
            self.lin=nn.Linear(48,self.n_outputs)
        def forward(self,x):
            return self.lin(x.flatten(1))
    with patch("domainbed.algorithms.dicm.networks.Featurizer",side_effect=lambda *a,**k:Features()):
        return DICM(shape,classes,domains,hp)


def new(group, shape, device, cls=tiny):
    torch.manual_seed(0)
    return cls(shape,7,3,settings(group)).to(device)


def verify_reference_estimation():
    tensors=[torch.tensor([[3.],[3.],[3.]]),torch.tensor([[9.]]),torch.tensor([[6.],[6.]])]
    split=[(SimpleNamespace(underlying_dataset=[(v,0) for v in row],keys=list(range(len(row)))),None)
           for row in tensors]
    before=torch.get_rng_state().clone()
    with patch("domainbed.lib.fixed_reference.basic",side_effect=lambda x:x):
        center,counts=domain_balanced_reference(nn.Identity().eval(),split,[0,1,2],"cpu",workers=0)
    assert counts=={"0":3,"1":1,"2":2} and center.item()==6
    assert torch.equal(before,torch.get_rng_state())
    print("PASS: reference means weight domains equally and preserve the sampler RNG")


def verify_rank_and_init():
    hashes={}
    for g in ("A8","B8","A16"):
        model=new(g,(3,224,224),"cpu",DICM)
        hashes[g]=cp.shared_initialization(model)
        rank=16 if g=="A16" else 8
        assert model.projection.subspace.basis.shape==(2048,rank)
        if g=="A8":
            old=model.projection.subspace.basis.detach().clone()
        elif g=="B8":
            torch.testing.assert_close(model.projection.subspace.basis,old,rtol=0,atol=0)
        else:
            assert not torch.equal(model.projection.subspace.basis[:,:8],old)
        del model
    assert hashes["A8"]==hashes["B8"]==hashes["A16"]
    print("PASS: actual pretrained ResNet-50/Classifier initialized identically; A16 starts with new 2048x16 basis")


def check_update_and_state():
    shape=((3,4,4),7,3)
    x=[torch.randn(7,3,4,4,device="cuda") for _ in range(3)]
    y=[torch.arange(7,device="cuda") for _ in range(3)]
    aug=[v-.07 for v in x]
    a8=new("A8",shape[0],"cuda")
    b8=new("B8",shape[0],"cuda")
    mean=torch.arange(32,device="cuda").float().div(32)
    p=b8.projection
    p.initialize_reference_mean(mean)
    try:
        p.initialize_reference_mean(mean*0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Fixed-mean buffer accepted a second initialization")
    b8.eval()
    with torch.no_grad():
        z=b8.featurizer(x[0])
        q=p.subspace.orthonormal_basis()
        expected=z-.75*((z-mean)@q)@q.T
        torch.testing.assert_close(b8.predict_embed(x[0]),expected)
        torch.testing.assert_close(b8._stable(z,q,torch.randn_like(z)),expected)
    a8.train(); b8.train()
    old=a8.update(x,y,x_2=aug,step=0)
    current=b8.update(x,y,x_2=aug,step=0)
    for key in ("loss_isr","loss_isr_aug","loss_isr_dom","weighted_loss_isr"):
        assert current[key]==old[key],key
    torch.testing.assert_close(a8.residual_bank.prototypes,b8.residual_bank.prototypes)
    torch.testing.assert_close(a8.projection.subspace.basis,b8.projection.subspace.basis)
    assert current["supcon_weight"]==current["precl_weight"]==current["gt_weight"]==0
    print("PASS: B8 fixed projection, unchanged sample PIRE/CDRM/ISR and initial optimizer update to Basis")
    for name, model in (("B8",b8),("A16",new("A16",shape[0],"cuda"))):
        for _ in range(2):
            model.update(x,y,x_2=aug,step=1)
        with tempfile.TemporaryDirectory(prefix="pacs-projection-check-") as tmp:
            writer=cp.FoldCheckpoints(tmp,{"target":0},shape,settings(name),model,5001,100)
            kwargs=dict(n_converge=3,n_tolerance=6,tolerance_ratio=.3)
            valley=LossValley(None,**kwargs)
            segment=AveragedModel(model)
            segment.update_parameters(model,step=0)
            valley.update_and_evaluate(segment,.73,1.,None)
            sample=x[0][:2]
            writer.checkpoint(model,valley,AveragedModel(model),sample,[dict(step=0,train_out=.73)])
            writer.save_inference(writer.final,"swad",segment.cuda(),sample,{"start":0,"end":0})
            iid=cp.verify_inference(writer.best,"cuda",tiny)
            swa=cp.verify_inference(writer.final,"cuda",tiny)
            other=new(name,shape[0],"cuda")
            restored=LossValley(None,**kwargs)
            count,records,_=writer.resume(other,restored,Path(tmp)/"results.jsonl","cuda")
            assert count==1 and other.optimizer.state and records[-1]["train_out"]==.73
            torch.testing.assert_close(cp.prediction(model,sample),cp.prediction(other,sample),atol=0,rtol=0)
            if name=="B8":
                assert torch.equal(other.projection.reference_mean,mean)
                reloaded,value=cp.load_inference(writer.final,"cuda",tiny)
                assert torch.equal(reloaded.module.network[1].reference_mean,mean)
                assert reloaded.module.reference is not None
            print(f"PASS: {name} IID/SWAD reload logits {iid:.3g}/{swa:.3g}, optimizer and rolling recovery")


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    verify_reference_estimation()
    verify_rank_and_init()
    check_update_and_state()
