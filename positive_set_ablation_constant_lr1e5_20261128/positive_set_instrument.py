"""Observe the fixed Adam budget without changing training or RNG state."""
import json
import os
import runpy
import sys
from pathlib import Path
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

base = Path(__file__).resolve().parent
code = base / 'code/DCCL/DCCL'
destination = Path(sys.argv[1])
args = sys.argv[2:]
config = json.loads(Path(args[1]).read_text())
sys.path.insert(0, str(code))
from domainbed.algorithms.cipt_prompt import PromptLearner
from domainbed.algorithms.cipt_env_kl import CIPTDCCL

count = 0
optimizer_seen = None
original_prompt = PromptLearner.__init__
original_encode = CIPTDCCL.encode_eval_visual
checked_cache = False

def prompt(self, *a, **kw):
    original_prompt(self, *a, **kw)
    assert self.context.shape[0] == config['cipt_prompt_length']
    print('PROMPT_CONTEXT_LENGTH', self.context.shape[0], flush=True)

def encode(self, x):
    global checked_cache
    assert all(not p.requires_grad for p in self.clip_model.parameters())
    raw = original_encode(self, x)
    if not checked_cache:
        with torch.no_grad():
            cached = self.predict_from_eval_visual(raw)
            direct = self.predict(x)
        assert torch.allclose(cached, direct, rtol=1e-5, atol=1e-5)
        assert torch.equal(cached.argmax(-1), direct.argmax(-1))
        checked_cache = True
        print('CACHE_DIRECT_LOGITS_EQUIVALENT', float((cached-direct).abs().max()), flush=True)
    return raw

stream = destination.open('x', buffering=1)
def pre(optimizer, a, kw):
    global optimizer_seen
    assert isinstance(optimizer, torch.optim.Adam)
    if optimizer_seen is None:
        optimizer_seen = optimizer
    assert optimizer is optimizer_seen and count < 5000
    assert all(g['lr'] == 1e-5 and g['weight_decay'] == 0 for g in optimizer.param_groups)

def post(optimizer, a, kw):
    global count
    stream.write(json.dumps({'step':count, 'lr_used':1e-5, 'optimizer_updates':count+1,
                             'scheduler_updates':0})+'\n')
    count += 1
    if count % 300 == 0 or count == 5000:
        stream.flush()
        os.fsync(stream.fileno())

PromptLearner.__init__ = prompt
CIPTDCCL.encode_eval_visual = encode
h1 = register_optimizer_step_pre_hook(pre)
h2 = register_optimizer_step_post_hook(post)
sys.argv = [str(code/'train_all.py'), *args]
try:
    runpy.run_path(str(code/'train_all.py'), run_name='__main__')
    assert count == 5000 and checked_cache
    print('ADAM_UPDATES 5000 CONSTANT_LR 1e-5 SCHEDULER_UPDATES 0', flush=True)
finally:
    stream.close()
    h1.remove()
    h2.remove()

