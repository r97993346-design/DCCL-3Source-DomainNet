#!/usr/bin/env python3
"""Capture E and native Z_k during predict(x), then plot source/target panels.

Use the original training repository and trusted local checkpoints. This script
never gives image labels to predict(), text selection, or TDA. A point in the Z
panels is the mean of the model-predicted candidate's normalized Z_k branches.
It is a diagnostic summary; native logits remain the classification output.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import numpy as np


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_from_checkpoint(checkpoint):
    state = checkpoint.get('model_dict', checkpoint.get('state_dict', checkpoint))
    suffix = 'causal_decomposition.causal_adapter.weight'
    matches = [k for k in state if k.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one causal adapter in checkpoint')
    prefix = matches[0][:-len(suffix)]
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def flatten_ids(value):
    if isinstance(value, (list, tuple)):
        return [i for part in value for i in flatten_ids(part)]
    return [int(value)]


def read_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique image paths')
    for r in rows:
        parts = Path(r['relative_path']).parts
        if len(parts) < 3 or parts[0] != r['domain']:
            raise ValueError('Manifest domain/path mismatch')
    return rows


def extract_one(args, rows, checkpoint_path, model_id, expected_mode):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from domainbed.algorithms import get_algorithm_class

    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False  # User's trusted training checkpoints.
    checkpoint = torch.load(checkpoint_path, **kwargs)
    hp = dict(checkpoint.get('model_hparams', checkpoint.get('hparams', {})))
    if hp.get('cipt_objective', 'legacy') != 'legacy':
        raise ValueError('This paired comparison requires legacy TDA checkpoints')
    if hp.get('cipt_template_mode') != expected_mode or not hp.get('cipt_use_tda', True):
        raise ValueError('Checkpoint is not the expected trained TDA model')
    if hp.get('cipt_selector_mode', 'random') != 'random':
        raise ValueError('Expected the archived random-training / first-K-evaluation setup')
    classes = hp.get('cipt_class_names')
    if not classes:
        raise ValueError('Checkpoint lacks class names; do not guess class order')
    canon = lambda s: str(s).replace('_', ' ').strip().lower()
    if any(canon(r['class']) != canon(classes[int(r['class_id'])]) for r in rows):
        raise ValueError('Manifest class order differs from checkpoint')
    domains = sorted({r['domain'] for r in rows})
    test_ids = checkpoint.get('test_envs')
    if test_ids is None:
        ca = checkpoint.get('args', {})
        test_ids = ca.get('real_test_envs', ca.get('test_envs'))
    if test_ids is None:
        raise ValueError('Checkpoint does not identify its held-out domain')
    test_ids = flatten_ids(test_ids)
    targets = [domains[i] for i in test_ids]
    hp['cipt_clip_path'] = str(Path(args.clip_path).resolve())
    model = get_algorithm_class('CIPTDCCL')((3,224,224), len(classes),
                                          len(domains)-len(targets), hp)
    model.load_state_dict(state_from_checkpoint(checkpoint), strict=True)
    model.to(args.device).eval().requires_grad_(False)
    transform_path = Path(args.repo_root)/'domainbed/datasets/transforms.py'
    spec = importlib.util.spec_from_file_location('_native_cipt_transform', transform_path)
    transforms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transforms)
    base = Path(args.data_dir).expanduser().resolve()
    data_root = (base/'PACS' if (base/'PACS').is_dir() else base).resolve()
    images = []
    for row in rows:
        p = (data_root/row['relative_path']).resolve()
        if not p.is_file() or data_root not in p.parents:
            raise FileNotFoundError(f'Manifest image missing or outside data root: {p}')
        images.append(p)
    captures = {}
    def capture_e(module, inputs, output):
        captures.setdefault('e', []).append(output[0].detach())
    def capture_z(module, inputs, output):
        captures.setdefault('z', []).append(output.detach())
    hooks = [model.causal_decomposition.register_forward_hook(capture_e),
             model.tda.register_forward_hook(capture_z)]
    chunks = {key: [] for key in ['e','z_native','z_pred_mean','logits','predictions']}
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                tensor_list = []
                for p in images[start:start+args.batch_size]:
                    with Image.open(p) as image:
                        tensor_list.append(transforms.clip_basic(image))
                x = torch.stack(tensor_list).to(args.device)
                captures.clear()
                logits = model.predict(x)  # No labels passed to the inference path.
                if not isinstance(logits, torch.Tensor) or logits.shape != (len(x),len(classes)):
                    raise ValueError('Unexpected native prediction shape')
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite native logits')
                if len(captures.get('e',[])) != 1 or len(captures.get('z',[])) != 1:
                    raise ValueError('Native inference did not call E and TDA exactly once')
                e = F.normalize(captures['e'][0].float(),dim=-1)
                z = F.normalize(captures['z'][0].float(),dim=-1)
                pred = logits.argmax(-1)
                if expected_mode == 'b5b':
                    if z.ndim != 3 or z.shape[0] != len(x)*len(classes):
                        raise ValueError('Expected all-candidate B5B TDA output [B*C,K,D]')
                    z = z.reshape(len(x),len(classes),z.shape[1],z.shape[2])
                    selected = z[torch.arange(len(x),device=pred.device),pred]
                else:
                    if z.ndim != 3 or z.shape[0] != len(x):
                        raise ValueError('Expected shared B5A TDA output [B,K,D]')
                    selected = z
                summarized = F.normalize(selected.mean(dim=1),dim=-1)
                for key,value in [('e',e),('z_native',z),('z_pred_mean',summarized),
                                  ('logits',logits),('predictions',pred)]:
                    chunks[key].append(value.cpu().numpy())
                print(f'{model_id}: {start+len(x)}/{len(rows)}',flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    arrays = {k:np.concatenate(v) for k,v in chunks.items()}
    arrays.update(labels=np.array([int(r['class_id']) for r in rows]),
                  domains=np.array([r['domain'] for r in rows]),
                  relative_paths=np.array([r['relative_path'] for r in rows]))
    out = Path(args.output)
    np.savez_compressed(out/f'{model_id}_native.npz',**arrays)
    metadata = {'model_id':model_id,'checkpoint':str(Path(checkpoint_path).resolve()),
                'checkpoint_sha256':sha256(checkpoint_path),'hparams':hp,
                'class_names':classes,'target_domains':targets,
                'feature_extraction':'Forward hooks inside native model.predict(x); no ground-truth labels used',
                'z_native_shape':list(arrays['z_native'].shape),
                'z_plot_definition':'Predicted candidate branch (B5B) / shared branch (B5A), mean over normalized native K branches, then normalized',
                'coverage':'Fixed visualization manifest; sampled target accuracy is not formal DomainBed test_out accuracy',
                'source_coverage':'Source manifest may contain training images',
                'sample_manifest_sha256':sha256(args.manifest)}
    (out/f'{model_id}_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    del model, checkpoint, captures, chunks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_and_plot(output, seed=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    output = Path(output)
    ids = ['official','subject80']
    models = {k:dict(np.load(output/f'{k}_native.npz',allow_pickle=False)) for k in ids}
    metas = {k:json.loads((output/f'{k}_metadata.json').read_text()) for k in ids}
    a,b = [models[k] for k in ids]
    for key in ['relative_paths','labels','domains']:
        if not np.array_equal(a[key],b[key]):
            raise ValueError(f'The two models use different {key}')
    if metas['official']['target_domains'] != metas['subject80']['target_domains']:
        raise ValueError('Checkpoints hold out different domains')
    if metas['official']['class_names'] != metas['subject80']['class_names']:
        raise ValueError('Checkpoint class orders differ')
    target = np.isin(a['domains'],metas['official']['target_domains'])
    source = ~target
    if not source.any() or not target.any():
        raise ValueError('Both source and held-out target images are required')
    class_names = metas['official']['class_names']
    rows = []
    for name,m in models.items():
        for feature in ['e','z_pred_mean']:
            f=unit(m[feature].astype(float))
            centers=unit(np.stack([f[source & (m['labels']==c)].mean(0) for c in range(len(class_names))]))
            sims=f[target]@centers.T
            local_neighbors=np.argsort(-(f[target]@f[source].T),axis=1)[:,:5]
            rows.append({'model':name,'feature':feature,
                'sample_target_accuracy':float((m['predictions'][target]==m['labels'][target]).mean()),
                'sample_source_accuracy':float((m['predictions'][source]==m['labels'][source]).mean()),
                'target_to_same_class_source_centroid_cosine_distance':float((1-sims[np.arange(target.sum()),m['labels'][target]]).mean()),
                'target_to_source_5nn_class_agreement':float((m['labels'][source][local_neighbors]==m['labels'][target,None]).mean())})
    with (output/'sample_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    changes=[]
    for i in np.flatnonzero(target):
        changes.append({'relative_path':a['relative_paths'][i],'true_class':class_names[a['labels'][i]],
            'official_prediction':class_names[a['predictions'][i]],'subject80_prediction':class_names[b['predictions'][i]],
            'official_correct':bool(a['predictions'][i]==a['labels'][i]),
            'subject80_correct':bool(b['predictions'][i]==b['labels'][i])})
    with (output/'target_prediction_changes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    # One common projection for both methods and both feature types, then masks.
    feature_blocks=[models[k][feature] for k in ids for feature in ['e','z_pred_mean']]
    features=unit(np.concatenate(feature_blocks).astype(float))
    features=PCA(n_components=min(50,features.shape[0]-1,features.shape[1]),random_state=seed).fit_transform(features)
    kw={'n_components':2,'perplexity':min(30,len(features)-1),'init':'pca',
        'learning_rate':'auto','random_state':seed}
    kw['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1000
    xy=TSNE(**kw).fit_transform(features)
    n=len(a['labels']);blocks=np.split(xy,4)
    np.savez_compressed(output/'joint_projection.npz',official_e=blocks[0],official_z=blocks[1],
        subject80_e=blocks[2],subject80_z=blocks[3],labels=a['labels'],domains=a['domains'])
    lo,hi=xy.min(0),xy.max(0);center=(lo+hi)/2;span=max(hi-lo)*1.1
    limits=[(center[j]-span/2,center[j]+span/2) for j in [0,1]]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#6E60AA']
    domains=sorted(set(a['domains']));markers=dict(zip(domains,['o','s','^','D']))
    fig,axes=plt.subplots(2,4,figsize=(17,9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.98,top=.83,bottom=.20,wspace=.12,hspace=.20)
    for r,name in enumerate(ids):
        for c,(feature,mask) in enumerate([('e',source),('e',target),('z_pred_mean',source),('z_pred_mean',target)]):
            coords=blocks[r*2+(feature!='e')];ax=axes[r,c]
            for label in range(len(class_names)):
                for domain in domains:
                    take=mask & (a['labels']==label) & (a['domains']==domain)
                    ax.scatter(coords[take,0],coords[take,1],s=15,c=colors[label%len(colors)],
                        marker=markers[domain],alpha=.8,edgecolors='white',linewidths=.25)
            wrong=mask & (models[name]['predictions']!=a['labels'])
            ax.scatter(coords[wrong,0],coords[wrong,1],s=35,facecolors='none',edgecolors='#C72D2D',linewidths=.8)
            ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_aspect('equal',adjustable='box')
            ax.tick_params(labelleft=True,labelbottom=True,labelsize=8)
            ax.set_xlabel('t-SNE 1',fontsize=8);ax.set_ylabel('t-SNE 2',fontsize=8)
            if r==0:ax.set_title(['E: source','E: unseen target','Predicted-branch Z: source','Predicted-branch Z: unseen target'][c],fontsize=10,pad=12)
        pos=axes[r,0].get_position()
        fig.text(.018,(pos.y0+pos.y1)/2,['Original CIPT','Class-agnostic CIPT'][r],rotation=90,ha='center',va='center',fontsize=11,weight='bold')
    fig.suptitle('Native inference: E and fused Z branches',fontsize=17,y=.965)
    fig.text(.52,.923,'Identical images | one joint projection | shared axis limits | red rings: native prediction errors',ha='center',fontsize=10)
    handles=[Line2D([],[],marker='o',linestyle='',color=colors[i%len(colors)],label=name) for i,name in enumerate(class_names)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.125),ncol=len(class_names),frameon=False,fontsize=9)
    handles=[Line2D([],[],marker=markers[d],linestyle='',color='#556476',label=d) for d in domains]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.085),ncol=len(domains),frameon=False,fontsize=8)
    fig.text(.52,.052,'Z panels summarize the model-predicted candidate, averaged over native K branches; ground-truth labels only color points.',ha='center',fontsize=8.5)
    fig.text(.52,.027,'Z is a diagnostic projection. Report formal DomainBed test_out accuracy separately; these images are a fixed visualization sample.',ha='center',fontsize=8.5)
    for extension in ['png','svg']:
        fig.savefig(output/f'native_E_Z_source_target.{extension}',dpi=250,facecolor='white')
    plt.close(fig)
    print(json.dumps(rows,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    extract=sub.add_parser('extract',help='Load checkpoints, capture predict() features, then plot')
    for name in ['repo-root','data-dir','clip-path','manifest','official-checkpoint','subject-checkpoint','output']:
        extract.add_argument('--'+name,required=True)
    extract.add_argument('--batch-size',type=int,default=16)
    extract.add_argument('--device',default='cuda')
    extract.add_argument('--seed',type=int,default=0)
    plot=sub.add_parser('plot',help='Replot a completed native extraction without checkpoints')
    plot.add_argument('--output',required=True);plot.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.command=='extract':
        sys.path.insert(0,str(Path(args.repo_root).expanduser().resolve()))
        import random,torch
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run on the training server')
        if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        out=Path(args.output)
        if out.exists() and any(out.iterdir()):
            raise ValueError('Use an empty output directory; existing results will not be overwritten')
        rows=read_manifest(args.manifest);out.mkdir(parents=True,exist_ok=True)
        extract_one(args,rows,args.official_checkpoint,'official','b5b')
        extract_one(args,rows,args.subject_checkpoint,'subject80','b5a')
    summarize_and_plot(args.output,args.seed)


if __name__=='__main__':
    main()
