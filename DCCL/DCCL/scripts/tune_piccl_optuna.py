#!/usr/bin/env python3
"""Log-derived Optuna TPE tuner for PACS PICCL.

Target-domain accuracy is recorded as report-only and is never used in the
objective, ranking, pruning, or best-config selection.
"""
import argparse, csv, json, math, os, shutil, subprocess, sys
from pathlib import Path
from statistics import mean, pstdev

PICCL_KEYS = ["piccl_rank", "piccl_beta_max", "piccl_isr_weight", "use_piccl"]


def load_jsonl(path):
    p=Path(path)
    return [json.loads(l) for l in p.read_text(errors='ignore').splitlines() if l.strip()] if p.exists() else []

def final_row(run_dir):
    rows=load_jsonl(Path(run_dir)/'results.jsonl')
    if not rows: raise FileNotFoundError(Path(run_dir)/'results.jsonl')
    return max(rows, key=lambda r: r.get('step', -1))

def target_env(row):
    real=row.get('args',{}).get('real_test_envs')
    if real: return int(real[0])
    te=row.get('args',{}).get('test_envs') or []
    if te and isinstance(te[0], list): return int(te[0][0])
    return 0

def source_score(row):
    tgt=target_env(row); vals=[]
    for i in range(4):
        if i != tgt and f'env{i}_out' in row: vals.append(float(row[f'env{i}_out']))
    if not vals: return float('nan')
    return mean(vals) - 0.2*pstdev(vals)

def global_objective(scores):
    vals=[float(scores[i]['source_score']) for i in range(4)]
    return mean(vals) - 0.2*pstdev(vals)

def recover_params(row):
    hp=row.get('hparams',{}); args=row.get('args',{}); out={}
    for k in PICCL_KEYS:
        if k in hp: out[k]=hp[k]
        elif k in args: out[k]=args[k]
    return out

def read_history(history_root):
    runs=[]
    root=Path(history_root)
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        row=final_row(d); params=recover_params(row); tgt=target_env(row)
        runs.append({'name': d.name, 'algorithm': row.get('args', {}).get('algorithm'),
                     'step': row.get('step'), 'target_env': tgt, 'params': params,
                     'source_score': source_score(row),
                     'source_mean': mean([float(row[f'env{i}_out']) for i in range(4)
                                          if i != tgt and f'env{i}_out' in row]),
                     'target_report_only': row.get(f"env{tgt}_out"),
                     'train_out': row.get('train_out'),
                     'metrics': {k: row.get(k) for k in ['loss_cls', 'weighted_loss_isr',
                                 'loss_isr_aug', 'loss_isr_dom', 'piccl_beta', 'has_nan_or_inf'] if k in row}})
    return runs

def suggest_params(trial, cfg):
    params=dict(cfg.get('fixed_params',{}))
    for name,spec in cfg['search_space'].items():
        if spec.get('type')=='categorical': params[name]=trial.suggest_categorical(name, spec['choices'])
        else: params[name]=trial.suggest_float(name, spec['low'], spec['high'], log=bool(spec.get('log',False)))
    return params

def params_to_cli(params):
    cli=[]
    for k,v in params.items(): cli += [f'--{k}', str(v).lower() if isinstance(v,bool) else str(v)]
    return cli

def build_env_command(args, cfg, params, trial_dir, test_env, gpu=None):
    name=f"trial_{int(trial_dir.name.split('_')[-1]):04d}_env{test_env}"
    cmd=[sys.executable,'train_all.py',name,'--dataset',cfg.get('dataset','PACS'),'--algorithm','PICCL','--model',cfg.get('model','resnet50'),'--data_dir',str(args.data_dir),'--output_root',str(trial_dir/f'env{test_env}'),'--steps',str(args.steps),'--checkpoint_freq',str(args.checkpoint_freq),'--trial_seed',str(args.seed),'--seed',str(args.seed),'--test_envs',str(test_env),'--deterministic']
    cmd += params_to_cli(params)
    return cmd

def latest_results(env_dir):
    c=sorted(Path(env_dir).glob('*/results.jsonl'), key=lambda p:p.stat().st_mtime)
    return c[-1] if c else None

def evaluate_trial_outputs(trial_dir):
    env_scores={}; target={}; rows=[]
    for t in range(4):
        rp=latest_results(Path(trial_dir)/f'env{t}')
        if not rp: raise RuntimeError(f'missing results for env{t}')
        row=final_row(rp.parent); rows.append(row)
        if row.get('has_nan_or_inf',0): raise RuntimeError(f'nan_or_inf env{t}')
        env_scores[t]={'source_score':source_score(row),'source_mean':mean([float(row[f'env{i}_out']) for i in range(4) if i!=t and f'env{i}_out' in row]),'source_std':pstdev([float(row[f'env{i}_out']) for i in range(4) if i!=t and f'env{i}_out' in row])}
        target[t]=row.get(f'env{t}_out')
    obj=global_objective(env_scores)
    return obj, env_scores, target, rows

def write_csv(path, rows):
    if not rows: return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with open(path,'w',newline='') as f: w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)

def create_study(cfg, output_root):
    import optuna
    ocfg=cfg.get('optuna',{})
    sampler=optuna.samplers.TPESampler(seed=ocfg.get('seed',0), n_startup_trials=ocfg.get('n_startup_trials',10), multivariate=ocfg.get('multivariate',True), group=ocfg.get('group',True))
    return optuna.create_study(study_name=cfg.get('study_name','piccl_pacs_from_logs'), direction=ocfg.get('direction','maximize'), sampler=sampler, storage=f"sqlite:///{Path(output_root)/'study.db'}", load_if_exists=True)

def enqueue_history(study, cfg, history):
    wanted=set(cfg.get('historical_trial_names',[])); queued=[]
    for run in history:
        if run['name'] not in wanted or run['algorithm']!='PICCL': continue
        p={k:run['params'][k] for k in cfg['search_space'] if k in run['params']}
        if set(p)==set(cfg['search_space']):
            study.enqueue_trial(p, user_attrs={'historical_run':run['name'],'historical_negative_anchor': run['source_score'] < 0.966})
            queued.append(p)
    return queued

def objective_factory(args,cfg):
    def objective(trial):
        params=suggest_params(trial,cfg); trial_dir=Path(args.output_root)/'trials'/f'trial_{trial.number:04d}'; trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir/'params.json').write_text(json.dumps(params,indent=2,sort_keys=True))
        cmds=[build_env_command(args,cfg,params,trial_dir,t) for t in range(4)]
        (trial_dir/'command.sh').write_text('\n'.join(' '.join(c) for c in cmds)+'\n')
        if args.dry_run: raise RuntimeError('dry_run must not optimize')
        failure=''
        try:
            pending=[]
            gpu_ids=[g for g in args.gpus.split(',') if g] or [None]
            for t,cmd in enumerate(cmds):
                if args.resume and latest_results(trial_dir/f'env{t}'):
                    continue
                env=os.environ.copy()
                gpu=gpu_ids[t % len(gpu_ids)]
                if gpu is not None:
                    env['CUDA_VISIBLE_DEVICES']=gpu
                pending.append((t, subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)))
                if len(pending) >= max(1, args.max_concurrent):
                    t0, proc0 = pending.pop(0)
                    stdout, stderr = proc0.communicate()
                    (trial_dir/f'env{t0}_stdout.log').write_text(stdout)
                    (trial_dir/f'env{t0}_stderr.log').write_text(stderr)
                    if proc0.returncode != 0:
                        raise RuntimeError(f'env{t0}_exit_{proc0.returncode}')
            for t0, proc0 in pending:
                stdout, stderr = proc0.communicate()
                (trial_dir/f'env{t0}_stdout.log').write_text(stdout)
                (trial_dir/f'env{t0}_stderr.log').write_text(stderr)
                if proc0.returncode != 0:
                    raise RuntimeError(f'env{t0}_exit_{proc0.returncode}')
            obj,scores,target,rows=evaluate_trial_outputs(trial_dir)
            metrics={'status':'completed','failure_reason':'','source_scores':scores,'target_report_only':target,'global_objective':obj,'git_commit_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
            (trial_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True))
            with open(trial_dir/'results.jsonl','w') as f:
                for r in rows: f.write(json.dumps(r)+'\n')
            for k,v in {'target_report_only':target,'source_scores':scores}.items(): trial.set_user_attr(k,v)
            return obj
        except Exception as e:
            failure=str(e); (trial_dir/'metrics.json').write_text(json.dumps({'status':'failed','failure_reason':failure},indent=2)); raise
    return objective

def dry_run(args,cfg,history):
    out=Path(args.output_root); print(json.dumps({'historical_runs':history,'search_space':cfg['search_space'],'fixed_params':cfg.get('fixed_params',{}),'sampler':'TPESampler(seed=0,n_startup_trials=10,multivariate=True,group=True)','n_trials':args.n_trials,'commands':[ ' '.join(build_env_command(args,cfg,{**cfg.get('fixed_params',{}), **{k:('SUGGESTED') for k in cfg['search_space']}}, out/'trials'/'trial_0000', t)) for t in range(4)],'output_root':str(out),'sqlite':str(out/'study.db')}, indent=2, ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--history-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--data-dir',required=True)
    ap.add_argument('--gpus',default=''); ap.add_argument('--max-concurrent',type=int,default=1); ap.add_argument('--n-trials',type=int,default=40); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--checkpoint-freq',type=int,default=100); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--resume',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(); cfg=json.loads(Path(args.config).read_text()); out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True); (out/'trials').mkdir(exist_ok=True)
    history=read_history(args.history_root); (out/'historical_runs.json').write_text(json.dumps(history,indent=2,sort_keys=True)); (out/'resolved_search_space.json').write_text(json.dumps({'search_space':cfg['search_space'],'fixed_params':cfg.get('fixed_params',{})},indent=2,sort_keys=True))
    if args.dry_run: dry_run(args,cfg,history); return
    study=create_study(cfg,out); enqueue_history(study,cfg,history); study.optimize(objective_factory(args,cfg), n_trials=args.n_trials)
    rows=[{'trial':t.number,'status':t.state.name,'objective':t.value,**t.params} for t in study.trials]
    write_csv(out/'trials.csv',rows); ranked=[r for r in rows if r['status']=='COMPLETE' and r.get('objective') is not None]; ranked.sort(key=lambda r:r['objective'],reverse=True); write_csv(out/'ranking.csv',ranked)
    if ranked:
        best={**cfg.get('fixed_params',{}), **{k:ranked[0][k] for k in cfg['search_space']}}
        (out/'best_config.json').write_text(json.dumps(best,indent=2,sort_keys=True)); (out/'best_command.sh').write_text('\n'.join(' '.join(build_env_command(args,cfg,best,out/'best_full',t)) for t in range(4))+'\n')
    (out/'summary.md').write_text(f"# PICCL PACS Optuna summary\n\nTrials: {len(rows)}\nBest: {ranked[0] if ranked else None}\n")
if __name__=='__main__': main()
