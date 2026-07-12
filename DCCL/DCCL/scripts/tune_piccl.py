#!/usr/bin/env python3
"""Leakage-safe PICCL hyperparameter search wrapper.

Runs the existing train_all.py entrypoint, parses results.jsonl, and ranks trials
only by source-domain out-split accuracy. Target-domain metrics are reported but
never used for ranking or pruning.
"""
import argparse, csv, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path

STATUS_DONE={"completed","skipped"}

def load_jsonl(path):
    rows=[]
    p=Path(path)
    if not p.exists(): return rows
    for line in p.read_text(errors='ignore').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def infer_target_envs(row):
    args=row.get('args',{})
    real=args.get('real_test_envs')
    if real is not None: return [int(x) for x in real]
    te=args.get('test_envs') or []
    if te and isinstance(te[0], list) and len(te)==1: return [int(x) for x in te[0]]
    return []

def source_objective(row):
    target=set(infer_target_envs(row))
    vals=[]
    for k,v in row.items():
        if k.startswith('env') and k.endswith('_out'):
            try: env=int(k[3:].split('_')[0])
            except ValueError: continue
            if env not in target: vals.append(float(v))
    if not vals: return float('nan')
    mean=sum(vals)/len(vals)
    var=sum((x-mean)**2 for x in vals)/len(vals)
    return mean - 0.2*math.sqrt(var)

def best_and_final(results_path):
    rows=load_jsonl(results_path)
    if not rows: return None,None
    best=max(rows, key=lambda r: (source_objective(r) if math.isfinite(source_objective(r)) else -1e9, r.get('step',-1)))
    final=max(rows, key=lambda r: r.get('step',-1))
    return best, final

def detect_failure(code, stdout, stderr, trial_dir):
    text=(stdout+'\n'+stderr).lower()
    if code == 0: return ''
    if 'out of memory' in text or 'cuda oom' in text: return 'oom'
    if 'nan' in text or 'inf' in text: return 'nan_or_inf'
    if not (Path(trial_dir)/'results.jsonl').exists(): return 'no_results'
    return f'exit_{code}'

def sample_params(rng, space):
    params={}
    for name, spec in space.items():
        if isinstance(spec, list): params[name]=rng.choice(spec)
        elif isinstance(spec, dict) and spec.get('type')=='loguniform':
            lo,hi=math.log10(spec['min']),math.log10(spec['max']); params[name]=10**rng.uniform(lo,hi)
        elif isinstance(spec, dict) and spec.get('type')=='uniform':
            params[name]=rng.uniform(spec['min'],spec['max'])
        else: params[name]=spec
    return params

def build_command(cfg, params, trial_dir, budget_steps):
    train=cfg.get('train_script','train_all.py')
    cmd=[sys.executable, train, Path(trial_dir).name]
    cmd += cfg.get('base_args', [])
    cmd += ['--steps', str(budget_steps)]
    for k,v in params.items():
        cmd += [f'--{k}', str(v).lower() if isinstance(v,bool) else str(v)]
    return cmd

def write_csv(path, rows):
    if not rows: return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/piccl_hpo.json')
    ap.add_argument('--output', default='train_output/piccl_hpo')
    ap.add_argument('--trials', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--max-concurrent', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--backend', choices=['random','optuna'], default='random')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    rng=random.Random(args.seed if args.seed is not None else cfg.get('seed',0))
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    n=args.trials or cfg.get('trials',24)
    budgets=cfg.get('budgets',[cfg.get('steps',500)])
    summaries=[]
    if args.backend=='optuna':
        try: import optuna  # noqa: F401
        except Exception: print('Optuna unavailable; falling back to standard-library random search.', file=sys.stderr)
    for i in range(n):
        trial_dir=out/f'trial_{i+1:04d}'
        params=sample_params(rng, cfg['search_space'])
        status='completed'; failure=''; best=None; final=None
        for budget in budgets:
            trial_dir.mkdir(parents=True, exist_ok=True)
            done=trial_dir/'DONE.json'
            if args.resume and done.exists():
                rec=json.loads(done.read_text()); best=rec.get('best'); final=rec.get('final'); status='skipped'; continue
            cmd=build_command(cfg, params, trial_dir, int(budget))
            (trial_dir/'command.sh').write_text(' '.join(cmd)+'\n')
            if args.dry_run:
                status='dry_run'; break
            env=os.environ.copy()
            if args.gpu is not None: env['CUDA_VISIBLE_DEVICES']=args.gpu
            proc=subprocess.run(cmd, cwd=cfg.get('cwd','.'), env=env, text=True, capture_output=True, timeout=args.timeout)
            (trial_dir/'stdout.txt').write_text(proc.stdout)
            (trial_dir/'stderr.txt').write_text(proc.stderr)
            failure=detect_failure(proc.returncode, proc.stdout, proc.stderr, trial_dir)
            if failure: status='failed'; break
            # Find actual timestamped train_all output by newest matching name.
            candidates=sorted(Path(cfg.get('cwd','.')).glob(f"train_output/{cfg.get('dataset','PACS')}/*{trial_dir.name}*"), key=lambda p:p.stat().st_mtime)
            result_path=(candidates[-1]/'results.jsonl') if candidates else trial_dir/'results.jsonl'
            best, final=best_and_final(result_path)
            if result_path.exists() and not (trial_dir/'results.jsonl').exists(): shutil.copy2(result_path, trial_dir/'results.jsonl')
            (trial_dir/'DONE.json').write_text(json.dumps({'status':status,'failure':failure,'params':params,'best':best,'final':final}, indent=2))
        row={'trial':i+1,'status':status,'failure':failure,**params}
        if best:
            row.update({'best_step':best.get('step'),'objective':source_objective(best),'target_test_out_report_only':best.get('test_out')})
        if final: row.update({'final_step':final.get('step'),'final_objective':source_objective(final)})
        summaries.append(row)
        with open(out/'study_state.jsonl','a') as f: f.write(json.dumps(row, sort_keys=True)+'\n')
    write_csv(out/'trials.csv', summaries)
    ranked=[r for r in summaries if isinstance(r.get('objective'),float) and math.isfinite(r['objective'])]
    ranked.sort(key=lambda r:r['objective'], reverse=True)
    if ranked:
        best=ranked[0]
        (out/'best_config.json').write_text(json.dumps({k:v for k,v in best.items() if k in cfg['search_space']}, indent=2))
        (out/'best_command.sh').write_text(' '.join(build_command(cfg,{k:best[k] for k in cfg['search_space']}, out/'best_full', cfg.get('full_steps',5001)))+'\n')
    (out/'summary.md').write_text('# PICCL HPO Summary\n\nObjective: mean(source env out_acc) - 0.2*std(source env out_acc). Target env is report-only.\n')
    print(json.dumps({'output':str(out),'best': ranked[0] if ranked else None}, indent=2))
if __name__=='__main__': main()
