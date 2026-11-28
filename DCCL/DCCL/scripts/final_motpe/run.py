import fcntl
import itertools
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from statistics import mean

import optuna
from optuna.trial import TrialState as State
import core

p, runner = core.p, core.runner
ROOT, DATASET, NAME = core.ROOT, core.DATASET, core.NAME
GPU = core.GPU
SPACE = dict(s_weight=[.5,1.,2.],condcov_weight=[1.,2.,3.,5.],
             rd_weight=[.5,1.,2.,3.],contrastive_weight=[.01,.025,.05])
FIRST = dict(s_weight=1.,condcov_weight=2.,rd_weight=1.,contrastive_weight=.025)
COMBOS = list(itertools.product(*SPACE.values()))
TARGET = dict(PACS=.975,OfficeHome=.86,VLCS=.845,TerraIncognita=.58)[DATASET]
LABEL = 'MOTPE / Oracle Exploration'

def key(params):
    return tuple(params[k] for k in SPACE)

def status(state, **details):
    p.write_json(ROOT/'status.json',dict(state=state,dataset=DATASET,gpu=GPU,controller_pid=os.getpid(),**details))
    (ROOT/'sync.request').touch()

def log(message):
    with (ROOT/'decision_log.md').open('a') as stream:
        stream.write(message+'\n')

def reports(study):
    complete=study.get_trials(states=(State.COMPLETE,))
    front={t.number for t in study.best_trials}
    results=[]
    folds=[]
    for trial in complete:
        result=trial.user_attrs['result']
        results.append(dict(trial=trial.number,origin=trial.user_attrs.get('origin','motpe'),
             **trial.params,full_hparams=json.dumps(core.hp(trial.params),sort_keys=True),
             **{k:v for k,v in result.items() if k!='folds'},pareto=trial.number in front))
        folds.extend(dict(trial=trial.number,fold=f['fold'],domain=f['domain'],seed=f['seed'],
                          selected_step=f['selected_step'],source_val=f['source_val'],iid=f['iid'],
                          test_in=f['test_in'],test_out=f['test_out'],updates=f['updates'],
                          elapsed_seconds=f['elapsed_seconds'],last_adam_lr=f['last_adam_lr'],
                          final_scheduler_lr=f['final_scheduler_lr'],losses=json.dumps(f['losses']),
                          log=f['log'],learning_rates=f['learning_rates']) for f in result['folds'])
    fields=['trial','origin',*SPACE,'full_hparams','source_val','worst_fold_source_val','iid','test_in','test_out','elapsed_seconds','pareto']
    p.write_csv(ROOT/'trials_summary.csv',results,fields)
    pareto=[r for r in results if r['pareto']]
    p.write_csv(ROOT/'pareto_front.csv',pareto,fields)
    p.write_json(ROOT/'pareto_front.json',pareto)
    p.write_csv(ROOT/'folds_summary.csv',folds,['trial','fold','domain','seed','selected_step','source_val','iid','test_in','test_out','updates','elapsed_seconds','last_adam_lr','final_scheduler_lr','losses','log','learning_rates'])
    if results:
        p.write_json(ROOT/'best_source_val.json',max(results,key=lambda r:(r['source_val'],r['iid'])))
        best=max(results,key=lambda r:(r['iid'],r['source_val']))
        p.write_json(ROOT/'best_target_iid.json',dict(best,target=TARGET,gap_to_target=TARGET-best['iid']))
    return complete

def run_fold(trial, folder, fold, lock):
    marker=folder/f'fold_{fold}.json'
    candidates=sorted(folder.glob(f'env{fold}/attempt_*'))
    successful=[a for a in candidates if (a/'exit_code').exists() and (a/'exit_code').read_text().strip()=='0']
    if successful:
        result=core.read_attempt(successful[-1],DATASET,fold,trial.params)
    else:
        assert not marker.exists(),f'missing successful attempt for {marker}'
        parent=folder/f'env{fold}'
        parent.mkdir(exist_ok=True)
        attempt=parent/f'attempt_{len(candidates)+1:03d}'
        status('running',trial=trial.number,fold=fold,domain=p.DATASETS[DATASET][0][fold],params=trial.params)
        runner.start_attempt(attempt,DATASET,fold,trial.number,folder/'config.json',GPU,runner.environment_for(runner.gpu_uuids()[GPU]),lock.fileno())
        result=core.read_attempt(attempt,DATASET,fold,trial.params)
    p.write_json(marker,result)
    (ROOT/'sync.request').touch()
    return result

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rng=ROOT/'sampler.pkl'
        settings=dict(seed=42,n_startup_trials=10,n_ei_candidates=24,multivariate=True)
        sampler=pickle.loads(rng.read_bytes()) if rng.exists() else optuna.samplers.TPESampler(**settings)
        study=optuna.create_study(study_name=NAME+'_'+DATASET,storage=f"sqlite:///{ROOT/'study.sqlite3'}",
             load_if_exists=True,directions=['maximize','maximize'],sampler=sampler,pruner=optuna.pruners.NopPruner())
        metadata=dict(dataset=DATASET,label=LABEL,objectives=['mean_source_val','mean_target_iid_at_source_selected_checkpoint'],
             directions=['maximize','maximize'],sampler=settings,pruner='NopPruner',first=FIRST,space_size=len(COMBOS),
             target=TARGET,stop_rules=['target_iid','search_space_exhausted'],gpu=GPU,checkpoint_selection='earliest_max_source_val',
             fixed_args=p.FIXED_ARGS,initial_hparams=core.hp(FIRST),seeds=[p.seed_for(DATASET,i) for i in range(4)],
             lr_policy=p.LR_POLICY,eval_cache=True,training_commit=p.git('rev-parse','HEAD'))
        p.write_json(ROOT/'protocol.json',metadata)
        p.write_json(ROOT/'search_space.json',SPACE)
        if not study.trials:
            study.enqueue_trial(FIRST,user_attrs={'origin':'requested_first_candidate'})
        while True:
            completed=reports(study)
            used={key(t.params) for t in completed}
            best=max((t.values[1] for t in completed),default=None)
            progress=dict(completed=len(used),space_size=len(COMBOS),coverage=len(used)/len(COMBOS),best_iid=best,target=TARGET,
                          gap_to_target=None if best is None else TARGET-best)
            if best is not None and best>=TARGET:
                status('TARGET_REACHED',**progress)
                return
            if len(used)==len(COMBOS):
                sensitivity={k:[dict(value=v,mean_source_val=mean(t.values[0] for t in completed if t.params[k]==v),
                                     mean_target_iid=mean(t.values[1] for t in completed if t.params[k]==v)) for v in choices]
                             for k,choices in SPACE.items()}
                p.write_json(ROOT/'parameter_sensitivity.json',sensitivity)
                p.write_text(ROOT/'next_stage_analysis.md','Review boundary values in parameter_sensitivity.json before proposing extensions. If performance improves toward an upper bound, consider testing above that bound in a separate study.\n')
                status('SEARCH_SPACE_EXHAUSTED',**progress)
                return
            if (ROOT/'pause.request').exists():
                status('PAUSED',**progress)
                return
            active=study.get_trials(states=(State.RUNNING,))
            if len(active)>1:
                raise RuntimeError('multiple RUNNING trials')
            trial=optuna.Trial(study,active[0]._trial_id) if active else study.ask()
            params={k:trial.suggest_categorical(k,v) for k,v in SPACE.items()}
            if key(params) in used:
                trial.set_user_attr('origin','duplicate_suggestion')
                study.tell(trial,state=State.FAIL)
                candidate=random.Random(42+len(study.trials)).choice([c for c in COMBOS if c not in used])
                study.enqueue_trial(dict(zip(SPACE,candidate)),user_attrs={'origin':'unsearched_fallback'})
                log(f'Trial {trial.number}: duplicate skipped. Fallback candidate {candidate}.')
                continue
            saved=rng.with_suffix('.tmp')
            saved.write_bytes(pickle.dumps(sampler))
            saved.replace(rng)
            folder=ROOT/DATASET/'trials'/f'trial_{trial.number:03d}'
            folder.mkdir(parents=True,exist_ok=True)
            config=folder/'config.json'
            if config.exists() and p.read_json(config)!=core.hp(params):
                raise RuntimeError(f'persisted configuration mismatch: {config}')
            p.write_json(config,core.hp(params))
            try:
                folds=[run_fold(trial,folder,fold,lock) for fold in range(4)]
            except (ValueError,KeyError,OSError) as exc:
                error=f'{type(exc).__name__}: {exc}'
                p.write_json(ROOT/'validation_error.json',dict(trial=trial.number,error=error))
                status('PAUSED_VALIDATION_ERROR',trial=trial.number,error=error,**progress)
                log(error)
                return
            result=dict(p.summarize(folds),worst_fold_source_val=min(f['source_val'] for f in folds))
            p.write_json(folder/'result.json',result)
            trial.set_user_attr('result',result)
            study.tell(trial,[result['source_val'],result['iid']])
            log(f'Trial {trial.number} complete with objectives {[result["source_val"],result["iid"]]}.')
            (ROOT/'sync.request').touch()

if __name__=='__main__':
    main()

