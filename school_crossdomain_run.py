import os,sys,json,time,itertools,subprocess,fcntl,shlex,pickle,threading,shutil,random
from pathlib import Path
from statistics import mean
import optuna
from optuna.trial import TrialState
BASE=Path(__file__).resolve().parent
CODE=BASE/'code/DCCL/DCCL'
NAME='condcov_rd_crossdomain_constant1e5_optuna_20261128'
REMOTE='git@github.com:r97993346-design/DCCL-3Source-DomainNet.git'
BRANCH='results/'+NAME
sys.path.insert(0,str(CODE/'scripts/cipt_clip_supcon_optuna'))
import protocol as p
SPACE={'s_weight':[.5,.75,1.], 'condcov_weight':[1.,2.,3.], 'rd_weight':[1.,2.,3.], 'contrastive_weight':[.01,.025,.05]}
COMBOS=list(itertools.product(*SPACE.values()))
def write(path,x):p.write_json(path,x)
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def hp(v):
    return dict(p.fixed_hparams(),lr=1e-5,cipt_lr_schedule='constant',cipt_e_cls_weight=1.,cipt_s_uniform_weight=v['s_weight'],cipt_condcov_weight=v['condcov_weight'],cipt_env_loss_weight=v['rd_weight'],cipt_causal_contrastive_weight=v['contrastive_weight'],cipt_contrastive_weight=v['contrastive_weight'],cipt_prompt_length=8,cipt_clip_teacher_mix=.25,cipt_contrastive_type='clip_guided_cross_domain',cipt_clip_teacher_topk=0,ind_grad_steps=[],cipt_neighbor_diagnostics=False)
def signature(v):return tuple(v[k] for k in SPACE)
def sync():
    publication=BASE/'publication';publication.mkdir(exist_ok=True)
    with (BASE/'.sync.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        def git(*a):return subprocess.run(['git','-C',str(publication),*a],check=True,capture_output=True,timeout=60,env=dict(os.environ,GIT_SSH_COMMAND='ssh -o BatchMode=yes',GIT_TERMINAL_PROMPT='0'))
        if not (publication/'.git').exists():
            git('init','-q');git('checkout','-b',BRANCH);git('config','user.name','CIPT experiments');git('config','user.email','cipt@localhost')
        for ds in ['PACS','OfficeHome']:
            root=BASE/'outputs'/ds
            if not root.exists():continue
            for path in root.rglob('*'):
                if path.is_file() and path.suffix in ['.json','.jsonl','.csv','.log']:
                    dest=publication/NAME/path.relative_to(BASE/'outputs');dest.parent.mkdir(parents=True,exist_ok=True)
                    data=path.read_bytes()
                    if path.suffix=='.jsonl' and data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
                    dest.write_bytes(data)
        git('add','.')
        if subprocess.run(['git','-C',str(publication),'diff','--cached','--quiet']).returncode:
            git('commit','-qm','Sync school cross-domain constant LR experiments')
        if (publication/'.git/refs/heads'/BRANCH).exists():git('push',REMOTE,'HEAD:refs/heads/'+BRANCH)
def pulse():
    while True:
        try:sync()
        except Exception as e:print('RESULT_SYNC_RETRY',str(e),flush=True)
        time.sleep(300)
def validate(attempt,ds,fold,v):
    logs=list((attempt/'raw'/ds).glob('*/results.jsonl'));assert len(logs)==1
    data=rows(logs[0]);trace=rows(attempt/'learning_rates.jsonl')
    assert [x['step'] for x in data]==[*range(0,5001,300),5000]
    assert len(trace)==5001 and all(t['step']==i and t['lr_used']==1e-5 and t['scheduler_updates']==0 for i,t in enumerate(trace))
    for x in data:
        assert all(x['hparams'].get(k)==a for k,a in hp(v).items())
        assert x['args']['seed']==p.seed_for(ds,fold) and x['e_only_infer']==1
        assert x['contrastive_weight']==v['contrastive_weight'] and x['ind_weight']==v['condcov_weight']
    best=max(data,key=lambda x:x['train_out'])
    assert not list(attempt.rglob('*.pth'))
    return dict(fold=fold,domain=p.DATASETS[ds][0][fold],seed=p.seed_for(ds,fold),selected_step=best['step'],source_val=best['train_out'],iid=best['test_in'],test_in=best['test_in'],test_out=best['test_out'],updates=5001,log=str(logs[0]))
def run(ds,gpu):
    root=BASE/'outputs'/ds;root.mkdir(parents=True,exist_ok=True)
    with (root/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sampler_path=root/'sampler.pkl'
        sampler=pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else optuna.samplers.TPESampler(seed=42,n_startup_trials=5,n_ei_candidates=24,multivariate=False)
        study=optuna.create_study(study_name=NAME+'_'+ds,storage='sqlite:///'+str(root/'study.sqlite3'),load_if_exists=True,direction='maximize',sampler=sampler,pruner=optuna.pruners.NopPruner())
        first={'s_weight':1.,'condcov_weight':2. if ds=='PACS' else 1.,'rd_weight':1.,'contrastive_weight':.025 if ds=='PACS' else .05}
        if not study.trials:study.enqueue_trial(first)
        write(root/'protocol.json',dict(name=NAME,dataset=ds,space=SPACE,combinations=81,gpu=gpu,fixed_hparams=hp(first),steps=5001,checkpoint_freq=300,trial_seed=1,source_selection='earliest_max_source_val',objective='mean_four_fold_source_val',num_workers=2,eval_num_workers=0,registry=False,target_iid=p.DATASETS[ds][1],target_policy='report_only_pending_user_clarification',sampler=dict(type='TPESampler',seed=42,n_startup_trials=5,n_ei_candidates=24,multivariate=False),training_commit=p.git('rev-parse','HEAD',cwd=CODE)))
        threading.Thread(target=pulse,daemon=True).start()
        while True:
            used={signature(t.params) for t in study.trials if set(t.params)==set(SPACE) and t.user_attrs.get('trained',False)}
            active=[t for t in study.trials if t.state==TrialState.RUNNING]
            if active:trial=optuna.Trial(study,active[0]._trial_id);v=dict(trial.params)
            else:
                remaining=[c for c in COMBOS if c not in used]
                if not remaining:
                    write(root/'status.json',dict(state='EXHAUSTED',unique_configurations=len(used)));break
                trial=study.ask();v={k:trial.suggest_categorical(k,choices) for k,choices in SPACE.items()}
                if signature(v) in used:
                    trial.set_user_attr('duplicate_suggestion',True);study.tell(trial,state=TrialState.FAIL)
                    study.enqueue_trial(dict(zip(SPACE,remaining[0])))
                    continue
                trial.set_user_attr('trained',True)
                sampler_path.write_bytes(pickle.dumps(study.sampler))
            folder=root/'trials'/f'trial_{trial.number:03d}';folder.mkdir(parents=True,exist_ok=True)
            write(folder/'config.json',hp(v));folds=[]
            for fold in range(4):
                marker=folder/f'fold_{fold}.json'
                if marker.exists():folds.append(read(marker));continue
                attempts=sorted(folder.glob(f'env{fold}/attempt_*'))
                complete=[a for a in attempts if (a/'exit_code').exists() and (a/'exit_code').read_text().strip()=='0']
                if complete:result=validate(complete[-1],ds,fold,v)
                else:
                    while True:
                        ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])
                        free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                        if ram>10*1024**2 and free>=8192:break
                        write(root/'status.json',dict(state='WAITING_RESOURCES',gpu=gpu,free_vram_mib=free,available_ram_kib=ram));time.sleep(30)
                    attempt=folder/f'env{fold}'/f'attempt_{len(attempts)+1:03d}';attempt.mkdir(parents=True)
                    args=p.command(ds,fold,trial.number,folder/'config.json',attempt/'raw',attempt/'learning_rates.jsonl')
                    args[3]=str(BASE/'school_constant_instrument.py')
                    args+=['--cipt_cache_eval_visual']
                    wrapper=attempt/'command.sh'
                    wrapper.write_text('#!/bin/bash\ncd '+shlex.quote(str(CODE))+'\n'+shlex.join(args)+'\nrc=$?\necho "$rc" > '+shlex.quote(str(attempt/'exit_code'))+'\nexit "$rc"\n')
                    with (attempt/'console.log').open('wb') as log:
                        child=subprocess.Popen(['bash',str(wrapper)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),pass_fds=(lock.fileno(),))
                    write(root/'status.json',dict(state='RUNNING',gpu=gpu,trial=trial.number,params=v,fold=fold,completed_folds=len(folds),pid=child.pid))
                    rc=child.wait()
                    if rc:
                        write(root/'status.json',dict(state='FAILED_REQUIRES_REVIEW',trial=trial.number,fold=fold,exit_code=rc));return
                    result=validate(attempt,ds,fold,v)
                write(marker,result);folds.append(result)
            result=dict(folds=folds,**{k:mean(f[k] for f in folds) for k in ['source_val','iid','test_in','test_out']})
            write(folder/'result.json',result);trial.set_user_attr('result',result);study.tell(trial,result['source_val'])
            records=[dict(trial=t.number,**t.params,**{k:t.user_attrs['result'][k] for k in ['source_val','iid','test_out']}) for t in study.trials if t.state==TrialState.COMPLETE]
            p.write_csv(root/'trials_summary.csv',records,['trial',*SPACE,'source_val','iid','test_out'])
            try:sync()
            except Exception as e:print('RESULT_SYNC_RETRY',e,flush=True)
        try:sync()
        except Exception:pass
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
