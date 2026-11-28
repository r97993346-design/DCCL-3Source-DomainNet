"""Two fixed independent ablations; no Optuna study is opened or changed."""
import base64
import csv
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid

BASE = Path(__file__).resolve().parent
CODE = BASE/'code/DCCL/DCCL'
REF = BASE/'historical_reference'
NAME = 'positive_set_ablation_constant_lr1e5_20261128'
BRANCH = 'results/positive-set-ablation-constant-lr1e5-20261128'
PYTHON = '/root/miniconda3/envs/cipt/bin/python'
sys.path[:0] = ['/root/autodl-tmp/global-optuna-registry','/root/autodl-tmp/cipt-global-rollout',str(CODE/'scripts/cipt_clip_supcon_optuna')]
from global_registry import configuration, digest, model_fingerprint, Unavailable, now
from github_api_registry import APIRegistry
import protocol as p

def write(path, obj):
    p.write_json(path,obj)

def rows(path):
    if not path.exists(): return []
    lines = path.read_text().splitlines()
    output=[]
    for i,line in enumerate(lines):
        try: output.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
    return output

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def retry(fn):
    while True:
        try: return fn()
        except (Unavailable, subprocess.SubprocessError, OSError) as exc:
            print('Network operation waiting:',str(exc),flush=True)
            time.sleep(30)

class SharedRegistry(APIRegistry):
    def bytes(self,branch,path):
        doc=self.api(f'contents/{path}?ref={branch}')
        if not doc.get('content'):
            doc=self.api(f'git/blobs/{doc["sha"]}')
        return base64.b64decode(doc['content'])

    def published(self,branch,path,expected_hash):
        assert branch==BRANCH
        doc=json.loads(self.bytes(branch,path))
        assert doc['config_hash']==expected_hash
        assert [f['fold'] for f in doc['folds']]==list(range(4))
        for fold in doc['folds']:
            for label in ['results','learning_rates']:
                entry=fold[label]
                content=self.bytes(branch,entry['path'])
                assert hashlib.sha256(content).hexdigest()==entry['sha256']
                data=[json.loads(x) for x in content.splitlines() if x.strip()]
                if label=='results':
                    assert [x['step'] for x in data]==[*range(0,5000,300),4999]
                    selected=max(data,key=lambda x:x['train_out'])
                    assert all(fold[k]==selected[v] for k,v in [('source_val','train_out'),('iid','test_in'),('test_out','test_out'),('selected_step','step')])
                else:
                    assert len(data)==5000
                    assert all(x['step']==i and x['lr_used']==1e-5 and x['scheduler_updates']==0 for i,x in enumerate(data))
        return doc

registry=SharedRegistry()
ALLOWED={'protocol.json','config.json','config_diff.json','status.json','console.log','controller.log','results.jsonl',
         'learning_rates.jsonl','folds_summary.csv','trials_summary.csv','old_vs_new_comparison.csv',
         'folds_old_vs_new.csv','loss_comparison.csv','registry_manifest.json','result.json','effective_config.json',
         'source_diff.patch','positive_set_run.py','positive_set_instrument.py','audit_summary.json','launch_state.json'}

def publish():
    with (BASE/'.publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for attempt in range(10):
            head=registry.api(f'git/ref/heads/{BRANCH}',optional=True)
            if head is None:
                tree=registry.api('git/trees',method='POST',body={'tree':[{'path':'README.md','mode':'100644','type':'blob','content':'Fixed PACS and VLCS positive set ablations. Results only.\n'}]})['sha']
                commit=registry.api('git/commits',method='POST',body={'message':'Initialize positive set ablation results','tree':tree,'parents':[]})['sha']
                try: registry.api('git/refs',method='POST',body={'ref':'refs/heads/'+BRANCH,'sha':commit})
                except Unavailable: pass
                continue
            head=head['object']['sha']
            base_tree=registry.api(f'git/commits/{head}')['tree']['sha']
            remote=registry.api(f'git/trees/{base_tree}?recursive=1')['tree']
            existing={x['path']:x['sha'] for x in remote if x['type']=='blob'}
            entries=[]
            paths=[x for x in BASE.rglob('*') if x.is_file() and x.name in ALLOWED and not any(y in x.relative_to(BASE).parts for y in ('code','historical_reference','.git'))]
            for path in sorted(paths):
                content=path.read_bytes()
                if path.suffix=='.jsonl' and content and not content.endswith(b'\n'):
                    content=content[:content.rfind(b'\n')+1]
                dest=NAME+'/'+path.relative_to(BASE).as_posix()
                blobsha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                if existing.get(dest)==blobsha: continue
                blob=registry.api('git/blobs',method='POST',body={'content':base64.b64encode(content).decode(),'encoding':'base64'})['sha']
                entries.append({'path':dest,'mode':'100644','type':'blob','sha':blob})
            if not entries:return head
            tree=registry.api('git/trees',method='POST',body={'base_tree':base_tree,'tree':entries})['sha']
            commit=registry.api('git/commits',method='POST',body={'message':'Sync fixed positive set ablation results','tree':tree,'parents':[head]})['sha']
            try:
                answer=registry.api(f'git/refs/heads/{BRANCH}',method='PATCH',body={'sha':commit,'force':False})
                assert answer['object']['sha']==commit
                return commit
            except Unavailable:
                if registry.api(f'git/ref/heads/{BRANCH}')['object']['sha']==head: raise
        raise Unavailable('result branch contention')

def prepare():
    assert subprocess.check_output(['git','-C',str(BASE/'code'),'status','--porcelain'],text=True)==''
    implementation=model_fingerprint(CODE)
    source=read(Path('/root/autodl-tmp/cipt-global-rollout/global-cipt-identities.json'))
    changes=[]
    patch=[]
    for old in sorted((REF/'old_code').rglob('*.py')):
        new=BASE/'code'/old.relative_to(REF/'old_code')
        if new.exists() and old.read_bytes()!=new.read_bytes():
            changes.append(str(old.relative_to(REF/'old_code')))
            patch.extend(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='old/'+str(old.relative_to(REF/'old_code')),tofile='new/'+str(old.relative_to(REF/'old_code'))))
    p.write_text(BASE/'source_diff.patch',''.join(patch))
    audit={'changed_training_files':changes,'reviewed_differences':[
      'All same-class non-self positives include within-domain pairs; denominator becomes all non-self samples.',
      'Eval cache added; first real evaluation batch compares cached and direct logits and predictions.',
      'SupCon scalar component diagnostics added; no extra gradient diagnostics.',
      'Training RNG isolation, Prompt implementation, data transforms, sampling, CondCov and RD unchanged.',
      'Historical optimizer norm diagnostics omitted as requested; no optimizer settings change.',
      'AutoDL paths differ; execution host and paths excluded from hash.',
      'Independent telemetry wrapper observes every Adam call; it does not change the model or optimizer.'],
      'historical_branch':'results/condcov-optuna-joint-20261125','training_code_commit':p.git('rev-parse','HEAD',cwd=CODE),
      'implementation_sha256':implementation,'instrumentation_sha256':sha(BASE/'positive_set_instrument.py')}
    for dataset,n in [('PACS',7),('VLCS',0)]:
        root=BASE/dataset;root.mkdir(exist_ok=True)
        old=read(REF/dataset/'config.json');result=read(REF/dataset/'result.json')
        seeds=[]
        for f in result['folds']:
            log=rows(REF/dataset/f'env{f["fold"]}'/'results.jsonl')
            assert [x['step'] for x in log]==[*range(0,5000,300),4999]
            assert f['updates']==5000 and f['final_step']==4999
            assert f['seed']==p.seed_for(dataset,f['fold'])
            assert log[0]['args']['trial_seed']==1 and log[0]['args']['holdout_fraction']==.2
            selected=max(log,key=lambda x:x['train_out'])
            assert selected['step']==f['selected_step'] and selected['train_out']==f['source_val']
            seeds.append(f['seed'])
        hp=dict(old,cipt_contrastive_type='clip_guided_supcon',cipt_neighbor_diagnostics=False)
        args=dict(source[dataset]['args'],steps=5000,checkpoint_freq=300,batch_size=32,cipt_cache_eval_visual=True)
        merged=dict(rows(REF/dataset/'env0/results.jsonl')[0]['hparams'],**hp)
        cfg=configuration(dataset,source[dataset]['data'],source[dataset]['weights'],implementation,args,merged,seeds,
                          {'name':'constant','initial_lr':1e-5,'warmup_steps':0,'optimizer_updates':5000,'scheduler_updates':0,
                           'evaluation_steps':[*range(0,5000,300),4999]})
        cfg.update(positive_definition='all_same_class_nonself_including_same_and_cross_domain',
                   implementation_protocol='positive-set-constant-adam5000-v1')
        write(root/'config.json',hp);write(root/'effective_config.json',cfg)
        write(root/'protocol.json',dict(audit,dataset=dataset,old_trial=n,fold_seeds=seeds,
            actual_adam_updates=5000,eval_steps=[*range(0,5000,300),4999],initial_lr=1e-5,schedule='constant',
            num_workers=2,eval_num_workers=0,model_save=0,source_selection='earliest_max_source_val',
            ram_start_minimum_gib=10,free_vram_minimum_mib=8192,resource_lock='shared-local-admission'))
        write(root/'config_diff.json',dict(audit,old_config=old,new_config=hp,
            config_changes={k:{'old':old.get(k),'new':hp.get(k)} for k in set(old)|set(hp) if old.get(k)!=hp.get(k)},
            old_result=result,hash=digest(cfg)))
        write(root/'status.json',{'state':'prepared','config_hash':digest(cfg),'updated_at':now()})
    write(BASE/'audit_summary.json',audit)
    tables()

def tables():
    comparisons=[];foldcomp=[];loss=[];trials=[]
    for ds in ['PACS','VLCS']:
        old=read(REF/ds/'result.json')
        folds=[read(x) for x in sorted((BASE/ds).glob('fold_*.json'))]
        p.write_csv(BASE/ds/'folds_summary.csv',folds,['fold','domain','seed','selected_step','source_val','iid','test_in','test_out','final_step','updates','log','learning_rates'])
        new={k:statistics.mean(x[k] for x in folds) for k in ['source_val','iid','test_in','test_out']} if len(folds)==4 else {}
        if new:trials.append(dict(dataset=ds,**new,folds_complete=4))
        comparisons.append(dict(dataset=ds,old_iid=old['iid']*100,new_iid=new.get('iid',0)*100 if new else '',
          delta_iid=(new['iid']-old['iid'])*100 if new else '',old_test_out=old['test_out']*100,
          new_test_out=new.get('test_out',0)*100 if new else '',delta_test_out=(new['test_out']-old['test_out'])*100 if new else '',
          old_source_val=old['source_val']*100,new_source_val=new.get('source_val',0)*100 if new else '',completed_folds=len(folds)))
        for f in folds:
            previous=old['folds'][f['fold']]
            foldcomp.append(dict(dataset=ds,domain=f['domain'],old_iid=previous['iid'],new_iid=f['iid'],old_test_out=previous['test_out'],new_test_out=f['test_out'],old_source_val=previous['source_val'],new_source_val=f['source_val'],old_selected_step=previous['selected_step'],new_selected_step=f['selected_step']))
            a=rows(REF/ds/f'env{f["fold"]}/results.jsonl');b=rows(Path(f['log']))
            for x,y in zip(a,b):
                for term in ['loss_e_cls','loss_s_uniform','loss_condcov','loss_rd','loss_clip_contrast','loss_total']:
                    if term in x and term in y:loss.append(dict(dataset=ds,fold=f['fold'],step=y['step'],term=term,old=x[term],new=y[term]))
    for name,data,fields in [('old_vs_new_comparison.csv',comparisons,list(comparisons[0])),('trials_summary.csv',trials,['dataset','source_val','iid','test_in','test_out','folds_complete']),('folds_old_vs_new.csv',foldcomp,['dataset','domain','old_iid','new_iid','old_test_out','new_test_out','old_source_val','new_source_val','old_selected_step','new_selected_step']),('loss_comparison.csv',loss,['dataset','fold','step','term','old','new'])]:
        p.write_csv(BASE/name,data,fields)

def resource():
    host=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    ram=min(host,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())) if limit!='max' else host
    gpu=subprocess.check_output(['nvidia-smi','--id','0','--query-gpu=memory.free,uuid','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return ram,int(gpu[0]),gpu[1].strip()

def run(ds):
    root=BASE/ds
    with (root/'.run.lock').open('a') as local:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=read(root/'effective_config.json');token_path=root/'owner_token'
        if not token_path.exists():p.write_text(token_path,uuid.uuid4().hex)
        token=token_path.read_text().strip();key=registry.key(ds,digest(cfg))
        def claim(waiting=False):
            return registry.claim(cfg,server_id='autodl-39636',study=NAME+'_'+ds,trial_id=0,
                commit=read(root/'protocol.json')['training_code_commit'],owner_token=token,waiting=waiting)
        _,reservation=retry(lambda:claim(True))
        if reservation['owner_token']!=token:
            write(root/'status.json',dict(state='occupied_skip',record=reservation));return
        if reservation['status']=='COMPLETE':return
        finished=[read(x) for x in sorted(root.glob('fold_*.json'))]
        def state(s,**extra):write(root/'status.json',dict(state=s,dataset=ds,config_hash=digest(cfg),completed_folds=[f['fold'] for f in finished],updated_at=now(),**extra))
        stopped=threading.Event()
        def heartbeat():
            while not stopped.wait(120):
                try:
                    registry.update(key,lambda old:dict(old,heartbeat_at=now(),updated_at=now()) if old and old['owner_token']==token and old['status'] in ['WAITING','RUNNING'] else None)
                except Exception as exc: print('heartbeat delayed',exc,flush=True)
        threading.Thread(target=heartbeat,daemon=True).start()
        try:
            for fold in range(4):
                if any(f['fold']==fold for f in finished):continue
                attempt=root/f'env{fold}'
                attempt.mkdir(exist_ok=True)
                assert not (attempt/'learning_rates.jsonl').exists(),'existing partial fold needs manual review; refusing duplicate'
                while True:
                    with Path('/root/autodl-tmp/.positive-ablation-admission.lock').open('a') as admission:
                        fcntl.flock(admission,fcntl.LOCK_EX)
                        ram,free,gpu=resource()
                        # Include RAM headroom for one extra process after launch.
                        if ram>14*1024**3 and free>=8192:
                            won,row=retry(lambda:claim(False))
                            assert won and row['owner_token']==token and row['status']=='RUNNING'
                            args=[PYTHON,'-B','-u',str(BASE/'positive_set_instrument.py'),str(attempt/'learning_rates.jsonl'),f'{ds}_positive_set_fold{fold}',str(root/'config.json'),
                                  '--test_envs',str(fold),'--dataset',ds,'--data_dir','/root/autodl-tmp/datasets','--cipt_clip_path','/root/autodl-tmp/weights/ViT-B-16.pt',
                                  '--batch_size','32','--deterministic','--source_only_selection','--cipt_cache_eval_visual','--output_root',str(attempt/'raw'),
                                  '--algorithm','CIPTDCCL','--source_sampling','domain_balanced','--steps','5000','--checkpoint_freq','300','--trial_seed','1',
                                  '--hparams_seed','0','--holdout_fraction','0.2','--aug','0.0','--evalmode','all','--num_workers','2','--eval_num_workers','0','--model_save','0']
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                            with (attempt/'console.log').open('xb') as console:
                                child=subprocess.Popen(args,cwd=CODE,env=env,stdout=console,stderr=subprocess.STDOUT,pass_fds=(local.fileno(),))
                            state('RUNNING',fold=fold,pid=child.pid,ram_available_bytes=ram,gpu_free_mib=free)
                            # Keep admission locked briefly so the second launch sees the new CUDA allocation.
                            for i in range(20):
                                if child.poll() is not None:break
                                time.sleep(1)
                            break
                        state('WAITING_RESOURCES',ram_available_bytes=ram,gpu_free_mib=free)
                    time.sleep(20)
                last_sync=0
                while child.poll() is None:
                    if time.monotonic()-last_sync>300:
                        try:
                            with (BASE/'.tables.lock').open('a') as lock:
                                fcntl.flock(lock,fcntl.LOCK_EX);tables()
                            publish()
                        except Exception as exc:print('periodic publication delayed',exc,flush=True)
                        last_sync=time.monotonic()
                    time.sleep(10)
                assert child.returncode==0,f'training failed, exit {child.returncode}'
                logs=list((attempt/'raw'/ds).glob('*/results.jsonl'));assert len(logs)==1
                data=rows(logs[0]);trace=rows(attempt/'learning_rates.jsonl')
                assert [x['step'] for x in data]==[*range(0,5000,300),4999]
                assert len(trace)==5000 and all(x['step']==i and x['lr_used']==1e-5 and x['scheduler_updates']==0 for i,x in enumerate(trace))
                hp=read(root/'config.json')
                for row in data:
                    assert all(row['hparams'].get(k)==v for k,v in hp.items())
                    assert row['args']['seed']==cfg['fold_seeds'][fold] and row['args']['steps']==5000
                    assert row['cg_all_class_active']==1 and row['e_only_infer']==1
                    assert all(math.isfinite(row[k]) for k in ['loss_total','loss_e_cls','loss_s_uniform','loss_condcov','loss_rd','loss_contrastive'])
                assert not list(attempt.rglob('*.pth')) and not list(attempt.rglob('*.pt'))
                selected=max(data,key=lambda x:x['train_out'])
                f=dict(fold=fold,domain=p.DATASETS[ds][0][fold],seed=cfg['fold_seeds'][fold],selected_step=selected['step'],source_val=selected['train_out'],iid=selected['test_in'],test_in=selected['test_in'],test_out=selected['test_out'],final_step=4999,updates=5000,log=str(logs[0]),learning_rates=str(attempt/'learning_rates.jsonl'))
                write(root/f'fold_{fold}.json',f);finished.append(f)
                changed,record=retry(lambda:registry.progress(key,token,completed_folds=[x['fold'] for x in finished]))
                assert record['owner_token']==token and record['status']=='RUNNING'
                state('FOLD_COMPLETE',fold=fold)
                with (BASE/'.tables.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);tables()
                retry(publish)
            manifest=[]
            for f in finished:
                def artifact(path):return {'path':NAME+'/'+Path(path).relative_to(BASE).as_posix(),'sha256':sha(Path(path))}
                manifest.append(dict(f,results=artifact(f['log']),learning_rates=artifact(f['learning_rates'])))
            write(root/'registry_manifest.json',dict(config_hash=digest(cfg),dataset=ds,folds=manifest))
            write(root/'result.json',dict(folds=finished,**{k:statistics.mean(f[k] for f in finished) for k in ['source_val','iid','test_in','test_out']}))
            state('PUBLISHING_COMPLETE');retry(publish)
            changed,record=retry(lambda:registry.complete(key,token,BRANCH,NAME+'/'+ds+'/registry_manifest.json',lambda m:{k:statistics.mean(f[k] for f in m['folds']) for k in ['source_val','iid','test_out']}))
            assert record['status']=='COMPLETE' and record['owner_token']==token
            state('COMPLETE');retry(publish)
        except Exception as exc:
            state('FAILED',error=str(exc))
            try:registry.progress(key,token,status='FAILED',error=str(exc))
            except Exception:pass
            raise
        finally:stopped.set()

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='publish':print(publish())
    else:run(sys.argv[1])
