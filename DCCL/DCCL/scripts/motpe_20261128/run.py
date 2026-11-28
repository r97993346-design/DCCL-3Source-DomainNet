"""Local multiobjective TPE, with serial folds and source-only selection."""
import csv
import fcntl
import importlib.util
import itertools
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from statistics import mean
import optuna
from optuna.trial import TrialState
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'scripts/cipt_clip_supcon_optuna'))
import protocol as p
spec = importlib.util.spec_from_file_location('fold_runner', CODE / 'scripts/cipt_clip_supcon_optuna/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
NAME = 'condcov_rd_crossdomain_constant1e5_motpe_20261128'
DATASET = sys.argv[1]
assert DATASET in ('VLCS', 'TerraIncognita'), json.dumps(dict(condition="DATASET in ('VLCS', 'TerraIncognita')", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
ROOT = Path('/root/autodl-tmp/outputs') / NAME / DATASET
ROOT.mkdir(parents=True, exist_ok=True)
SPACE = {'s_weight': (0.5, 0.75, 1.0), 'condcov_weight': (1.0, 2.0, 3.0), 'rd_weight': (1.0, 2.0, 3.0), 'contrastive_weight': (0.01, 0.025, 0.05)}
COMBINATIONS = list(itertools.product(*SPACE.values()))
FIRST = dict(zip(SPACE, (1.0, 3.0, 3.0, 0.05) if DATASET == 'VLCS' else (1.0, 1.0, 1.0, 0.01)))
p.DATASETS['TerraIncognita'] = (('L100', 'L38', 'L43', 'L46'), 0.58, 'terra_incognita_fixed')
p.FIXED_ARGS.update(num_workers=8, eval_num_workers=0)
p.PROTOCOL_ARGS.update(num_workers=8, eval_num_workers=0, cipt_cache_eval_visual=True)
p.OUTPUT = runner.OUTPUT = ROOT
runner.PAUSE = ROOT / 'pause.request'
runner.GPU_FOR_DATASET = {DATASET: 0}
runner.ACTIVE_GPU = {0: 0}
p.LR_POLICY = {'name': 'constant', 'lr': 1e-05, 'updates': 5001, 'scheduler_updates': 0}

def hp(params):
    return dict(p.fixed_hparams(), lr=1e-05, cipt_lr_schedule='constant', cipt_contrastive_type='clip_guided_cross_domain', cipt_prompt_length=8, cipt_clip_teacher_mix=0.25, cipt_s_uniform_weight=params['s_weight'], cipt_condcov_weight=params['condcov_weight'], cipt_env_loss_weight=params['rd_weight'], cipt_causal_contrastive_weight=params['contrastive_weight'], cipt_contrastive_weight=params['contrastive_weight'])
p.hparams_for = runner.hparams_for = hp
original_command = p.command

def command(*args, **kw):
    cmd = original_command(*args, **kw)
    cmd[3] = 'scripts/motpe_20261128/instrument.py'
    return [*cmd, '--cipt_cache_eval_visual']
runner.command = command
original_validate = p.validate_fold

def validate(rows, dataset, fold, params, *, updates=p.STEPS):
    assert [r['step'] for r in rows] == p.EVAL_STEPS, json.dumps(dict(condition="[r['step'] for r in rows] == p.EVAL_STEPS", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    expected = hp(params)
    weights = dict(zip(p.TERMS, (1.0, params['s_weight'], params['condcov_weight'], params['rd_weight'], params['contrastive_weight'])))
    for row in rows:
        (args, h) = (row['args'], row['hparams'])
        assert all((args.get(k) == v for (k, v) in p.PROTOCOL_ARGS.items())), json.dumps(dict(condition='all((args.get(k) == v for (k, v) in p.PROTOCOL_ARGS.items()))', step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert args['dataset'] == dataset and args['real_test_envs'] == [fold] and (args['seed'] == p.seed_for(dataset, fold)), json.dumps(dict(condition="args['dataset'] == dataset and args['real_test_envs'] == [fold] and (args['seed'] == p.seed_for(dataset, fold))", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert all((h.get(k) == v for (k, v) in expected.items())), json.dumps(dict(condition='all((h.get(k) == v for (k, v) in expected.items()))', step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert row['optimizer_updates'] == row['step'] + 1 and row['scheduler_updates'] == 0 and math.isclose(row['actual_lr'], 1e-05, rel_tol=1e-12, abs_tol=1e-15), json.dumps(dict(condition="row['optimizer_updates'] == row['step'] + 1 and row['scheduler_updates'] == 0 and math.isclose(row['actual_lr'], 1e-05, rel_tol=1e-12, abs_tol=1e-15)", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert row['ind_mode'] == 'condcov' and row['cipt_ind_loss'] == row['nbr_check'] == 0, json.dumps(dict(condition="row['ind_mode'] == 'condcov' and row['cipt_ind_loss'] == row['nbr_check'] == 0", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert all((row[k] == 1 for k in ('ablation_de_enabled', 'env_anchor_active', 'e_only_infer'))), json.dumps(dict(condition="all((row[k] == 1 for k in ('ablation_de_enabled', 'env_anchor_active', 'e_only_infer')))", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert row.get('cg_active', 0) > 0, json.dumps(dict(condition="row.get('cg_active', 0) > 0", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert row['eval_visual_cache_bytes'] > 0, json.dumps(dict(condition="row['eval_visual_cache_bytes'] > 0", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        if row['step'] > 0:
            assert row['eval_visual_cache_hit_splits'] == 8, json.dumps(dict(condition="row['eval_visual_cache_hit_splits'] == 8", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        for (term, w) in weights.items():
            assert math.isclose(row['weight_' + term], w, rel_tol=1e-12, abs_tol=1e-15), json.dumps(dict(condition="math.isclose(row['weight_' + term], w, rel_tol=1e-12, abs_tol=1e-15)", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
            assert math.isclose(row['weighted_' + term], w * row['loss_' + term], rel_tol=2e-06, abs_tol=2e-08), json.dumps(dict(condition="math.isclose(row['weighted_' + term], w * row['loss_' + term], rel_tol=2e-06, abs_tol=2e-08)", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert math.isclose(sum((row['weighted_' + k] for k in weights)), row['loss_total'], rel_tol=2e-06, abs_tol=2e-08), json.dumps(dict(condition="math.isclose(sum((row['weighted_' + k] for k in weights)), row['loss_total'], rel_tol=2e-06, abs_tol=2e-08)", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        assert all((math.isfinite(row[k]) for k in ('test_in', 'test_out', 'train_out', 'loss_total'))), json.dumps(dict(condition="all((math.isfinite(row[k]) for k in ('test_in', 'test_out', 'train_out', 'loss_total')))", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    selected = max(rows, key=lambda r: r['train_out'])
    return dict(fold=fold, domain=p.DATASETS[dataset][0][fold], seed=p.seed_for(dataset, fold), selected_step=selected['step'], source_val=selected['train_out'], iid=selected['test_in'], test_in=selected['test_in'], test_out=selected['test_out'], updates=5001, losses={k: v for (k, v) in selected.items() if k.startswith(('loss_', 'weighted_', 'weight_'))}, code_commit=p.git('rev-parse', 'HEAD'))

class ProtocolValidationError(RuntimeError):
    pass

def read_attempt(attempt, dataset, fold, params, *, updates=p.STEPS):
    assert (attempt / 'exit_code').read_text().strip() == '0', json.dumps(dict(condition="(attempt / 'exit_code').read_text().strip() == '0'", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    logs = list((attempt / 'raw').glob(f'{dataset}/*/results.jsonl'))
    assert len(logs) == 1 and (not list(attempt.rglob('*.pth'))), json.dumps(dict(condition="len(logs) == 1 and (not list(attempt.rglob('*.pth')))", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    result = validate(p.read_rows(logs[0]), dataset, fold, params)
    trace = p.read_rows(attempt / 'learning_rates.jsonl')
    assert len(trace) == 5001 and all((r['step'] == i and math.isclose(r['lr_used'], 1e-05, rel_tol=1e-12, abs_tol=1e-15) and (r['scheduler_updates'] == 0) for (i, r) in enumerate(trace))), json.dumps(dict(condition="len(trace) == 5001 and all((r['step'] == i and math.isclose(r['lr_used'], 1e-05, rel_tol=1e-12, abs_tol=1e-15) and (r['scheduler_updates'] == 0) for (i, r) in enumerate(trace)))", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    assert 'ADAM_UPDATES 5001 CONSTANT_LR 1e-5 SCHEDULER_UPDATES 0' in (attempt / 'console.log').read_text(), json.dumps(dict(condition="'ADAM_UPDATES 5001 CONSTANT_LR 1e-5 SCHEDULER_UPDATES 0' in (attempt / 'console.log').read_text()", step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
    return dict(result, log=str(logs[0]), learning_rates=str(attempt / 'learning_rates.jsonl'), elapsed_seconds=float((attempt / 'elapsed_seconds').read_text()))
verified_read_attempt = read_attempt

def read_attempt(*args, **kwargs):
    attempt = Path(args[0])
    if not (attempt / 'exit_code').exists() or (attempt / 'exit_code').read_text().strip() != '0':
        raise ValueError('attempt has no successful exit code: ' + str(attempt))
    try:
        return verified_read_attempt(*args, **kwargs)
    except AssertionError as exc:
        raise ProtocolValidationError(str(exc)) from exc
runner.read_attempt = read_attempt
original_run_fold = runner.run_fold

def recover_fold(root, dataset, fold, number, params, gpu, env, lock_fd, **kwargs):
    handoff=ROOT/'repair_handoff.json'
    if handoff.exists():
        info=p.read_json(handoff)
        exit_path=Path(info['attempt'])/'exit_code'
        while not exit_path.exists():
            p.write_json(ROOT/'status.json',dict(state='running_preserved_worker',trial=number,fold=fold,worker_pid=info['worker_pid'],controller_pid=os.getpid(),params=params))
            time.sleep(15)
        try:
            if Path('/proc',str(info['old_controller_pid'])).exists():
                os.kill(info['old_controller_pid'],9)
        except ProcessLookupError: pass
        handoff.rename(ROOT/'repair_handoff.completed.json')

    matches = []
    for config in sorted((ROOT / dataset / 'trials').glob('trial_*/config.json')):
        if p.read_json(config) != hp(params):
            continue
        for attempt in sorted(config.parent.glob(f'env{fold}/attempt_*')):
            if not (attempt / 'exit_code').exists() or (attempt / 'exit_code').read_text().strip() != '0':
                continue
            try:
                result = read_attempt(attempt, dataset, fold, params)
                matches.append((attempt, result))
            except (ProtocolValidationError, KeyError, OSError) as exc:
                p.write_json(attempt / 'revalidation_error.json', dict(error=str(exc)))
    if matches:
        (attempt, result) = matches[0]
        result = dict(result, recovered_from=str(attempt))
        with (ROOT / 'recovery_records.jsonl').open('a') as f:
            f.write(json.dumps(dict(local_trial=number, fold=fold, params=params, source=str(attempt), validated=True, duplicates_not_reused=[str(a) for (a, r) in matches[1:]])) + '\n')
        return result
    return original_run_fold(root, dataset, fold, number, params, gpu, env, lock_fd, **kwargs)
runner.run_fold = recover_fold

def key(params):
    return tuple((params[k] for k in SPACE))

def log(text):
    with (ROOT / 'decision_log.md').open('a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')

def summaries(study, state='running', **extra):
    trials = study.trials
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    rows = [dict(trial=t.number, state=t.state.name, **t.params, **{k: t.user_attrs.get('result', {}).get(k) for k in ('source_val', 'worst_fold_source_val', 'iid', 'test_in', 'test_out', 'elapsed_seconds')}, folds_json=json.dumps(t.user_attrs.get('result', {}).get('folds', []))) for t in trials]
    fields = ['trial', 'state', *SPACE, 'source_val', 'worst_fold_source_val', 'iid', 'test_in', 'test_out', 'elapsed_seconds', 'folds_json']
    p.write_csv(ROOT / 'trials_summary.csv', rows, fields)
    pareto = [rows[t.number] for t in study.best_trials]
    p.write_csv(ROOT / 'pareto_front.csv', pareto, fields)
    p.write_json(ROOT / 'pareto_front.json', pareto)
    if complete:
        for (label, i) in (('best_source_val', 0), ('best_worst_fold_source_val', 1)):
            t = max(complete, key=lambda t: t.values[i])
            p.write_json(ROOT / (label + '.json'), rows[t.number])
    finished = {key(t.params) for t in complete}
    p.write_json(ROOT / 'status.json', dict(state=state, dataset=DATASET, complete_trials=len(complete), searched_combinations=len(finished), remaining_combinations=81 - len(finished), directions=['maximize', 'maximize'], **extra))
    (ROOT / 'sync.request').touch()

def refresh(dataset, study, root, state, **kw):
    summaries(study, state, **kw)
runner.refresh = refresh

def resources():
    import subprocess
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
        shm = os.statvfs('/dev/shm')
        if runner.available_ram_kib() > 10 * 1024 ** 2 and free >= 4096 and (shm.f_bavail * shm.f_frsize > 4 * 1024 ** 3):
            return
        p.write_json(ROOT / 'status.json', dict(state='waiting_for_resources', free_vram_mib=free))
        time.sleep(15)

def main():
    protocol = dict(name=NAME, dataset=DATASET, directions=['maximize', 'maximize'], objectives=['mean_four_fold_source_val', 'minimum_four_fold_source_val'], sampler=dict(name='TPESampler', seed=42, n_startup_trials=5, n_ei_candidates=24, multivariate=False), optuna_version=optuna.__version__, pruner='NopPruner', fixed_args=p.FIXED_ARGS, base_hparams=hp(FIRST), seeds=[p.seed_for(DATASET, i) for i in range(4)], lr_policy=p.LR_POLICY, code_commit=p.git('rev-parse', 'HEAD'), evaluation_cache=True, training_rng_separate_from_evaluation=True, global_registry=False, stop_rule='all 81 legal combinations COMPLETE; target IID is reference only', worker_rng_note='Eight workers may change augmentation streams relative to previous worker counts.')
    for (name, value) in (('protocol', protocol), ('search_space', {k: list(v) for (k, v) in SPACE.items()})):
        path = ROOT / (name + '.json')
        if path.exists():
            existing = p.read_json(path)
            if name == 'protocol':
                value['code_commit'] = existing['code_commit']
            assert existing == value, json.dumps(dict(condition='existing == value', step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        else:
            p.write_json(path, value)
    with (ROOT / 'controller.repaired.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5, n_ei_candidates=24, multivariate=False)
        rng = ROOT / 'sampler.pkl'
        if rng.exists():
            sampler = pickle.loads(rng.read_bytes())
        study = optuna.create_study(study_name=NAME + '_' + DATASET, storage=f"sqlite:///{ROOT / 'optuna.db'}", load_if_exists=True, directions=['maximize', 'maximize'], sampler=sampler, pruner=optuna.pruners.NopPruner())
        assert study.directions == [optuna.study.StudyDirection.MAXIMIZE] * 2, json.dumps(dict(condition='study.directions == [optuna.study.StudyDirection.MAXIMIZE] * 2', step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
        if not study.trials:
            study.enqueue_trial(FIRST, user_attrs={'origin': 'requested_first_candidate'})
        log('Controller resumed with local deduplication and two source-validation objectives.')
        env = runner.environment_for('GPU-97473323-921d-3883-586d-1a0230891c12')
        while not runner.PAUSE.exists():
            complete = {key(t.params) for t in study.trials if t.state == TrialState.COMPLETE}
            if len(complete) == 81:
                summaries(study, 'space_exhausted')
                sensitivity = []
                for (name, choices) in SPACE.items():
                    for value in choices:
                        trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.params[name] == value]
                        sensitivity.append(dict(parameter=name, value=value, complete=len(trials), mean_source_val=mean((t.values[0] for t in trials)), mean_worst_fold_source_val=mean((t.values[1] for t in trials))))
                p.write_json(ROOT / 'parameter_sensitivity.json', sensitivity)
                p.write_text(ROOT / 'next_stage_analysis.md', 'All 81 combinations are complete. Review parameter_sensitivity.json and the selected-fold raw and weighted losses in trials_summary.csv. Compare source-validation responses to S, CondCov, RD and CL, and identify weak source-validation folds. Any wider range or architectural change requires approval. Target IID is a research reference only.\n')
                log('All 81 legal combinations complete. Inspect loss and parameter sensitivity before proposing a new approved space.')
                return
            active = study.get_trials(states=(TrialState.RUNNING,))
            assert len(active) <= 1, json.dumps(dict(condition='len(active) <= 1', step=locals().get('row', {}).get('step'), term=locals().get('term'), expected_weight=locals().get('w'), actual_weights={k: v for (k, v) in locals().get('row', {}).items() if k.startswith('weight_')}, args=locals().get('row', {}).get('args'), expected_args=p.PROTOCOL_ARGS), default=str)
            trial = optuna.Trial(study, active[0]._trial_id) if active else study.ask()
            params = {k: trial.suggest_categorical(k, v) for (k, v) in SPACE.items()}
            if key(params) in complete:
                trial.set_user_attr('origin', 'duplicate_suggestion')
                study.tell(trial, state=TrialState.FAIL)
                log('Duplicate suggestion skipped without training.')
                used = {key(t.params) for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.RUNNING) and set(t.params) == set(SPACE)}
                remaining = [c for c in COMBINATIONS if c not in used]
                chosen = random.Random(42 + len(study.trials)).choice(remaining)
                study.enqueue_trial(dict(zip(SPACE, chosen)), user_attrs={'origin': 'unsearched_fallback'})
                continue
            saved = rng.with_suffix('.tmp')
            saved.write_bytes(pickle.dumps(sampler))
            saved.replace(rng)
            trial.set_user_attr('effective_hparams', hp(params))
            resources()
            try:
                result = runner.finish_trial(ROOT / DATASET, DATASET, study, trial.number, params, 0, env, lock.fileno())
            except (ProtocolValidationError, KeyError) as exc:
                p.write_json(ROOT / 'validation_error.json', dict(trial=trial.number, error=str(exc)))
                runner.PAUSE.touch()
                summaries(study, 'paused_protocol_error', error=str(exc))
                log('Paused without failing or retrying Trial: ' + str(exc))
                return
            if 'error' in result:
                trial.set_user_attr('result', result)
                study.tell(trial, state=TrialState.FAIL)
                log(f'Trial {trial.number} failed; combination remains eligible.')
                study.enqueue_trial(params, user_attrs={'origin': 'failed_trial_retry', 'retry_of': trial.number})
                time.sleep(60)
            else:
                result['worst_fold_source_val'] = min((f['source_val'] for f in result['folds']))
                trial.set_user_attr('result', result)
                study.tell(trial, [result['source_val'], result['worst_fold_source_val']])
                log(f'Trial {trial.number} complete, source objectives {study.trials[trial.number].values}.')
            summaries(study)
        summaries(study, 'paused')
if __name__ == '__main__':
    main()
