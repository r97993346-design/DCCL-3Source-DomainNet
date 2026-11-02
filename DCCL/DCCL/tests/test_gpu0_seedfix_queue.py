"""Exercise PID-only waiting and serial jobs; GPU queries must never be called.

No training data, CLIP, PyTorch or GPU is used. Run from DCCL/DCCL:
python -m unittest discover -s tests -p 'test_gpu0_seedfix_queue.py' -v
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_gpu0_seedfix_6exp_after_pid.sh'
BRANCH = 'feature/cipt-per-target-seed-20261102'


class GPU0SeedFixQueueTests(unittest.TestCase):
    script = SCRIPT
    dataset = 'PACS'
    gpu = '0'
    wait_pid = 3112421

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.bin = self.folder / 'bin'
        self.bin.mkdir()
        self.events = self.folder / 'events.jsonl'
        self.output = self.folder / 'output'
        data = self.folder / 'data'
        (data / self.dataset).mkdir(parents=True)
        clip = self.folder / 'clip.pt'
        clip.touch()
        self.write_executable('git', '#!/bin/sh\nprintf "%s\\n" ' + BRANCH + '\n')
        self.write_executable('nvidia-smi', f'''#!{sys.executable}
import os,sys
from pathlib import Path
if '--id={self.gpu}' not in sys.argv:
    sys.exit(90)
if os.environ.get('FAKE_QUERY_FAILURE'):
    print('mock driver error',file=sys.stderr)
    sys.exit(7)
p=Path(os.environ['FAKE_GPU_COUNT'])
count=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(count))
if count<=int(os.environ.get('FAKE_GPU_BUSY_CHECKS','0')):
    print('987654')
''')
        self.python = self.write_executable('training-python', f'''#!{sys.executable}
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
if args and args[0]=='-':
    # This executor virtualizes Popen PIDs separately from /proc. Execute the
    # real identity reader against a controlled Linux-style proc tree.
    source=sys.stdin.read().replace("Path('/proc')", "Path(os.environ['FAKE_PROC_ROOT'])")
    sys.argv=args
    exec(compile(source,'process_identity','exec'),{{'__name__':'__main__','os':os}})
    sys.exit(0)
if len(args)>1 and args[0]=='-u' and args[1]=='train_all.py':
    name=args[2]
    path=Path(os.environ['FAKE_TRAIN_EVENTS'])
    def event(kind):
        with path.open('a') as f:
            f.write(json.dumps({{'event':kind,'name':name,'args':args,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}})+'\\n')
    event('START')
    time.sleep(.02)
    event('END')
    sys.exit(17 if os.environ.get('FAKE_FAIL_NAME')==name else 0)
os.execv(sys.executable,[sys.executable]+args)
''')
        self.env = dict(os.environ,
            PATH=str(self.bin)+os.pathsep+os.environ['PATH'],
            CIPT_PROJECT_DIR=str(ROOT), CIPT_PYTHON_BIN=str(self.python),
            CIPT_DATA_DIR=str(data), CIPT_CLIP_PATH=str(clip),
            CIPT_OUTPUT_ROOT=str(self.output), CIPT_POLL_SECONDS='0.02',
            FAKE_TRAIN_EVENTS=str(self.events),
            FAKE_GPU_COUNT=str(self.folder/'gpu_checks'),
            FAKE_PROC_ROOT=str(self.folder/'proc'))

    def write_executable(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o755)
        return path

    def run_queue(self, *args, **env):
        return subprocess.run(['bash', str(self.script), '--wait-pid', '0', *args],
            env=dict(self.env, **env), text=True, capture_output=True, timeout=20)

    def read_events(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []

    def test_dry_run_has_exact_six_approved_parameter_sets(self):
        result = subprocess.run(['bash', str(self.script), '--dry-run'],
            env=self.env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'wait only for PID {self.wait_pid} to exit',result.stdout)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.startswith(f'CUDA_VISIBLE_DEVICES={self.gpu} ')]
        self.assertEqual(len(commands), 6)
        expected = [(5000,'false','0','none','0'), (5000,'true','0.1','none','0'),
                    (5000,'true','0.2','none','0'), (5000,'true','0.5','none','0'),
                    (5000,'false','0','center_mse','1'), (3800,'false','0','center_mse','1')]
        for command, values in zip(commands, expected):
            def value(flag): return command[command.index(flag)+1]
            self.assertEqual(value('--dataset'),self.dataset)
            self.assertEqual(command[command.index('--test_envs')+1:command.index('--test_envs')+5],
                             ['0','1','2','3'])
            self.assertEqual(tuple(value(f) for f in ('--steps','--cipt_use_ind','--cipt_gamma',
                '--cipt_env_constraint','--cipt_env_loss_weight')), tuple(map(str,values)))
            for flag, setting in (('--lr','5e-5'),('--cipt_lr_schedule','constant'),
                ('--batch_size','32'),('--num_workers','2'),('--eval_num_workers','2'),
                ('--cipt_e_cls_weight','1'),('--cipt_s_uniform_weight','1'),
                ('--cipt_use_contrastive','false'),('--cipt_causal_contrastive_weight','0'),
                ('--cipt_env_selection','all'),('--cipt_env_topk','80'),('--swad','LossValley')):
                self.assertEqual(value(flag),setting)
        self.assertFalse(self.events.exists())
        self.assertFalse(self.output.exists())

    def test_busy_gpu_is_ignored_and_training_is_serial(self):
        result = self.run_queue(FAKE_GPU_BUSY_CHECKS='99999')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertFalse((self.folder/'gpu_checks').exists())
        self.assertNotIn(f'GPU{self.gpu} idle',result.stdout)
        events = self.read_events()
        self.assertEqual(len(events),12)
        self.assertEqual([e['event'] for e in events],['START','END']*6)
        for start,end in zip(events[::2],events[1::2]):
            self.assertEqual(start['name'],end['name'])
            self.assertEqual(start['gpu'],self.gpu)
        self.assertEqual(len(list(self.output.glob('run_*/*/command.sh'))),6)
        statuses = next(self.output.glob('run_*/status.tsv')).read_text()
        self.assertEqual(statuses.count('\tDONE\t0'),6)
        self.assertIn('\tQUEUE\tEXIT\t0',statuses)

    def test_failure_stops_remaining_experiments(self):
        prefix=self.dataset.lower()
        result = self.run_queue(FAKE_FAIL_NAME=f'02_{prefix}_es1_ind01_n5000')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual([e['name'] for e in self.read_events() if e['event']=='START'],
                         [f'01_{prefix}_es1_ind0_n5000',f'02_{prefix}_es1_ind01_n5000'])
        statuses = next(self.output.glob('run_*/status.tsv')).read_text()
        self.assertIn('\tFAILED\t17',statuses)

    def test_gpu_query_availability_does_not_gate_training(self):
        result = self.run_queue(FAKE_QUERY_FAILURE='1')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(len(self.read_events()),12)
        self.assertFalse((self.folder/'gpu_checks').exists())

    def test_live_pid_is_the_only_wait_condition(self):
        target_pid = 515151
        proc = self.folder/'proc'/str(target_pid)
        proc.mkdir(parents=True)
        fields = ['S']+['0']*18+['424242']+['0']*15
        (proc/'stat').write_text(str(target_pid)+' (python) '+' '.join(fields)+'\n')
        (proc/'cmdline').write_bytes(b'python\0-u\0train_all.py\0current_experiment\0')
        process = None
        log = self.folder/'queue.log'
        try:
            with log.open('w') as stream:
                process = subprocess.Popen(['bash',str(self.script),'--wait-pid',str(target_pid)],
                    env=dict(self.env,FAKE_GPU_BUSY_CHECKS='99999'),
                    stdout=stream,stderr=subprocess.STDOUT,text=True)
                deadline = time.monotonic()+5
                while 'Waiting for original training PID' not in log.read_text():
                    self.assertIsNone(process.poll(),log.read_text())
                    self.assertLess(time.monotonic(),deadline,log.read_text())
                    time.sleep(.02)
                time.sleep(.1)
                self.assertEqual(self.read_events(),[])
                (proc/'stat').unlink()
                process.wait(timeout=15)
            self.assertEqual(process.returncode,0,log.read_text())
            self.assertEqual(len(self.read_events()),12)
            self.assertFalse((self.folder/'gpu_checks').exists())
        finally:
            if process is not None and process.poll() is None:
                process.terminate(); process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
