"""Read-only resource and legacy trial-boundary observations."""
import json, subprocess, time
from pathlib import Path
root=Path("/root/autodl-tmp/outputs/condcov_rd_crossdomain_constant1e5_motpe_20261128")
while True:
    row={"time":time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in ("cpu.max","cpu.pressure","memory.current","memory.max","memory.events"):
        row[name]=(Path("/sys/fs/cgroup")/name).read_text()
    row["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader"],text=True).strip()
    row["shm"]=subprocess.check_output(["df","-B1","/dev/shm"],text=True).strip()
    row["legacy"]={}
    for path in root.parent.glob("*_fixed6*"):
        if (path/"status.json").exists():
            row["legacy"][path.name]={"status":json.loads((path/"status.json").read_text()),"pause":(path/"pause.request").exists()}
    with (root/"resource_observations.jsonl").open("a") as f:
        f.write(json.dumps(row)+"\n")
    time.sleep(60)
