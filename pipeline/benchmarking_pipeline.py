#!/usr/bin/env python3
"""
Orchestration entry point for benchmarking runs.
Submits the Slurm job and logs the job ID.
Uploading results data to the DB is handled inside the job itself by scripts/scripts_executor.py.
"""
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_SCRIPT = REPO_ROOT / "pipeline" / "run_sinq20_dev.sh"

result = subprocess.run(["sbatch", str(SLURM_SCRIPT)], capture_output=True, text=True)
print(result.stdout.strip())
if result.returncode != 0:
    print(result.stderr.strip(), file=sys.stderr)
    sys.exit(result.returncode)