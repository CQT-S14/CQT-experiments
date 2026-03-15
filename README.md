# CQT-experiments

This project provides a benchmarking suite for quantum experiments. The system uses a batch runner (`scripts_executor`) that executes experiments defined in an ini configuration file (experiments_list.ini), organizing results in a standardized directory structure.

## Project Structure

```
CQT-experiments/
├── workspace/
│   ├── setup_dev_env.sh       # Creates the Python virtual environment
│   ├── compatibility_matrix.toml  # Pinned package version combos
│   └── resolve_constraints.py # Reads the matrix and emits pip pins
├── pipeline/
│   ├── run_sinq20_dev.sh      # Slurm job script (dev environment)
│   ├── run_sinq20.sh          # Slurm job script (legacy)
│   ├── run_numpy.sh           # Slurm job script (numpy simulation)
│   ├── experiment_list.ini    # Defines which experiments to run
│   ├── upload_experiment.py   # Uploads results to the remote database
│   └── set_best_run.py        # Marks a run as the current best
├── scripts/
│   ├── <experiment>/
│   │   └── main.py            # Experiment entry point
│   ├── config.py              # Path resolution helpers
│   └── scripts_executor.py   # Batch runner
├── clientdb/
│   └── client.py              # HTTP client for the remote DB API
├── data/                      # Experiment output (gitignored)
└── pyproject.toml
```

## Setup: Creating the Workspace Environment

The workspace environment is a Python virtual environment with pinned, compatible versions of the qibo ecosystem packages. It is defined entirely in `workspace/setup_dev_env.sh`.

### 1. Configure your local package clones

Open `workspace/setup_dev_env.sh` and edit the two variables at the top:

```bash
ENV_DIR=~/envs/workspace_env   # path where the venv will be created

REPOS=(
  "CQT-experiments-pipeline-benchmarking"  # your CQT-experiments clone (always first)
  "qibocal"                                 # any other local package clones to install as editable
)
```

- `ENV_DIR` — where the virtual environment will be created on disk.
- `REPOS` — folder names under `~/` to install as editable packages (`pip install -e`). The **first entry** must always be the CQT-experiments clone you want to run from. Add any other locally cloned packages (e.g. `qibocal`, `qibolab`) that you want to develop against.

Package versions for everything not listed in `REPOS` are pinned automatically from `workspace/compatibility_matrix.toml`.

### 2. Create the environment

From inside the repo:

```bash
bash workspace/setup_dev_env.sh
```

This will create the venv, install all packages with correct version pins, and configure a fallback to the system `qibo` module for hardware-specific packages (e.g. Keysight drivers).

### 3. Credentials

Scripts that interact with the remote database read credentials from `~/.env_user` (outside any repo). Copy the provided template and fill in your values:

```bash
cp .env_user.example ~/.env_user
chmod 600 ~/.env_user
# edit ~/.env_user with real values
```

## Running Experiments

To submit a batch of experiments to the sinq20 device via Slurm:

```bash
sbatch pipeline/run_sinq20_dev.sh
```

To cancel a submitted job:

```bash
scancel <job_id>
```

The job ID is printed by `sbatch` on submission. Logs are written to `logs/slurm_sinq20_dev_<job_id>.out / .err`.

`run_sinq20_dev.sh` sources `workspace/setup_dev_env.sh` at runtime to pick up `ENV_DIR` and `REPOS` — there is no duplication between the two files.

## How It Works

`scripts_executor.py` reads `pipeline/experiment_list.ini` to determine which experiments to run. Each experiment is organised in sections by qubit count:

- **[calibration]** — core calibration experiments (do not comment out)
- **[i]** — experiments using i qubits

The executor runs each enabled experiment's `main.py`, collecting results under `data/`.

## Rules for Adding New Experiments

- **Package Dependencies**: Add new packages to `pyproject.toml` (dependencies), e.g. torch, quiboml, quibocal
- The **Directory** of your experiment must be `scripts/<experiment>/`
- The **Entry Point** of your script must be `scripts/<experiment>/main.py`
- The **output** of your script must be  obtained as follows:

```python
from pathlib import Path
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); import config

# Resolve paths
out_dir = config.output_dir_for(__file__) / args.device
out_dir.mkdir(parents=True, exist_ok=True)
```


- **Device Support**: `device` is provided via `--device` and must be either `numpy` or `nqch`
- **Directory Creation**: Use `mkdir(parents=True, exist_ok=True)` before writing results
- Other arguments are `--qubits_list`, which is a list of edges, where the edge is a list of length 2: `"[[0, 1], [0, 3], [1, 4], [2, 3]"`

### Additional Guidelines

- **Arguments**: Provide `argparse` with sensible defaults; do not override CLI args in code
- **Integration**: Ensure the script runs when added to the ini configuration file
- **Artifacts**: Optional extra artifacts (plots, params, matrices) go in `data/<experiment>/<device>/...`
- **Histograms**: For histogram-like results, include all bitstrings (zero for missing) in frequencies dict
- **Documentation**: Document that `"numpy"` is supported for local simulation

### Configuration File

Add your experiment to the appropriate section in `config.ini`. The `[calibration]` section contains core experiments that should remain uncommented, while other sections can be selectively enabled/disabled by commenting out experiment entries.