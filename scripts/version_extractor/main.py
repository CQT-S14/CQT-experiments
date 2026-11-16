import pathlib
import os
import json
import argparse
import time
import importlib.metadata

# from dynaconf import Dynaconf

# os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()
import numpy as np
import matplotlib.pyplot as plt
from qibo import Circuit, gates, set_backend
from qibo.transpiler import NativeGates, Passes, Unroller
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y

# Add scripts/ to sys.path so we import scripts/config.py
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # scripts/config.py


from pathlib import Path
from git.repo.base import Repo


def get_package_version(package_name):
    """Get the version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(device):
    # List of packages to check versions for
    packages_to_check = [
        "qibo",
        "numpy",
        "qibolab",
        "qibocal",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pandas",
        "networkx",
        "sympy",
        "torch",
        "tensorflow",
        "pennylane",
    ]

    # Extract versions
    versions = {}
    for package in packages_to_check:
        version = get_package_version(package)
        if version is not None:
            versions[package] = version

    platform = "sinq20" if device == "sinq20" else device
    repo = Repo(config.CURRENT_CALIBRATION_DIRECTORY)
    hash_id = repo.commit().hexsha

    # Load run_id from file if it exists
    run_id = None
    try:
        id_file_path = config.RUN_ID_FILE
        if id_file_path.exists():
            with open(id_file_path, "r") as f:
                id_data = json.load(f)
                run_id = id_data.get("run_id")
    except Exception as e:
        print(f"Warning: Could not load run_id: {e}")

    results = {
        "run_id": run_id,
        "commit_message": repo.commit().message,
        "commit_hash": hash_id,
        "commit_date": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(repo.commit().committed_date)
        ),
        "versions": versions,
        "device": device,
        "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "platform": platform,
        "experiment_note": "temporary note!!!",
    }

    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use",
    )
    args = vars(parser.parse_args())
    main(**args)
