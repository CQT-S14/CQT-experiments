import os
import sys
import subprocess
import argparse
import logging
from logging.handlers import RotatingFileHandler
import shutil
from git.repo.base import Repo
from datetime import datetime
import sys
from pathlib import Path
import json
import configparser
from client import (
    set_server,
    calibrations_upload, calibrations_list, calibrations_download, calibrations_get_latest,
    results_upload, results_download,unzip_bytes_to_folder,test,results_list,set_best_run, get_best_run,
    get_best_n_runs,upload_all_calibrations,upload_all_experiment_runs
)

# Base path to the scripts directory (run from project root)
base_path = "scripts/"


# Prepare JSON content with custom format
time_format = "%d-%m-%Y %H:%M"



if __package__ is None or __package__ == "":
    # invoked directly: add repo root to sys.path so 'scripts.*' resolves
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
else:
    # If imported as a package, determine repo root differently
    repo_root = Path(__file__).resolve().parents[1]

from scripts.config import CURRENT_CALIBRATION_DIRECTORY  # now this works in both cases
from scripts.config import load_experiment_list
from scripts.config import RUN_ID_FILE


def load_secrets():
    """Load credentials from ~/.env_user (KEY=VALUE format)."""
    env_file = Path.home() / ".env_user"
    secrets = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                secrets[key.strip()] = value.strip()
    return secrets


def get_best_qubits(hash_id):
    """
    Extract available qubits from calibration results.

    Args:
        hash_id (str): Git commit hash

    Returns:
        list[int]: List of qubit indices sorted by descending RB fidelity.
    """
    calib_path = repo_root / "data" / "calibrations" / hash_id / "sinq20" / "calibration.json"
    if not os.path.exists(calib_path):
        logging.warning(f"Calibration file not found: {calib_path}")
        return []

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Could not read calibration file {calib_path}: {e}")
        return []

    qubits_dict = data["single_qubits"]

    qubit_fidelities = []
    for k, v in qubits_dict.items():
        try:
            qidx = int(k)
        except Exception:
            # if keys aren't the indices, try to get index from nested structure if present
            try:
                qidx = int(v.get("index"))
            except Exception:
                continue

        fid = None
        # Prefer 'rb_fidelity'[0] if present
        if isinstance(v, dict):
            rb = v.get("rb_fidelity")
            if isinstance(rb, (list, tuple)) and len(rb) > 0:
                try:
                    fid = float(rb[0])
                except Exception:
                    fid = None
            # fallback to readout fidelity
            if fid is None:
                rd = v.get("readout", {})
                if isinstance(rd, dict) and "fidelity" in rd:
                    try:
                        fid = float(rd.get("fidelity"))
                    except Exception:
                        fid = None
        # If no fidelity found, skip
        if fid is None:
            logging.debug(f"No fidelity for qubit {qidx} in {calib_path}; skipping")
            continue

        qubit_fidelities.append((qidx, fid))

    # Sort by fidelity descending, tie-break by qubit index ascending
    qubit_fidelities.sort(key=lambda x: (-x[1], x[0]))

    sorted_qubits = [q for q, _ in qubit_fidelities]
    return sorted_qubits


def get_best_edges(hash_id, run_id):
    """
    Extract available edges from bell_tomography results.

    Args:
        hash_id (str): Git commit hash

    Returns:
        dict: Dictionary with qubit counts as keys and [qubit_list, fidelity] as values
    """
    results_file = repo_root / "data" / hash_id / run_id / "bell_tomography" / "results.json"
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        # Extract best_edges_k_qubits information from results
        best_edges_k_qubits = results.get("best_qubits", {})
        return best_edges_k_qubits
    except Exception as e:
        logging.error(f"Could not read bell_tomography results: {e}")
        # fallback with dummy data
        raise e


def has_results(hash_id, run_id, experiment_name, logger=None):
    """
    Check if results.json already exists for an experiment.

    Args:
        hash_id (str): Git commit hash
        run_id (str): Experiment run ID
        experiment_name (str): Name of the experiment
        logger: Optional logger instance

    Returns:
        bool: True if results.json exists, False otherwise
    """
    results_file = repo_root / "data" / hash_id / run_id / experiment_name / "results.json"
    exists = results_file.exists()
    if logger and exists:
        logger.info(f"Skipping {experiment_name}: results.json already exists at {results_file}")
    return exists


def parse_args():
    parser = argparse.ArgumentParser(description="Run all experiment scripts.")
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        help="Execution device to pass to each experiment script.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/runscripts.log",
        help="Path to the log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    return parser.parse_args()


def setup_logger(log_file: str, level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger("runscripts")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)

    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def run_script(logger: logging.Logger, script_path: str, device: str, tag: str) -> int:
    if not os.path.exists(script_path):
        logger.warning(f"main.py not found in {tag}")
        return 1

    logger.info(f"Running {script_path} with device={device}")
    cmd = [sys.executable, "-u", script_path, "--device", device]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info(f"[{tag}] {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            logger.error(f"{script_path} exited with code {proc.returncode}")
        else:
            logger.info(f"Finished {script_path}")
        return proc.returncode or 0
    except Exception:
        logger.exception(f"Error occurred while running {script_path}")
        return 1


def copytree_safe(src: Path, dst: Path, ignore_dirs=None):
    """
    Recursively copy directory `src` into `dst`, skipping directories in `ignore_dirs`
    and continuing on permission (or other) errors. Creates directories as needed.
    """
    src = Path(src)
    dst = Path(dst)
    ignore_dirs = set(ignore_dirs or [])

    for root, dirs, files in os.walk(src, topdown=True):
        root_path = Path(root)

        # prune ignored directories in-place so os.walk won't descend into them
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # compute destination folder corresponding to current root
        rel = root_path.relative_to(src)
        dest_root = dst / rel
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {dest_root}: {e}")
            # continue anyway; file copies may still succeed for siblings
            pass

        # copy files under this root
        for name in files:
            src_file = root_path / name
            dest_file = dest_root / name
            try:
                # create parent in case mkdir above failed
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
            except Exception as e:
                logger.warning(f"Skipping {src_file} -> {dest_file}: {e}")

def run_script_with_args(
    logger: logging.Logger, script_path: str, cmd_args: list, tag: str
) -> int:
    # Convert to absolute path and resolve symlinks for existence check
    abs_script_path = os.path.realpath(script_path)
    if not os.path.exists(abs_script_path):
        logger.warning(f"main.py not found in {tag} (tried: {abs_script_path})")
        return 1

    logger.info(f"Running {abs_script_path} with args={cmd_args}")
    # ensure all cmd args are strings to avoid subprocess TypeError
    safe_args = [str(a) for a in cmd_args]
    # Use the original script_path (not resolved) so that sys.argv[0] points to the symlink location
    original_abs_path = os.path.abspath(script_path)
    cmd = [sys.executable, "-u", original_abs_path] + safe_args
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info(f"[{tag}] {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            logger.error(f"{abs_script_path} exited with code {proc.returncode}")
        else:
            logger.info(f"Finished {abs_script_path}")
        return proc.returncode or 0
    except Exception:
        logger.exception(f"Error occurred while running {abs_script_path}")
        return 1


def clean_copied_git_directories(root_dir: str):
    # Remove the .git directory inside the copied calibration directory via shell
    git_dir = os.path.join(root_dir, ".git")
    if os.path.isdir(git_dir):
        logger.info(f"Removing git directory {git_dir}")
        try:
            subprocess.run(f"rm -rf -- '{git_dir}'", shell=True, check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Shell removal failed for {git_dir}; attempting fallback")
            try:
                shutil.rmtree(git_dir)
            except Exception:
                logger.exception(f"Fallback removal failed for {git_dir}")


def create_experiment_id(logger: logging.Logger) -> str:
    # If file already exists, read and return the existing RunID
    try:
        if RUN_ID_FILE.exists():
            with open(RUN_ID_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("run_id")
            if existing:
                logger.info(f"Using existing experiment RunID: {existing}")
                return str(existing)
    except Exception as e:
        logger.warning(f"Could not read existing experiment ID file {RUN_ID_FILE}: {e}")

    # Otherwise, create a new RunID and persist it
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    logger.info(f"Starting experiment-run with RunID: {run_id}")
    try:
        # File is in current directory, no need to create parent
        with open(RUN_ID_FILE, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id}, f)
    except Exception as e:
        logger.error(f"Failed to write experiment ID to {RUN_ID_FILE}: {e}")
    return run_id

def remove_experiment_id_file(logger: logging.Logger):
    logger.info(f"Removing experiment ID file {RUN_ID_FILE}")
    try:
        if RUN_ID_FILE.exists():
            RUN_ID_FILE.unlink()
    except Exception as e:
        logger.error(f"Failed to remove experiment ID file {RUN_ID_FILE}: {e}")


def main():
    args = parse_args()
    logger = setup_logger(args.log_file, args.log_level)

    # Generate RunID for these experiments and write it to /mnt/home/Scinawa/CQT-reporting/scripts/current_experiment_id.json
    run_id = create_experiment_id(logger)


    # COPY RUNCARD INTO DATA SECTION
    repo = Repo(CURRENT_CALIBRATION_DIRECTORY)
    commit = repo.commit()
    hash_id = commit.hexsha
    calibration_dir = repo_root / "data" / "calibrations" / hash_id
 
    # Load experiment list from configuration file
    experiment_groups = load_experiment_list()

    # Copy /mnt/scratch/qibolab_platforms_nqch into data/<hash_id>/
    try:
        # Copy, skipping .git and continuing on any copy errors
        copytree_safe(
            Path(CURRENT_CALIBRATION_DIRECTORY),
            calibration_dir,
            ignore_dirs={".git", "__pycache__"},
        )

        commit_info = {
            "commit_hash": hash_id,
            "commit_message": commit.message.strip(),
            "experiment_date": datetime.now().strftime(time_format),
            "calibration_date": datetime.fromtimestamp(commit.committed_date).strftime(
                time_format
            ),
        }

        msg_file = calibration_dir / "commit_info.json"
        with open(msg_file, "w") as f:
            json.dump(commit_info, f, indent=4)

    except Exception as e:
        logger.error(f"Failed to copy calibration directory: {e}")

    clean_copied_git_directories(calibration_dir)

    overall_rc = 0

    # Phase 1: Run initial experiments
    logger.info("Phase 1: Running initial experiments")
    for experiment in experiment_groups.get("calibration", []):
        print("\n\n\n")
        # Check if results already exist
        if has_results(hash_id, run_id, experiment, logger):
            logger.info(f"Skipping {experiment}: results already exist")
            continue
        
        script_path = os.path.join(base_path, experiment, "main.py")
        logger.info(f"Starting initial experiment: {experiment}")
        rc = run_script(logger, script_path, args.device, experiment)
        overall_rc = overall_rc or rc

    # Get best edges from bell_tomography results
    best_edges_k_qubits = get_best_edges(hash_id, run_id)
    logger.info(f"Best edges found: {best_edges_k_qubits}")



    # Get best qubits from "initial experiments" results
    best_qubits = get_best_qubits(hash_id)
    logger.info(f"Best qubits found: {best_qubits}")

    # Phase 2: Run single qubit experiments based on available qubits
    logger.info("Phase 2: Running single-qubit experiments")
    for experiment in experiment_groups.get("1", []):
        print("\n\n\n")
        # Check if results already exist
        if has_results(hash_id, run_id, experiment, logger):
            logger.info(f"Skipping {experiment}: results already exist")
            continue
        
        script_path = os.path.join(base_path, experiment, "main.py")
        # pass qubit id as string
        cmd_args = ["--device", args.device, "--qubit_id", str(best_qubits.pop())]
        rc = run_script_with_args(logger, script_path, cmd_args, experiment)
        overall_rc = overall_rc or rc
    

    # Phase 3: Run qubit-specific experiments based on available edges
    logger.info("Phase 3: Running qubit-specific experiments")
    for qubit_count_key, qubit_data in best_edges_k_qubits.items():
        section_name = str(qubit_count_key)
        
        if section_name in experiment_groups:
            nodes: list[int] = []
            avg_fidelity = None
            edge_pairs =   best_edges_k_qubits[section_name][0][0]

            for experiment in experiment_groups[section_name]:
                # Check if results already exist
                if has_results(hash_id, run_id, experiment, logger):
                    logger.info(f"Skipping {experiment}: results already exist")
                    continue
                
                script_path = os.path.join(base_path, experiment, "main.py")
                print("\n\n\n")
                qubit_list_str = json.dumps(edge_pairs)
                
                cmd_args = ["--device", args.device, "--qubits_list", qubit_list_str]
                # import pdb
                # pdb.set_trace()
                rc = run_script_with_args(logger, script_path, cmd_args, experiment)
                overall_rc = overall_rc or rc


    # if overall_rc == 0:  # disabled: upload always runs regardless of experiment failures
    # Load credentials from ~/.env_user
    secrets = load_secrets()
    set_server(server_url=secrets["CQT_SERVER_URL"], api_token=secrets["CQT_API_TOKEN"])

    # Upload whatever results exist
    rsp = results_upload(hashID=hash_id, runID=run_id, data_folder="./data")
    logging.info(rsp)

    # Cleanup: always remove experiment ID file
    remove_experiment_id_file(logger)
    sys.exit(overall_rc)





if __name__ == "__main__":
    main()
