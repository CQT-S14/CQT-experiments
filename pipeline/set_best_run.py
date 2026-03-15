import sys
from pathlib import Path

# Ensure repo root is on sys.path so clientdb/ is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clientdb.client import (
    set_server,
    calibrations_upload, calibrations_list, calibrations_download, calibrations_get_latest,
    results_upload, results_download,unzip_bytes_to_folder,test,results_list,set_best_run, get_best_run,
    get_best_n_runs,upload_all_calibrations,upload_all_experiment_runs
)


def _load_env_user():
    env_file = Path.home() / ".env_user"
    env = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env

_env = _load_env_user()
set_server(server_url=_env["CQT_SERVER_URL"], api_token=_env["CQT_API_TOKEN"])

# rsp = calibrations_upload(hashID="1e1f7e1d1af58009eda1986bb3689e6b9b2356b6", calibrations_folder="./data/calibrations")
# print(rsp)

# rsp = calibrations_upload(hashID="3826882f81128980b5e49b0e1bec76e24e40e158", calibrations_folder="./data/calibrations")
# print(rsp)

# rsp = results_upload(hashID="1e1f7e1d1af58009eda1986bb3689e6b9b2356b6", runID="20251123023814", data_folder="./data")
# print(rsp)
# rsp = results_upload(hashID="3826882f81128980b5e49b0e1bec76e24e40e158", runID="20251201101512", data_folder="./data")
# print(rsp)

rsp = get_best_run()
print(rsp)

rsp = set_best_run(calibrationHashID="3826882f81128980b5e49b0e1bec76e24e40e158",runID='20251201134523') 
print(rsp)

rsp = get_best_run()
print(rsp)

# Mark a few best runs over time
# set_best_run("cal_hash_A", "run_001")
# set_best_run("cal_hash_B", "run_002")
# set_best_run("cal_hash_A", "run_003")


# Get the most recent best run
# cal_hash, run_id, ts = get_best_run()
# print("Current best:", cal_hash, run_id, ts)

# Get the last 5 best runs
# history = get_best_n_runs(5)
# for cal_hash, run_id, ts in history:
#     print("Best at:", ts, "->", cal_hash, run_id)

# print(get_best_run())
