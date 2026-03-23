import sys
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so clientdb/ is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clientdb.client import (
    set_server,
    calibrations_upload, calibrations_list, calibrations_download, calibrations_get_latest,
    results_upload, results_download,unzip_bytes_to_folder,test,results_list,set_best_run, get_best_run,
    get_best_n_runs,upload_all_calibrations,upload_all_experiment_runs
)


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


def main():
    parser = argparse.ArgumentParser(description="Upload experiment results")
    parser.add_argument("--hashid", required=True, help="Hash ID for the calibration")
    parser.add_argument("--runid", help="Run ID for the experiment")
    parser.add_argument("--upload-calibration", action="store_true",
                        help="Upload calibration instead of results")
    args = parser.parse_args()

    if not args.upload_calibration and not args.runid:
        parser.error("--runid is required when not using --upload-calibration")

    # Load credentials from .secrets.toml
    secrets = load_secrets()
    server_url = secrets["CQT_SERVER_URL"]
    api_token = secrets["CQT_API_TOKEN"]

    set_server(server_url=server_url, api_token=api_token)

    # Always ensure calibration is registered before uploading results
    print("Uploading calibration...")
    rsp = calibrations_upload(hashID=args.hashid, calibrations_folder="./data/calibrations")
    print(f"Calibration upload: {rsp}")

    if not args.upload_calibration:
        # Upload results too (default behavior)
        print("Uploading results...")
        rsp = results_upload(hashID=args.hashid, runID=args.runid, data_folder="./data")
        print(f"Results upload: {rsp}")

if __name__ == "__main__":
    main()


