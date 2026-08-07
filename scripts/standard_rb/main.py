import os
import pathlib

# os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path("/mnt/scratch/qibolab_platforms_nqch").as_posix()

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

import argparse
import json
import sys
from pathlib import Path as _P
import time

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


def main(nqubits, nshots, niter, depths, device, **kwargs):

    results = dict()
    data = dict()

    results["fidelity"] = {}
    data["nqubits"] = nqubits
    data["nshots"] = nshots
    data["niter"] = niter
    data["depths"] = depths
    data["device"] = device

    _tmp_runtimes = []

    out_dir = config.output_dir_for(__file__, device)

    # import pdb
    # pdb.set_trace()

    root_path = out_dir  # / "standard_rb"
    platform = device

    params = {
        "depths": depths,  # [1, 2, 3, 6, 12, 21, 40, 73, 135, 250]
        "niter": niter,  # 10
    }

    with Executor.open(
        path=root_path,
        platform=platform,
        targets=list(range(nqubits)),
        update=True,
        force=True,
    ) as e:

        start_time = time.time()
        e.standard_rb(parameters=params)
        end_time = time.time()
        _tmp_runtimes.append(end_time - start_time)

        report(e.path, e.history)

        for qubit in range(1, nqubits + 1):
            qubit_id = qubit - 1
            results["fidelity"][f"{qubit_id}"] = float(
                e.platform.calibration.single_qubits[qubit_id].rb_fidelity[0]
            )

    runtime_seconds = sum(_tmp_runtimes) / len(_tmp_runtimes) if _tmp_runtimes else 0.0
    results["runtime"] = f"{runtime_seconds:.2f} seconds."
    results["description"] = (
        f"Single qubit Randomized Benchmarking executed on {device} backend with {nshots} shots per circuit and {niter} iterations per depth. \n The gate fidelity is computed in a SPAM robust way."
    )
    results["qubits_used"] = list(range(nqubits))

    # Write to data/<scriptname>/<device>/results.json

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (out_dir / "data.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits",
        default=5,
        type=int,
        help="Total number of qubits",
    )
    parser.add_argument(
        "--nshots",
        default=100,
        type=int,
        help="Number of shots for each circuit",
    )
    parser.add_argument(
        "--niter",
        default=3,
        type=int,
        help="Iteration on each depths",
    )
    parser.add_argument(
        "--depths",
        default=[1, 2, 3, 5, 6, 12],  #  , 21, 40, 73, 135, 250],
        type=list,
        help="List of depths for the protocol",
    )
    parser.add_argument(
        "--device",
        choices=["sinq20", "numpy"],
        default="sinq20",
        type=str,
        help="Device to use",
    )
    parser.add_argument(
        "--qubit_id",
        default="-",
        type=str,
        help="Dummy parameter for compliance",
    )

    args = vars(parser.parse_args())
    main(**args)
