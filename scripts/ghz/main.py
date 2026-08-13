import json
import numpy as np
from qibo import Circuit, gates, set_backend
import os
import argparse
import sys
from pathlib import Path as _P
import time
import ast
import networkx as nx

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

from config import find_longest_chain, find_all_chains
from qibo import Circuit, gates
from qibo.transpiler import (
    NativeGates,
    Passes,
    Unroller
)

glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
natives = NativeGates(0).from_gatelist(glist)
custom_passes = [Unroller(native_gates=natives)]
custom_pipeline = Passes(custom_passes)

def create_ghz_circuit(chain_of_qubits):
    nqubits = len(chain_of_qubits)
    c = Circuit(nqubits, wire_names=chain_of_qubits)
    c.add(gates.H(0))
    for i in range(nqubits - 1):
        c.add(gates.CNOT(i, i + 1))
    c.add(gates.M(_i) for _i in range(nqubits))
    return c


def prepare_ghz_results(frequencies, nshots, nqubits):
    # Calculate success rate for GHZ state (all 0s or all 1s)
    success_keys = ["0" * nqubits, "1" * nqubits]
    total_success = sum(frequencies.get(k, 0) for k in success_keys)
    success_rate = total_success / nshots if nshots else 0.0

    # Prepare output structure
    all_bitstrings = [format(i, f"0{nqubits}b") for i in range(2**nqubits)]
    freq_dict = {bitstr: frequencies.get(bitstr, 0) for bitstr in all_bitstrings}
    return {"success_rate": success_rate, "plotparameters": {"frequencies": freq_dict}}


def run_ghz_experiment(chain_of_qubits, device, nshots, root_path):
    nqubits = len(chain_of_qubits)

    if device == "numpy":
        set_backend("numpy")
        platform = None
    else:
        set_backend("qibolab", platform=device)
        platform = device

    results = {}
    data = {
        "chain_of_qubits": chain_of_qubits,
        "nshots": nshots,
        "device": device,
    }

    # Only run Executor if using qibolab
    if platform:
        with Executor.open(
            path=root_path,
            platform=platform,
            targets=chain_of_qubits,
            update=True,
            force=True,
        ) as e:
            circuit = create_ghz_circuit(chain_of_qubits)
            circuit, _ = custom_pipeline(circuit)
            start_time = time.time()
            # For GHZ, we'll run the circuit directly since qibocal doesn't have a specific GHZ method
            # You might need to adapt this based on available qibocal methods
            result = circuit(nshots=nshots)
            end_time = time.time()
            runtime_seconds = end_time - start_time
            report(e.path, e.history)

            frequencies = result.frequencies()
            ghz_results = prepare_ghz_results(frequencies, nshots, nqubits)

            results.update(ghz_results)
            results["runtime"] = f"{runtime_seconds:.2f} seconds."
            results["qubits_used"] = chain_of_qubits
            results["description"] = (
                f"GHZ circuit with {nqubits} qubits executed on {device} backend with {nshots} shots."
            )
    else:
        # Simulate with numpy backend
        start_time = time.time()
        circuit = create_ghz_circuit(chain_of_qubits)
        result = circuit(nshots=nshots)
        end_time = time.time()
        runtime_seconds = end_time - start_time

        frequencies = result.frequencies()
        ghz_results = prepare_ghz_results(frequencies, nshots, nqubits)

        results.update(ghz_results)
        results["description"] = (
            f"GHZ circuit with {nqubits} qubits executed on numpy backend with {nshots} shots."
        )
        results["runtime"] = f"{runtime_seconds:.2f} seconds."
        results["qubits_used"] = chain_of_qubits

    return data, results


def main(device, nshots, qubits_list):
    scriptname = _P(__file__).stem
    out_dir = config.output_dir_for(__file__, device)
    out_dir_tmp = out_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain_of_qubits = find_longest_chain(qubits_list)
    print(f"{len(chain_of_qubits)}-qubit GHZ on qubits {chain_of_qubits}.")

    try:
        data, results = run_ghz_experiment(chain_of_qubits, device, nshots, out_dir_tmp)
    except Exception as e:
        print(f"Failed to run GHZ experiment on qubits {chain_of_qubits}: {e}")
        return

    try:
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits_list", 
        type=ast.literal_eval, 
        default=[[0, 1]],
        help="List of coupled qubit pairs, e.g. '[[0,1],[0,2],[1,4],[1,5]]', from which we find the longest chain of qubits for GHZ circuit.",
    )
    parser.add_argument("--nshots", type=int, default=1000)
    parser.add_argument("--device", choices=["numpy", "sinq20"], default="numpy")
    args = parser.parse_args()

    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = args.qubits_list
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)

    main(args.device, args.nshots, qubits_list)
