import pathlib
import os
import json
import argparse
import time
import ast

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

def compute_mermin(frequencies, mermin_coefficients):
    """Computes the chsh inequality out of the frequencies of the 4 circuits executed."""
    assert len(frequencies) == len(mermin_coefficients)
    m = 0
    for j, freq in enumerate(frequencies):
        for key in freq:
            m += (
                mermin_coefficients[j]
                * freq[key]
                * (-1) ** (sum([int(key[k]) for k in range(len(key))]))
            )
    nshots = sum(freq[x] for x in freq)
    if nshots != 0:
        return float(m / nshots)

    return 0


def get_mermin_polynomial(n):
    assert n > 1
    m0 = X(0)
    m0p = Y(0)
    for i in range(1, n):
        mn = m0 * (X(i) + Y(i)) + m0p * (X(i) - Y(i))
        mnp = m0 * (Y(i) - X(i)) + m0p * (X(i) + Y(i))
        m0 = mn.expand()
        m0p = mnp.expand()
    m = m0 / 2 ** ((n - 1) // 2)
    return SymbolicHamiltonian(m.expand())


def get_readout_basis(mermin_polynomial: SymbolicHamiltonian):
    return [
        "".join([factor.name[0] for factor in term.factors])
        for term in mermin_polynomial.terms
    ]


def get_mermin_coefficients(mermin_polynomial: SymbolicHamiltonian):
    return [term.coefficient.real for term in mermin_polynomial.terms]


def create_mermin_circuit(qubits, nqubits=20):
    c = Circuit(nqubits)
    c.add(gates.H(qubits[0]))
    c.add([gates.CNOT(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)])
    c.add(gates.RZ(qubits[0], 0))
    return c

def create_mermin_circuit_edges(qubit_edge_list, nqubits=20):
    edges_left = qubit_edge_list.copy()

    c = Circuit(20)

    qubits_done = set()
    edge = qubit_edge_list[0]
    edges_left.remove(edge)
    qubits_done |= set(edge)

    c.add(gates.H(edge[0]))
    c.add(gates.CNOT(edge[0], edge[1]))

    while len(edges_left) > 0:
        for edge in edges_left:
            if len(set(edge)&qubits_done)==2:
                edges_left.remove(edge) 
                continue
            elif len(set(edge)&qubits_done)==1:
                q1 = list(set(edge)&qubits_done)[0]
                q2 = list(set(edge)-qubits_done)[0]
                c.add(gates.CNOT(q1, q2))
                edges_left.remove(edge)
                qubits_done |= set(edge)
    c.add(gates.RZ(qubit_edge_list[0][0], 0))
    return c


def create_mermin_circuits(qubits: list[int], readout_basis: list[str]):
    c = create_mermin_circuit(qubits)
    circuits = [c.copy(deep=True) for _ in readout_basis]

    for circuit, basis in zip(circuits, readout_basis):
        for q, base in zip(qubits, basis):
            if base == "Y":
                circuit.add(gates.SDG(q))
            circuit.add(gates.H(q))
            circuit.add(gates.M(q))

    return circuits


def create_mermin_circuits_edges(qubit_edge_list: list[list[int]], readout_basis: list[str]):
    c = create_mermin_circuit_edges(qubit_edge_list)
    circuits = [c.copy(deep=True) for _ in readout_basis]

    qubits = set()
    for edge in qubit_edge_list:
        qubits |= set(edge)
    qubits = list(qubits)

    for circuit, basis in zip(circuits, readout_basis):
        for q, base in zip(qubits, basis):
            if base == "Y":
                circuit.add(gates.SDG(q))
            circuit.add(gates.H(q))
            circuit.add(gates.M(q))

    return circuits


def main(nqubits, qubits_list, device, nshots):
    results = dict()
    data = dict()

    qubits = set()
    for edge in qubits_list:
        qubits |= set(edge)
    qubits = list(qubits)

    results["x"] = {}
    results["y"] = {}
    data["qubits_list"] = qubits_list
    data["nqubits"] = nqubits
    data["nshots"] = nshots
    data["device"] = device

    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)

    _tmp_runtimes = []

    if device == "numpy":
        set_backend("numpy")
    else:
        set_backend("qibolab", platform=device)

    poly = get_mermin_polynomial(nqubits)
    coeff = get_mermin_coefficients(poly)
    basis = get_readout_basis(poly)

    circuits = create_mermin_circuits_edges(qubits_list, basis)
    theta_array = np.linspace(0, 2 * np.pi, 50)
    result = np.zeros(len(theta_array))
    for idx, theta in enumerate(theta_array):
        frequencies = []
        for circ in circuits:
            circ.set_parameters([theta])
            circ, _ = custom_pipeline(circ)
            start_time = time.time()
            freq = circ(nshots=nshots).frequencies()
            end_time = time.time()
            _tmp_runtimes.append(end_time - start_time)

            frequencies.append(freq)
        result[idx] = compute_mermin(frequencies, coeff)

    results["x"][f"{qubits}"] = theta_array.tolist()
    results["y"][f"{qubits}"] = result.tolist()

    runtime_seconds = (
        sum(_tmp_runtimes) if _tmp_runtimes else 0.0
    )
    results["runtime"] = runtime_seconds
    results["description"] = f"Mermin's algorithm for {nqubits} qubits."
    results["qubits_used"] = qubits

    # Write to data/<scriptname>/<device>/results.json
    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(out_dir / f"data_mermin_{nqubits}q.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nqubits",
        default=3,
        type=int,
        help="Total number of qubits",
    )
    parser.add_argument(
        "--qubits_list",
        default="[[18, 14], [18, 19]]",
        type=str,
        help="Target qubits list as string representation",
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = parser.parse_args()

    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = ast.literal_eval(args.qubits_list)
        # Ensure all elements are integers
        qubits_list = [[int(q) for q in qubits_list[i]] for i in range(len(qubits_list))]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)

    main(args.nqubits, qubits_list, args.device, args.nshots)