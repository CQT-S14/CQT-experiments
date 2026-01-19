import numpy as np
import qibo
import os
import json
import sys
import time
from pathlib import Path as _P

''' Quantum Fourier Transform circuit from scratch (ONLY THREE QUBITS)
    using qibo framework with !MANUAL TRANSPILATION!
'''

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

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

def transpiled_cnot(circuit, control, target):
    """Transpile CNOT using native gates."""
    circuit.add(gates.H(target))
    circuit.add(gates.CZ(control, target))
    circuit.add(gates.H(target))
    
    return circuit

def transpiled_swap(circuit, qubit1, qubit2):
    """Transpile SWAP using native gates."""
    circuit = transpiled_cnot(circuit, qubit1, qubit2)
    circuit = transpiled_cnot(circuit, qubit2, qubit1)
    circuit = transpiled_cnot(circuit, qubit1, qubit2)
    
    return circuit

def order_chain_edges(edges):
    """Ensure edges form a -> b -> c ordering."""
    (a, b), (c, d) = edges
    # Find shared qubit
    shared = list(set([a, b]) & set([c, d]))
    assert len(shared) == 1, "Edges do not share exactly one qubit."
    shared = shared[0]

    # Determine left and right
    left  = a if a != shared else b
    right = c if c != shared else d

    return [[left, shared], [shared, right]]

def QFT(qubits_list):
    '''
    Quantum Fourier Transform circuit from scratch (ONLY THREE QUBITS)
    
    Args:   - list of edges (connections)
            
    Returns: qibo circuit
    '''
    
    qubits_set = set(sum(qubits_list, []))
    total_qubits = int(max(qubits_set) + 1)     # number of total qubits of the circuit

    circuit = qibo.Circuit(total_qubits)                # Circuit initialization

    # Add Hadamard at the beginning
    for q in qubits_set:
        circuit.add(qibo.gates.H(q))

    qubits_list = order_chain_edges(qubits_list)
    
    # Ensure edges are ordered in a chain
    qubits_list = order_chain_edges(qubits_list)
    
    # QFT from scratch for three qubits with swap for 
    # connection q0--q1--q2
    circuit.add(qibo.gates.H(qubits_list[0][0]))
    theta1 = np.pi / 2
    circuit.add(qibo.gates.CU1(qubits_list[0][1], qubits_list[0][0], theta1))
    # circuit.add(qibo.gates.SWAP(qubits_list[0][1], qubits_list[1][1]))
    circuit = transpiled_swap(circuit, qubits_list[0][1], qubits_list[1][1])
    theta2 = np.pi / 4
    circuit.add(qibo.gates.CU1(qubits_list[0][1], qubits_list[0][0], theta2))
    circuit.add(qibo.gates.H(qubits_list[1][1]))
    circuit.add(qibo.gates.CU1(qubits_list[1][1], qubits_list[1][0], theta1))
    circuit.add(qibo.gates.H(qubits_list[0][1]))
    
    # Add measurement
    for q in qubits_set:
        circuit.add(qibo.gates.M(q))

    return circuit


def main(qubits_list, device, nshots):
    # Remove all qibo_client usage and via_client logic
    # Set backend as in template/main.py, GHZ/main.py, mermin/main.py
    if device == "numpy":
        qibo.set_backend("numpy")
    else:
        qibo.set_backend("qibolab", platform=device)

    # Here the list of the best three qubits
    
    qubits_set = set(sum(qubits_list, []))
    num_qubits = len(qubits_set)                         # number of qubits 

    frequencies = dict()
    all_bitstrings = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]

    print(f"Trying edges: {qubits_list}")

    circuit = QFT(qubits_list)
    circuit, _ = custom_pipeline(circuit)

    start = time.perf_counter()
    result = circuit(nshots=nshots)
    end = time.perf_counter()

    # Circuit output frequencies and store them
    frequencies = {bitstr: result.frequencies().get(bitstr, 0) for bitstr in all_bitstrings}

    num_gates = len(circuit.queue)
    depth = circuit.depth

    results = dict()
    data = dict()

    results["description"] = {}
    results["circuit_depth"] = {}
    results["gates_count"] = {}
    results["duration"] = {}
    results["frequencies"] = {}
    results["edges"] = {}
    data["nshots"] = nshots
    data["device"] = device

    results = {
        "description": f"Implementation of the Quantum Fourier Transform on three qubits with manual transpilation. The number of gates is {num_gates}, the depth of the circuit is {depth}",
        "circuit_depths": depth,
        "gates_counts": num_gates,
        "duration": f"{(end-start):.3f} seconds.",
        "frequencies": frequencies,
        "edges": qubits_list,
    }
    
    out_dir = config.output_dir_for(__file__, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f"results.json")

    with open(output_path, "w") as f:
        json.dump(results, f)
        print(f"File saved on {output_path}")

import argparse
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="numpy",
        type=str,
        help="Device to use (e.g., 'nqch' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=1000,
        type=int,
        help="Number of shots for each circuit",
    )
    parser.add_argument(
        "--qubits_list",
        default="[[9,8],[8,13]]",
        type=str,
        help="Target edges list as string representation",
    )
    args = parser.parse_args()
    # Parse the qubit list string into actual list of integers
    try:
        qubits_list = ast.literal_eval(args.qubits_list)
        # Ensure all elements are integers
        qubits_list = [[int(q) for q in edge] for edge in qubits_list]
    except (ValueError, SyntaxError, TypeError):
        print(f"Error: Invalid qubit list format: {args.qubits_list}")
        sys.exit(1)
    main(qubits_list, args.device, args.nshots)