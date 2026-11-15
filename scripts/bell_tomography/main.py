import argparse
import json
import sys
import pathlib
from pathlib import Path
import time
import importlib.util
import sys
from pathlib import Path
import networkx as nx
from itertools import combinations
import pdb


from qibo import Circuit, gates, set_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # scripts/config.py

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

from config import CURRENT_CALIBRATION_DIRECTORY
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

def bell_circuit():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    return circuit


def run_tomography(targets, device, nshots, root_path):
    if device == "numpy":
        set_backend("numpy")
        platform = None
    else:
        set_backend("qibolab", platform=device)
        platform = device

    results = {}
    data = {
        "targets": targets,
        "nshots": nshots,
        "device": device,
    }

    # Only run Executor if using qibolab
    if platform:
        with Executor.open(
            path=root_path,
            platform=platform,
            targets=[(targets[0], targets[1])],
            update=True,
            force=True,
        ) as e:
            circuit = bell_circuit()
            circuit, _ = custom_pipeline(circuit)
            start_time = time.time()
            output = e.two_qubit_state_tomography(
                circuit=circuit, targets=[(targets[0], targets[1])]
            )
            end_time = time.time()
            # import pdb
            # pdb.set_trace()
            runtime_seconds = end_time - start_time
            report(e.path, e.history)
            # Save frequencies if available
            
            results["fidelity"] = output.results.fidelity
            results["runtime"] = runtime_seconds
            results["qubits_used"] = targets
            results["description"] = f"Bell state tomography."
    else:
        # Simulate with numpy backend
        start_time = time.time()
        circuit = bell_circuit()
        r = circuit()
        probs = r.probabilities()
        num_bits = 2
        all_bitstrings = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
        # Convert probabilities to expected counts, then normalize to frequencies
        freq = {bs: probs[i] for i, bs in enumerate(all_bitstrings)}
        end_time = time.time()
        runtime_seconds = end_time - start_time
        results["frequencies"] = freq
        results["description"] = f"State tomography on bell states."
        results["runtime"] = runtime_seconds
        results["qubits_used"] = targets

    return data, results


def main(device, nshots):
    scriptname = Path(__file__)
    out_dir = config.output_dir_for(__file__, device)
    out_dir_tmp = out_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)



    targets = get_list_of_pairs()

    print(f"Using qubit pairs: {targets}")
    two_qb_fidelities = {}
    results = {}


    
    for pair in targets:
        try:
            data, results_exec = run_tomography(pair, device, nshots, out_dir_tmp)
        except Exception as e:
            print(f"Failed to run tomography on qubits {pair}: {e}")
            continue
        # postprocessing

        try:
            two_qb_fidelities[pair] = results_exec['fidelity'][pair]
            results[str(pair)] = results_exec['fidelity'][pair]
        except KeyError:
            two_qb_fidelities[pair] = 0


    G = nx.Graph()

    # Add edges with weights
    for (u, v), w in two_qb_fidelities.items():
        G.add_edge(u, v, weight=w)

    #pdb.set_trace()

    results["best_qubits"] = get_best_qubits_tuples(G)
    
    # import pdb
    #pdb.set_trace()

    try:
        # with (out_dir / "data.json").open("w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to write output files to {out_dir}: {e}")


def get_list_of_pairs():
    # gets the connectivity 

    # Path to your file
    file_path = Path(CURRENT_CALIBRATION_DIRECTORY) / Path("sinq20") / Path("platform.py")    

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("platform_module", file_path)
    platform_module = importlib.util.module_from_spec(spec)
    sys.modules["platform_module"] = platform_module
    spec.loader.exec_module(platform_module)

    # Now you can access connectivity
    connectivity = getattr(platform_module, "connectivity", None)

    return connectivity




def get_best_qubits_tuples(two_qubits_fidelity_graph, sizes=(2, 3, 4, 5), weight_attr="weight"):
    G = two_qubits_fidelity_graph
    best_qubits = {}

    for k in sizes:
        best_avg = float("-inf")
        best_total = float("-inf")
        best_nodes = None
        best_edges = None

        for nodes in combinations(G.nodes(), k):
            sub = G.subgraph(nodes)  # induced subgraph (view is fine; no need to copy)
            if not nx.is_connected(sub):
                continue

            # Compute average edge weight over existing edges in the induced subgraph
            m = sub.number_of_edges()
            if m == 0:
                continue  # connected subgraph of size k always has m >= k-1, but just in case

            total = sum(data.get(weight_attr, 1.0) for _, _, data in sub.edges(data=True))
            avg = total / m

            # Tie-break: higher avg -> higher total -> lexicographically smaller nodes
            if (avg > best_avg or
               (avg == best_avg and total > best_total) or
               (avg == best_avg and total == best_total and best_nodes is not None and tuple(sorted(nodes)) < tuple(sorted(best_nodes)))):
                best_avg = avg
                best_total = total
                best_nodes = list(nodes)
                # Store edges as list of [node1, node2] pairs
                best_edges = [[u, v] for u, v in sub.edges()]

        # Store as list containing one tuple: (edges, avg_fidelity)
        if best_edges is not None:
            best_qubits[k] = [([[u, v] for u, v in best_edges], best_avg)]
        else:
            best_qubits[k] = []  # No connected subgraph exists of size k

    # import pdb
    # pdb.set_trace()

    return best_qubits




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["numpy", "sinq20"],
        default="numpy",
        type=str,
        help="Device to use (numpy or sinq20)",
    )
    parser.add_argument(
        "--nshots",
        default=2000,
        type=int,
        help="Number of shots for each circuit",
    )
    args = parser.parse_args()
    main(args.device, args.nshots)
