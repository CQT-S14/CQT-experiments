import numpy as np
import qibo
from qibo import gates
import os
import json
import sys
from pathlib import Path as _P
import argparse
import pickle
from collections import Counter
from itertools import islice
from qibo.ui import plot_circuit
import matplotlib.pyplot as plt
import logging
import signal
import shutil
import atexit
import time
import ast
from pathlib import Path


from qibocal.auto.execute import Executor
from qibocal.cli.report import report

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py
from config import build_chain_from_edges, find_all_chains



logger = logging.getLogger(__name__)

def var_circuit(angles_dict, qubits_list, number_of_qubits=20):
    """
    Build a quantum circuit based on the angles dictionary from the JSON file.
    Now maps to physical qubits specified in qubits_list.
    """
    qubits_used = len(qubits_list)
    max_qubit = max(qubits_list)
    circuit = qibo.Circuit(number_of_qubits)

    if qubits_used == 3:
        for i, physical_qubit in enumerate(qubits_list):
            qkey = str(i)
            q_angles = angles_dict.get(qkey, {})
            
            logger.debug("Adding RY (depth 1) on physical qubit %d with angle %s", physical_qubit, q_angles["1"])
            circuit.add(gates.RY(physical_qubit, theta=q_angles["1"]))
        
            logger.debug("Adding RZ (depth 2) on physical qubit %d with angle %s", physical_qubit, q_angles["2"])
            circuit.add(gates.RZ(physical_qubit, theta=q_angles["2"]))
            
            logger.debug("Adding RY (depth 3) on physical qubit %d with angle %s", physical_qubit, q_angles["3"])
            circuit.add(gates.RY(physical_qubit, theta=q_angles["3"]))

        # First entangling layer
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

        # RY rotations
        circuit.add(gates.RY(qubits_list[0], theta=angles_dict["0"]["4"]))
        circuit.add(gates.RY(qubits_list[1], theta=angles_dict["1"]["5"]))
        circuit.add(gates.RY(qubits_list[2], theta=angles_dict["2"]["5"]))

        # Second entangling layer  
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["6"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["6"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

    elif qubits_used == 4:
        for i, physical_qubit in enumerate(qubits_list):
            qkey = str(i)
            q_angles = angles_dict.get(qkey, {})
            
            circuit.add(gates.RY(physical_qubit, theta=q_angles["1"]))
            circuit.add(gates.RZ(physical_qubit, theta=q_angles["2"]))
            circuit.add(gates.RY(physical_qubit, theta=q_angles["3"]))

        # First entangling layer
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))
        circuit.add(gates.RZ(qubits_list[1], theta=angles_dict["1"]["4"]))
        circuit.add(gates.CNOT(qubits_list[0], qubits_list[1]))

        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))
        circuit.add(gates.RZ(qubits_list[2], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[1], qubits_list[2]))

        circuit.add(gates.CNOT(qubits_list[2], qubits_list[3]))
        circuit.add(gates.RZ(qubits_list[3], theta=angles_dict["2"]["4"]))
        circuit.add(gates.CNOT(qubits_list[2], qubits_list[3]))

    return circuit

def load_data(cfg_path):
    """
    Load the configuration data from the JSON file located in the same directory as this script,
    unless an absolute path is provided.
    """
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    return data

def compute_statistics_and_dump_results(data, out_dir, logger):
    """
    Compute statistics from NQCH results and dump to file.
    
    Args:
        data: Dictionary containing experiment data with 'NQCH' and 'noiseless_experiment_ios' keys
        out_dir: Output directory path
        logger: Logger instance
    """
    nqch_results = data.get('NQCH', {})
    noiseless_ios = data.get('noiseless_experiment_ios', {})
    
    if not nqch_results:
        logger.warning("No NQCH results found in data")
        return
    
    # Count matches
    qibo_correct = 0
    pennylane_qibo_match = 0
    original_sim_correct = 0
    total_processed = 0
    
    for idx_str, nqch_result in nqch_results.items():
        # import pdb
        # pdb.set_trace()
        if not isinstance(nqch_result, dict):
            logger.warning(f"NQCH result for index {idx_str} is not a dict, skipping")
            continue

        if 'predicted_label' not in nqch_result:
            continue
            
        original_data = noiseless_ios.get(idx_str)
        if not original_data:
            logger.warning(f"Missing original data for index {idx_str}")
            continue
        
        qibo_pred = nqch_result['predicted_label']
        true_label = original_data.get('label')
        noiseless_label = original_data.get('predicted_label')
        
        if true_label is not None:
            if qibo_pred == true_label:
                qibo_correct += 1
            if true_label == noiseless_label:
                original_sim_correct += 1
        
        if noiseless_label is not None and qibo_pred == noiseless_label:
            pennylane_qibo_match += 1
        
        total_processed += 1
    
    if total_processed == 0:
        logger.warning("No valid results to compute statistics")
        return
    
    # Compute accuracies
    qibo_accuracy = qibo_correct / total_processed
    pennylane_vs_qibo_accuracy = pennylane_qibo_match / total_processed
    original_sim_accuracy = original_sim_correct / total_processed
    
    logger.info("Qibo accuracy (qibo vs true label): %.4f (%d/%d)", 
                qibo_accuracy, qibo_correct, total_processed)
    logger.info("PennyLane vs Qibo accuracy (qibo vs noiseless): %.4f (%d/%d)", 
                pennylane_vs_qibo_accuracy, pennylane_qibo_match, total_processed)
    logger.info(
        "Original noiseless simulation agreement (true vs noiseless): %.4f (%d/%d) (expected: %.4f)",
        original_sim_accuracy,
        original_sim_correct,
        total_processed,
        data.get("accuracy", float("nan")),
    )
    
    # Store statistics
    if 'NQCH' not in data:
        data['NQCH'] = {}
    
    data['NQCH']['_statistics'] = {
        'qibo_accuracy': qibo_accuracy,
        'verification_vs_NQHC_accuracy': pennylane_vs_qibo_accuracy,
        'original_sim_accuracy_over_processed_subset': original_sim_accuracy,
        'processed_count': total_processed,
        'qibo_correct': qibo_correct,
        'pennylane_qibo_match': pennylane_qibo_match,
        'original_sim_correct': original_sim_correct
    }
    
    # Write full updated data to results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Full results (with accuracies) written to %s", results_path)


def run_quantum_experiment(data, idx, qubits_list, num_qubits, physical_output_qubit, nshots):

    angles_dict = data['noiseless_experiment_ios'][str(idx)]['angles']
    true_label = data['noiseless_experiment_ios'][str(idx)]['label']
    noiseless_label = data['noiseless_experiment_ios'][str(idx)]['predicted_label']

    logger.debug("Building circuit for data index %s; true=%s noiseless=%s", idx, true_label, noiseless_label)
    circuit = var_circuit(angles_dict, qubits_list[:num_qubits])
    circuit.add(gates.M(physical_output_qubit))

    start_time = time.time()
    result = circuit(nshots=nshots)
    end_time = time.time()
    duration = end_time - start_time

    frequencies = result.frequencies()
    total_counts = sum(frequencies.values())
    if total_counts == 0:
        logger.warning("No counts returned for idx %s (total_counts=0)", idx)
        
    c1 = frequencies.get('1', 0)
    c0 = frequencies.get('0', 0)
    total_counts = c0 + c1
    if total_counts > 0:
        exp_z = (c0 - c1) / total_counts  # <Z> = P(0) - P(1)
    else:
        exp_z = 0.0
    sigmoid_expval = 1.0 / (1.0 + np.exp(-exp_z))  # sigmoid(<Z>)
    predicted_label = 1 if sigmoid_expval > 0.5 else 0

    return {
        'predicted_label': predicted_label,
        'sigmoid_expval': sigmoid_expval
    }



def main(qubits_list, device, nshots, debug=False, args=None, input_filename="input.json", number_of_datapoints_output=100000):
    # configure logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    qibo.set_backend("qibolab", platform=device)
    logger.info("Using backend: qibolab (platform=%s)", device)
    
    base_dir = _P(os.path.dirname(os.path.abspath(sys.argv[0])))
    cfg_path = base_dir / input_filename
    tmp_path = base_dir / "results_tmp.json"  # Temporary results file in current directory
    
    logger.debug("Loading config from %s", cfg_path)

    data = load_data(cfg_path)
    
    # Initialize NQCH results if not present
    if 'NQCH' not in data:
        data['NQCH'] = {}

    num_qubits = data['args']['num_qubits']
    output_qubit = data['args']['output_qubit']

    folder_name = Path(sys.argv[0]).parts[-2]  
    out_dir = Path(config.output_dir_for(sys.argv[0], device))
    # replace the last folder name with folder_name:
    out_dir = out_dir.with_name(folder_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # TODO understand which is the right output qubit....
    physical_output_qubit = qubits_list[min(output_qubit, len(qubits_list) - 1)]

    # Use Executor for hardware execution
    root_path = out_dir / "executor_tmp"

    processed_count = 0

    # Define cleanup function for graceful shutdown
    def cleanup_handler(signum=None, frame=None):
        logger.warning("Received termination signal, cleaning up...")
        # Move temporary results to output directory
        if tmp_path.exists():
            final_path = out_dir / "results.json"
            logger.info(f"Moving results from {tmp_path} to {final_path}")
            shutil.move(tmp_path, final_path)
        if signum is not None:
            sys.exit(0)
    
    # Register signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, cleanup_handler)
    atexit.register(cleanup_handler)

    with Executor.open(
        path=root_path,
        platform=device,
        targets=qubits_list[:num_qubits],
        update=True,
        force=True,
    ) as e:
        logger.info("Using Executor with platform: %s, qubits: %s", device, qubits_list[:num_qubits])
        

        start_time = time.time()
        for idx, _ in data['noiseless_experiment_ios'].items():
            # Skip if we've already processed this data point (has sigmoid_expval field)
            if str(idx) in data.get('NQCH', {}) and 'sigmoid_expval' in data['NQCH'][str(idx)]:
                logger.info(f"Skipping data point {idx} - already processed")
                continue
                
            # If we've processed enough points, break
            if processed_count >= number_of_datapoints_output:
                logger.info(f"Reached limit of {number_of_datapoints_output} data points, stopping")
                break

            experiment_results = run_quantum_experiment(data, idx, qubits_list, num_qubits, physical_output_qubit, nshots)

            # Store results in data structure
            data['NQCH'][str(idx)] = {
                'sigmoid_expval': float(experiment_results['sigmoid_expval']),
                'predicted_label': experiment_results['predicted_label'],
                'is_correct': experiment_results['predicted_label'] == data['noiseless_experiment_ios'][str(idx)]['label'],
            }

            # Write to temporary results file in current directory
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            logger.debug(f"Updated results file {tmp_path} with data point {idx}")
            logger.info("Processed data point %s: predicted=%s true=%s noiseless=%s prob=%.3f",
                        idx, 
                        experiment_results['predicted_label'],
                        data['noiseless_experiment_ios'][str(idx)]['label'],
                        data['noiseless_experiment_ios'][str(idx)]['predicted_label'],
                        experiment_results['sigmoid_expval'])
            
            processed_count += 1
        end_time = time.time()
        duration = end_time - start_time
        logger.info("Processed %d data points in %.2f seconds (%.2f s/data point)", 
                    processed_count, duration, duration / processed_count if processed_count > 0 else float('inf'))
        report(e.path, e.history)
    
    logger.info(f"Processing complete. Total processed data points: {processed_count}")

    # Compute statistics and write final results

    data["description"] = f"Classification of benchmarking ML dataset using a pretrained variational circuit."
    data["runtime"] = duration
    data["qubits_used"] = qubits_list
    compute_statistics_and_dump_results(data, out_dir, logger)
    
    # Remove temporary file from current directory after successful completion
    if tmp_path.exists():
        tmp_path.unlink()
        logger.info(f"Removed temporary results file {tmp_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qubits_list",
        type=ast.literal_eval, 
        default=[[1, 1], [1, 1], [1, 1]],
        help="List of coupled qubit pairs, e.g. '[[0,1],[0,2],[1,4],[1,5]]' ",
    )
    parser.add_argument(
        "--device",
        default="numpy",
        type=str,
        help="Device to use (e.g., 'sinq20' or 'numpy' for local simulation)",
    )
    parser.add_argument(
        "--nshots",
        default=500,
        type=int,
        help="Number of shots for each circuit execution",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input.json",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--number-datapoints-output",
        type=int,
        default=100000,
        help="Number of data points to output",
    )
    cnf = vars(parser.parse_args())

    print(f"DEBUG: Received qubits_list: {cnf['qubits_list']}")
    print(f"DEBUG: Type: {type(cnf['qubits_list'])}, Length: {len(cnf['qubits_list'])}")
    qubits_to_use = build_chain_from_edges(cnf["qubits_list"])
    print(f"Using qubits: {qubits_to_use}")


    main(
        qubits_list=qubits_to_use,
        device=cnf["device"],
        nshots=cnf["nshots"], 
        debug=cnf.get("debug", False), 
        args=None, 
        input_filename=cnf.get("input"), 
        number_of_datapoints_output=cnf.get("number_datapoints_output")
    )