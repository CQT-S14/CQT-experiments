from pathlib import Path
from git.repo.base import Repo
import logging
import configparser
import json
import networkx as nx

# Repository root (two levels above any script in scripts/<name>/main.py)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

CURRENT_CALIBRATION_DIRECTORY = "/mnt/scratch/qibolab_platforms_nqch"
RUN_ID_FILE = REPO_ROOT / "current_run_id.json"


def find_all_chains(pairs):
    """Given the list of coupled qubits, output all possible chains of coupled qubits (excluding coupled qubits).
        
        Args:
            pairs (list[list[int]]): List of coupled qubits, e.g. [[0, 1], [1, 2], [1, 3]]
        Returns:
            chains (list[list[int]]): List of all possible chains of coupled qubits, e.g. [[0, 1, 2], [0, 1, 3]]
    """
    G = nx.Graph()
    G.add_edges_from(pairs)
    chains = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        nodes = list(subgraph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for path in nx.all_simple_paths(subgraph, nodes[i], nodes[j]):
                    if len(path) > 2:
                        chains.append(path)
    return chains


def find_longest_chain(pairs):
    """Given the list of coupled qubits, first find all possible chains of coupled qubits using `find_all_chains`.
        Then, output the longest chain.

        Args:
            pairs (list[list[int]]): List of coupled qubits, e.g. [[0, 1], [1, 2], [1, 3]]
        Returns:
            chain_of_qubits list[int]: List containing the longest chain.
    """
    chains = find_all_chains(pairs)
    
    max_length = 0 # placeholder
    for ii in range(len(chains)):
        chain = chains[ii]
        if len(chain) > max_length:
            max_length = len(chain)
            idx = ii

    chain_of_qubits = find_all_chains(pairs)[idx]
    return chain_of_qubits


def build_chain_from_edges(edge_list):
    """
    Build a chain of qubits from edge pairs, preserving connectivity order.
    
    Examples:
        [[b, a], [b, c]] -> [a, b, c]
        [[a, c], [b, c]] -> [a, c, b]
        [[x, y], [y, z]] -> [x, y, z]
    
    Args:
        edge_list (list[list[int]]): List of edge pairs, e.g. [[a, b], [b, c]]
    
    Returns:
        list[int]: Chain of qubits preserving connectivity order
    """
    if len(edge_list) == 0:
        return []
    elif len(edge_list) == 1:
        return list(edge_list[0])
    
    # Check if input is already a flat list of qubits (not edges)
    # If all elements are integers, return as-is
    if all(isinstance(item, int) for item in edge_list):
        return list(edge_list)
    
    # Count occurrences of each qubit to find endpoints
    qubit_count = {}
    for edge in edge_list:
        for q in edge:
            qubit_count[q] = qubit_count.get(q, 0) + 1
    
    # Find an endpoint (qubit that appears in only one edge)
    endpoint = None
    for q, count in qubit_count.items():
        if count == 1:
            endpoint = q
            break
    
    # Build chain
    if endpoint:
        # Start from endpoint
        for edge in edge_list:
            if endpoint in edge:
                if edge[0] == endpoint:
                    chain = list(edge)
                else:
                    chain = list(reversed(edge))
                break
        remaining_edges = [e for e in edge_list if endpoint not in e]
    else:
        # No endpoint, start with first edge
        chain = list(edge_list[0])
        remaining_edges = edge_list[1:]
    
    # Connect remaining edges
    while remaining_edges:
        last_qubit = chain[-1]
        found = False
        for i, edge in enumerate(remaining_edges):
            if edge[0] == last_qubit:
                chain.append(edge[1])
                remaining_edges.pop(i)
                found = True
                break
            elif edge[1] == last_qubit:
                chain.append(edge[0])
                remaining_edges.pop(i)
                found = True
                break
        if not found:
            break
    
    return chain


def output_dir_for(script_file: str, device: str | Path) -> Path:
    """Return data/<script-dir-name>/ for the given script file."""
    script_path = Path(script_file).resolve()

    # Load run_id from file
    with open(RUN_ID_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    run_id = data.get("run_id")
    if not run_id:
        raise ValueError("run_id missing from RUN_ID_FILE")
    run_id = str(run_id)

    if device == "numpy":
        hash_id = "numpy"    
    else:
        repo = Repo(CURRENT_CALIBRATION_DIRECTORY)
        hash_id = repo.commit().hexsha

    output_dir = DATA_DIR  / hash_id / run_id / script_path.parent.name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_experiment_list(config_file=None, logger=None):
    if config_file is None:
        config_file = REPO_ROOT / "pipeline" / "experiment_list.ini"
    """
    Load experiment list from an INI configuration file.

    Args:
        config_file (str): Path to the experiment list INI configuration file

    Returns:
        dict: Dictionary with sections as keys and lists of experiments as values
    """
    experiments = {}
    try:
        config = configparser.ConfigParser()
        config.read(config_file)

        for section in config.sections():
            experiments[section] = []
            for key, value in config[section].items():
                # Only include experiments that are enabled (not commented out)
                if not key.startswith("#") and value.lower() in [
                    "enabled",
                    "true",
                    "1",
                ]:
                    experiments[section].append(key)

    except Exception as e:
        logger.error(f"Error reading experiment list from '{config_file}': {e}")
        return {}
    return experiments