import os
import argparse
import pathlib
import json
from timeit import time


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from qibo.config import log
from qibo import Circuit, gates, construct_backend
from qibo.transpiler import NativeGates, Passes, Unroller

from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.operations.differentiation import PSR

os.environ.setdefault("QIBOLAB_PLATFORMS", "/mnt/scratch/qibolab_platforms_nqch")

from pathlib import Path as _P
import sys

sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py


# Prepare the training dataset
def f(x):
    return 1 * torch.sin(x) ** 2 - 0.3 * torch.cos(x)


# Trainable layer
def trainable_circuit(nqubits, entanglement=True, density_matrix=False):
    """
    Construct a trainable quantum circuit where the amount of entanglement can be tuned
    if the argument `entanglement` is set equal to `True`."""
    circ = Circuit(nqubits, density_matrix=density_matrix)
    for q in range(nqubits):
        circ.add(gates.RY(q=q, theta=np.random.randn()))
        circ.add(gates.RZ(q=q, theta=np.random.randn()))
    if nqubits > 1 and entanglement:
        for q in range(nqubits):
            circ.add(gates.CRX(q, (q + 1) % nqubits, theta=np.random.randn()))
    return circ


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small quantum model on a sin^2 function."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="numpy",
        choices=["numpy", "sinq20"],
        help="Device to use: numpy, nqch-sim, or sinq20 (default: numpy)",
    )
    parser.add_argument(
        "--nqubits", type=int, default=1, help="Number of qubits (default: 1)"
    )
    parser.add_argument("--qubit_id", type=int, default=9, help="Qubit ID (default: 9)")
    parser.add_argument(
        "--nlayers", type=int, default=3, help="Number of layers (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.25, help="Learning rate (default: 0.25)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--nshots", type=int, default=500, help="Number of shots (default: 500)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="Number of training samples (default: 15)",
    )
    args = parser.parse_args()

    backend_str = args.device

    # Generate training data
    x_train = torch.linspace(
        0, 2 * np.pi, args.num_samples, dtype=torch.float64
    ).unsqueeze(1)
    y_train = f(x_train)
    y_train = 2 * (((y_train - y_train.min()) / (y_train.max() - y_train.min())) - 0.5)

    # Set up backend
    # backend = construct_backend(backend_str)
    if backend_str == "numpy":
        backend = construct_backend("numpy")
    elif backend_str == "nqch-sim":
        backend = construct_backend("numpy")  # placeholder
    elif backend_str == "sinq20":
        backend = construct_backend("qibolab", platform="sinq20")

    # Set up transpiler
    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)
    transpiler = custom_pipeline

    # Set up directories
    backend_name = backend.name.lower().replace(" ", "_")

    results_dir = os.path.join("data", "universal_approximant/")
    # Write to data/<scriptname>/<device>/results.json
    results_dir = config.output_dir_for(__file__, args.device)
    results_dir.mkdir(parents=True, exist_ok=True)

    params_dir = os.path.join(results_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # Build circuits
    encoding_circ = PhaseEncoding(
        nqubits=args.nqubits, encoding_gate=gates.RX, density_matrix=False
    )
    circuit_structure = []
    for _ in range(args.nlayers):
        circuit_structure.extend(
            [
                encoding_circ,
                trainable_circuit(
                    args.nqubits, entanglement=False, density_matrix=False
                ),
            ]
        )

    decoding_circ = Expectation(
        nqubits=args.nqubits,
        nshots=args.nshots,
        backend=backend,
        transpiler=transpiler,
        # mitigation_config=None,
        # noise_model=None,
        # density_matrix=False,
        # wire_names=[args.qubit_id],
    )

    # Model
    model_kwargs = dict(circuit_structure=circuit_structure, decoding=decoding_circ)
    model_kwargs["differentiation"] = PSR()
    model = QuantumModel(**model_kwargs)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    loss_history = []
    plot_data = []  # To store data for all epochs
    time_start = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        y_pred = torch.stack([model(x) for x in x_train]).squeeze(-1)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        torch.save(
            model.state_dict(), os.path.join(params_dir, f"epoch_{epoch:03d}.pt")
        )

        log.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        # Collect data for JSON
        epoch_data = {
            "epoch": epoch,
            "x_train": x_train.detach().numpy().tolist(),
            "y_train": y_train.detach().numpy().tolist(),
            "predictions": y_pred.detach().numpy().tolist(),
            "loss": loss.item(),
        }
        plot_data.append(epoch_data)

    time_end = time.time()
    duration = time_end - time_start

    # Final predictions with uncertainty
    preds = np.stack(
        [
            torch.stack([model(x) for x in x_train]).squeeze(-1).detach().numpy()
            for _ in range(20)
        ],
        axis=0,
    )
    median_pred = np.median(preds, axis=0)
    mad_pred = np.median(np.abs(preds - median_pred[None, :]), axis=0)

    # Save all results to a single JSON file
    results = {
        "epoch_data": plot_data,
        "loss_history": loss_history,
        "median_predictions": median_pred.tolist(),
        "mad_predictions": mad_pred.tolist(),
        "runtime": f"{duration:.2f} seconds.",
        "qubits_used": [args.qubit_id],
        "description": f"Universal approximant. ",
    }

    with open(os.path.join(results_dir, "results.json"), "w") as json_file:
        json.dump(results, json_file, indent=4)
        torch.save(
            model.state_dict(), os.path.join(params_dir, f"epoch_{epoch:03d}.pt")
        )
    log.info(f"Results saved to {os.path.join(results_dir, 'results.json')}")
