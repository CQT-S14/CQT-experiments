import os
import argparse
import pathlib
import json
import pickle
from timeit import time
import tqdm
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from qibo.config import log
from qibo import Circuit, gates, construct_backend, set_backend
from qibo.transpiler import NativeGates, Passes, Unroller
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z

from qiboml.models.encoding import PhaseEncoding, QuantumEncoding
from qiboml.models.decoding import Expectation
from qiboml.interfaces.pytorch import QuantumModel
from qiboml.differentiations.psr import PSR
from qiboml import ndarray

'''
from plots import plot_reuploading_classifier
from datetime import date
'''

os.environ["QIBOLAB_PLATFORMS"] = pathlib.Path(
    "/mnt/scratch/qibolab_platforms_nqch"
).as_posix()

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config  # scripts/config.py

# Prepare the training dataset
def _circle(points):
    labels = np.zeros(len(points), dtype=np.int32)
    ids = np.where(np.linalg.norm(points, axis=1) > np.sqrt(2 / np.pi))
    labels[ids] = 1

    return points, labels


def create_dataset(grid=None, samples=1000, seed=0):
    """Function to create training and test sets for classifying.

    Args:
        name (str): Name of the problem to create the dataset, to choose between
            ['circle', '3 circles', 'square', '4 squares', 'crown', 'tricrown', 'wavy lines'].
        grid (int): Number of points in one direction defining the grid of points.
            If not specified, the dataset does not follow a regular grid.
        samples (int): Number of points in the set, randomly located.
            This argument is ignored if grid is specified.
        seed (int): Random seed

    Returns:
        Dataset for the given problem (x, y)
    """
    if grid is None:
        np.random.seed(seed)
        points = 1 - 2 * np.random.rand(samples, 2)
    else:
        x = np.linspace(-1, 1, grid)
        points = np.array(list(product(x, x)))
    creator = _circle

    x, y = creator(points)
    return x, y


# Trivial trainable layer
def trainable_circuit(nqubits, entanglement=True, density_matrix=False):
    """
    Construct a trainable quantum circuit where the amount of entanglement can be tuned
    if the argument `entanglement` is set equal to `True`."""
    circ = Circuit(nqubits, density_matrix=density_matrix)

    for q in range(nqubits):
        circ.add(gates.RZ(q=q, theta=0.0))
    return circ


def predict(model, data):
    test_pred = torch.as_tensor([torch.sigmoid(model(x)) for x in data])
    test_pred = test_pred.to("cpu")
    test_pred_int = torch.round(test_pred)

    return test_pred_int.tolist()


def compute_accuracy(labels, predictions, tolerance=1e-2):
    """
    Args:
        labels: numpy.array with the labels of the quantum states to be classified.
        predictions: numpy.array with the predictions for the quantum states classified.
        tolerance: float tolerance level to consider a prediction correct (default=1e-2).
    Returns:
        float with the proportion of states classified successfully.
    """
    accur = 0
    error_list = []
    #for l, p in zip(labels, predictions):
    for ind, (l, p) in enumerate(zip(labels, predictions)):
        if np.allclose(l, p, rtol=0.0, atol=tolerance):
            accur += 1
        else:
            error_list.append(ind)

    accur = accur / len(labels)

    return accur, error_list


# RxRy Encoding
@dataclass(eq=False)
class RYRZEncoding(QuantumEncoding):
    nlayers = 1

    def update_nlayers(self, nlayers):
        self.nlayers = nlayers

    def __post_init__(
        self,
    ):
        """Ancillary post initialization: builds the internal circuit with the rotation gates."""
        super().__post_init__()

    @cached_property
    def _data_to_gate(self):
        """
        Mapping between the index of the input and the indices of the gates in the
        produced encoding circuit queue, where the input is encoded to.
        For instance, {0: [0,2], 1: [2]}, represents an encoding where the element
        0 of the inputs enters the gates with indices 0 and 2 of the queue, whereas
        the element 1 of the input affects only the the gate in position 2 of the
        queue.
        By deafult, the map reproduces a simple encoding where the
        i-th component of the data is uploaded in the i-th gate of the queue.
        """
        return {f"{i}": [i] for i in range(len(self.qubits) * 2 * self.nlayers)}

    def __call__(self, x: ndarray) -> Circuit:
        """Construct the circuit encoding the ``x`` data in the chosen encoding gate.

        Args:
            x (ndarray): the input real data to encode in rotation angles.

        Returns:
            (Circuit): the constructed ``qibo.Circuit``.
        """
        circuit = self.circuit
        x = x.ravel()
        nq = self.qubits
        for l in range(self.nlayers):
            for i, q in enumerate(nq):
                for j, gate in enumerate([gates.RY, gates.RZ]):
                    this_gate_params = {"trainable": False}
                    this_gate_params.update({"theta": x[2 * 1 * l + 2 * i + j]})
                    circuit.add(gate(q=q, **this_gate_params))
        return circuit


# Linear layer
def make_linear_layer(nlayers=1):
    n_in_features = 2
    n_out_features_per_layer = 2
    n_out_features = nlayers * n_out_features_per_layer
    linear_layer = nn.Linear(n_in_features, n_out_features)

    for j in range(n_out_features):
        i = int(np.mod(j + 1, 2))
        linear_layer.weight.data[j, i] = 0.0

    return linear_layer


# Mask tensor
def make_mask_tensor(nlayers=1):
    n_in_features = 2
    n_out_features_per_layer = 2
    n_out_features = nlayers * n_out_features_per_layer
    mask_tensor = torch.tensor(
        [
            [
                0.0 if j == int(np.mod(i + 1, n_in_features)) else 1.0
                for j in range(n_in_features)
            ]
            for i in range(n_out_features)
        ]
    )
    mask_tensor.requires_grad = False

    return mask_tensor


# Encoder
class LinearEncoder(nn.Module):
    def __init__(self, nlayers=1):
        super().__init__()
        self.linear_layer = make_linear_layer(nlayers)
        self.mask_tensor = make_mask_tensor(nlayers)

    def forward(self, x):
        masked_linear_layer = self.linear_layer
        masked_linear_layer.weight.data.mul_(self.mask_tensor)
        x = masked_linear_layer(x)
        return x


# Load trained weights
def load_trained_data(file_path):
    with open(file_path, "rb") as file:
        trained_data = pickle.load(file)

    trained_weights = trained_data["circle"][10][::2]
    trained_biases = trained_data["circle"][10][1::2]
    trained_weights = np.array(
        [np.diag(pair) for pair in trained_weights.reshape([10, 2])]
    )
    trained_weights = trained_weights.reshape([20, 2])
    trained_weights = torch.tensor(trained_weights, requires_grad=False)
    trained_biases = torch.tensor(trained_biases, requires_grad=False)
    return trained_weights, trained_biases


# Trained Encoder (nlayers=10 only)
class TrainedLinearEncoder(nn.Module):
    def __init__(self, file_path):
        super().__init__()
        trained_weights, trained_biases = load_trained_data(file_path)
        self.linear_layer = make_linear_layer(nlayers=10)
        self.linear_layer.weight.data = trained_weights
        self.linear_layer.bias.data = trained_biases

    def forward(self, x):
        x = self.linear_layer(x)
        return x


def main(
    device,
    qubit_id,
    nlayers,
    lr,
    epochs,
    nshots,
    ##grid,
    num_train_samples,
    num_test_samples,
    seed,
    gpu,
    training,
):
    nqubits = 1
    torch_device = torch.device("cpu")
    script_directory = os.path.dirname(__file__)

    # Set up device (backend)
    if device == "numpy":
        backend = construct_backend("numpy")
    elif device == "nqch-sim":
        backend = construct_backend("numpy")  # placeholder
    elif device == "sinq20":
        backend = construct_backend("qibolab", platform="sinq20")
    elif device == "qiboml":
        backend = construct_backend("qiboml", platform="pytorch")
        set_backend(backend="qiboml", platform="pytorch")
        if gpu:
            torch_device = torch.device(f"cuda:{gpu}")
    torch.set_default_device(torch_device)

    # Generate training data
    ##training_set = create_dataset(grid=grid)
    training_set = create_dataset(samples=num_train_samples, seed=1)
    test_set = create_dataset(samples=num_test_samples, seed=0)
    x_train = torch.tensor(training_set[0])
    y_train = torch.tensor(
        training_set[1].reshape([training_set[1].shape[0], 1]), dtype=torch.float64
    )
    x_test = torch.tensor(test_set[0])
    y_test = torch.tensor(
        test_set[1].reshape([test_set[1].shape[0], 1]), dtype=torch.float64
    )

    # Set up transpiler
    glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    natives = NativeGates(0).from_gatelist(glist)
    custom_passes = [Unroller(native_gates=natives)]
    custom_pipeline = Passes(custom_passes)
    transpiler = custom_pipeline

    # Set up directories
    backend_name = backend.name.lower().replace(" ", "_")
    output_dir = config.output_dir_for(__file__, device)
    params_dir = os.path.join(output_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up classical preprocessing
    if not training:
        file_path = f"{script_directory}/input_parameters.pkl"
        linear_encoder = TrainedLinearEncoder(file_path).double()
        nlayers = 10
        lr = None
        epochs = None
        seed = None
    else:
        linear_encoder = LinearEncoder(nlayers=nlayers).double()

    # Build circuits
    encoding = RYRZEncoding(nqubits=nqubits, density_matrix=False)
    encoding.update_nlayers(nlayers)
    circuit_structure = []

    circuit_structure.extend([encoding])
    circuit_structure.extend(
        [trainable_circuit(nqubits, entanglement=False, density_matrix=False)]
    )

    observable = SymbolicHamiltonian(-Z(0), nqubits=nqubits)

    decoding_circ = Expectation(
        nqubits=nqubits,
        observable=observable,
        nshots=nshots,
        backend=backend,
        transpiler=transpiler,
        mitigation_config=None,
        noise_model=None,
        density_matrix=False,
        wire_names=[qubit_id],
    )
    # Quantum model
    q_model_kwargs = dict(circuit_structure=circuit_structure, decoding=decoding_circ)
    q_model_kwargs["differentiation"] = PSR()
    q_model = QuantumModel(**q_model_kwargs)

    # Full Model
    model = nn.Sequential(linear_encoder, q_model)

    duration = None
    loss_history = []
    final_loss = None

    if training:
        # Optimizer & loss
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = F.binary_cross_entropy_with_logits

        # Training loop
        mask_tensor = make_mask_tensor(nlayers=nlayers)
        time_start = time.time()

        for epoch in range(epochs):
            permutation = torch.randperm(len(x_train))
            optimizer.zero_grad()
            y_pred = torch.stack([model(x) for x in x_train[permutation]]).squeeze(-1)
            loss = criterion(y_pred, y_train[permutation])
            loss.backward()
            model[0].linear_layer.weight.grad.data.mul_(mask_tensor)
            model[-1].circuit_parameters.grad = torch.tensor([0.0], dtype=torch.double)
            optimizer.step()

            loss_history.append(loss.item())
            torch.save(
                model.state_dict(), os.path.join(params_dir, f"epoch_{epoch:03d}.pt")
            )

            log.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        time_end = time.time()
        duration = time_end - time_start
        final_loss = loss_history[-1]

    # Compute train/test predictions and train/test accuracies
    predict_train_start = time.time()
    train_preds = predict(model, x_train)
    predict_train_end = time.time()
    predict_train_duration = predict_train_end - predict_train_start

    predict_test_start = time.time()
    test_preds = predict(model, x_test)
    predict_test_end = time.time()
    predict_test_duration = predict_test_end - predict_test_start

    train_acc, train_pred_errors = compute_accuracy(y_train, train_preds)
    test_acc, test_pred_errors = compute_accuracy(y_test, test_preds)

    # Generate results dictionary and save results and metadata to json files
    report_data = {
        "number_of_gates": nlayers*2*nqubits,
        "depth": nlayers*2,
        "nshots": nshots,
        "final_loss": final_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_pred_errors": train_pred_errors,
        "test_pred_errors": test_pred_errors,
        "runtime": duration,
        "x_train": x_train.detach().numpy().tolist(),
        "train_predictions": train_preds,
        "x_test": x_test.detach().numpy().tolist(),
        "test_predictions": test_preds,
        "loss_history": loss_history,
        "final_weights": model[0].linear_layer.weight.detach().numpy().tolist(),
        "final_biases": model[0].linear_layer.bias.detach().numpy().tolist(),
        "final_RZ_angle_check": model[-1].circuit_parameters.detach().numpy().tolist(),
        "predict_train_duration": predict_train_duration,
        "predict_test_duration": predict_test_duration,
        "qubits_used": [qubit_id],
        "description": f"Reuploading classifier with {nqubits} qubits, {nlayers} layers, depth of {nlayers*2}, {nshots} shots."
    }

    static_meta_data = {
        "backend": backend_name,
        "qubit_id": qubit_id,
        "nlayers": nlayers,
        "lr": lr,
        "epochs": epochs,
        "nshots": nshots,
        ##"grid": grid,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "seed": seed,
        "gpu": gpu,
        "training": training,
    }

    with open(os.path.join(output_dir, f"results.json"), "w") as file:
        json.dump(report_data, file, indent=4)
    with open(os.path.join(output_dir, f"settings.json"), "w") as file:
        json.dump(static_meta_data, file, indent=4)


    '''
    # Load from results.json and generate plots
    raw_data = os.path.join(output_dir, f"results.json")
    plot_reuploading_classifier(raw_data, exp_name=date.today(), output_path=output_dir)
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small quantum model on a classification task for the circle dataset."
    )
    parser.add_argument(
        "--device",
        choices=["numpy", "nqch-sim", "sinq20", "qiboml"],
        default="numpy",
        help="Device to use (default: numpy)",
    )
    # parser.add_argument(
    #   "--nqubits", type=int, default=1, help="Number of qubits (default: 1)"
    # )
    parser.add_argument("--qubit_id", type=int, default=9, help="Qubit ID (default: 9)")
    parser.add_argument(
        "--nlayers", type=int, default=6, help="Number of layers (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
        # "--epochs", type=int, default=10, help="Number of epochs (test: 10)"
    )
    parser.add_argument(
        "--nshots", type=int, default=500, help="Number of shots (default: 500)"
    )
    ##parser.add_argument(
     ##   "--grid",
     ##   type=int,
     ##   default=11,
     ##   help="Number of grid points along both paramter directions for generating training dataset (default: 11)",
    ##)
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=100,
        help="Number of train samples (default: 100)",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=100,
        help="Number of test samples (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=48,
        help="Random seed value for initializing trainable circuit parameters (default: 48)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Option to run job on GPU when backend is qiboml. Enter GPU cuda ID (default: None)",
    )
    parser.add_argument(
        "--training", 
        action="store_true", 
        help="Trigger training=True to perform training on model, otherwise load trained weights into model",
    )
    args = vars(parser.parse_args())
    #print(args)
    main(**args)
