import os
import pathlib
from qibo import Circuit, gates
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
import numpy as np
import importlib, sys

os.environ["QIBOLAB_PLATFORMS"] = "/mnt/scratch/qibolab_platforms_nqch"

for pkg in ["qibo", "qibocal", "qibolab"]:
    try:
        mod = importlib.import_module(pkg)
        print(f"{pkg}: version={mod.__version__}, location={mod.__file__}")
    except Exception as e:
        print(f"{pkg}: NOT FOUND ({e})")

print(f"\nPython: {sys.executable}")
print(f"sys.path: {sys.path}\n")

 
targets = (0, 1)
root_path = "tomo-test-results"
platform = "sinq20"
 
with Executor.open(
    path= root_path,
    platform= platform,
    targets=[targets],
    update=False, #True only if you need to persist the calibration data anything downstream
    force=True,
) as e:
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
 
    output = e.two_qubit_state_tomography(circuit = circuit, targets =[targets])
    report(e.path, e.history)