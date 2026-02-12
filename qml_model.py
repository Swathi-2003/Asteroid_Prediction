'''# qml_model.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC, PegasosQSVC
import numpy as np


def run_vqc(X_train, X_test, y_train, y_test):
    print("\n🚀 Running Variational Quantum Classifier (VQC)")

    # Use fewer features for speed
    X_train, X_test = X_train[:, :2], X_test[:, :2]

    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    backend = AerSimulator(shots=64)
    optimizer = COBYLA(maxiter=15)

    # Modern VQC – no one_hot arg
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=backend,
    )

    # y must be numpy arrays
    y_train, y_test = np.array(y_train), np.array(y_test)

    vqc.fit(X_train, y_train)
    score = vqc.score(X_test, y_test)
    print(f"✅ VQC Accuracy: {score:.3f}")
    return score


def run_qsvc(X_train, X_test, y_train, y_test):
    print("\n🚀 Running Quantum Support Vector Classifier (QSVC)")

    X_train, X_test = X_train[:, :2], X_test[:, :2]
    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    backend = AerSimulator(shots=64)

    qsvc = PegasosQSVC(
        feature_map=feature_map,
        C=1.0,
        quantum_instance=backend,
        num_steps=15,
    )

    y_train, y_test = np.array(y_train), np.array(y_test)

    qsvc.fit(X_train, y_train)
    score = qsvc.score(X_test, y_test)
    print(f"✅ QSVC Accuracy: {score:.3f}")
    return score

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

# ---------------------------------------------------------------------
def run_vqc(X_train, X_test, y_train, y_test):
    print("\n🚀 Running Variational Quantum Classifier (VQC)")

    # Limit to 4 qubits and fewer samples for speed
    X_train, y_train = X_train[:200, :4], y_train[:200]
    X_test, y_test = X_test[:60, :4], y_test[:60]

    feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
    ansatz = RealAmplitudes(num_qubits=4, reps=1)
    optimizer = COBYLA(maxiter=15)
    backend = AerSimulator(method="statevector")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=backend,
    )

    vqc.fit(X_train, y_train)
    acc = vqc.score(X_test, y_test)
    print(f"✅ VQC Accuracy: {acc:.3f}")
    return acc


# ---------------------------------------------------------------------
def run_qsvc(X_train, X_test, y_train, y_test):
    print("\n🚀 Running Quantum Support Vector Classifier (QSVC)")

    X_train, y_train = X_train[:200, :4], y_train[:200]
    X_test, y_test = X_test[:60, :4], y_test[:60]

    feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
    backend = AerSimulator(method="statevector")
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

    qsvc = PegasosQSVC(quantum_kernel=qkernel, num_steps=20)
    qsvc.fit(X_train, y_train)
    acc = qsvc.score(X_test, y_test)
    print(f"✅ QSVC Accuracy: {acc:.3f}")
    return acc
'''
''' correct code
# qml_model.py
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Qiskit imports
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel

# BlueQubit import (optional)
USE_BLUEQUBIT = False  # <---- set to True to run on BlueQubit cloud (requires token)
BLUEQUBIT_API_ENV = "BLUEQUBIT_API_TOKEN"  # make sure you set this env var if using cloud

if USE_BLUEQUBIT:
    try:
        from bluequbit import BQClient
    except Exception as e:
        raise RuntimeError("bluequbit SDK not installed or import failed: " + str(e))

def get_quantum_instance(use_cloud=USE_BLUEQUBIT):
    if use_cloud:
        token = os.getenv(BLUEQUBIT_API_ENV)
        if not token:
            raise RuntimeError(f"Set {BLUEQUBIT_API_ENV} environment variable for BlueQubit.")
        bq = BQClient(api_token=token, execution_mode="cloud")
        return bq
    else:
        # Use fast local statevector simulator
        return Aer.get_backend("aer_simulator_statevector")

# ----------------------------
# VQC
# ----------------------------
def run_vqc(X_train, X_test, y_train, y_test,
            qubits=4, reps=3, maxiter=75, use_cloud=USE_BLUEQUBIT,
            train_sub=400, test_sub=120):
    """
    Run VQC (paper-like) with 4 features/qubits.
    For speed, we use subsets of data (train_sub/test_sub).
    """
    print("\n🚀 Running Variational Quantum Classifier (VQC)")

    # Ensure numpy arrays
    X_train = np.array(X_train); X_test = np.array(X_test)
    y_train = np.array(y_train); y_test = np.array(y_test)

    # subset for speed
    X_tr = X_train[:train_sub, :qubits]
    y_tr = y_train[:train_sub]
    X_te = X_test[:test_sub, :qubits]
    y_te = y_test[:test_sub]

    feature_map = ZZFeatureMap(feature_dimension=qubits, reps=1)
    # ansatz choice: use RealAmplitudes (paper used RealAmplitudes + EfficientSU2 as two ansatze)
    ansatz = RealAmplitudes(num_qubits=qubits, reps=reps)

    optimizer = COBYLA(maxiter=maxiter)
    quantum_instance = get_quantum_instance(use_cloud)

    # Build VQC (modern qiskit VQC expects quantum_instance that can run circuits)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=quantum_instance,
    )

    vqc.fit(X_tr, y_tr)
    acc = vqc.score(X_te, y_te)
    print(f"✅ VQC Accuracy: {acc:.4f}")
    return acc

# ----------------------------
# PQSVC (Pegasos Quantum SVC)
# ----------------------------
def run_pqsvc(X_train, X_test, y_train, y_test,
              qubits=4, reps=1, num_steps=100, C=100.0, use_cloud=USE_BLUEQUBIT,
              train_sub=400, test_sub=120):
    """
    Run PegasosQSVC using a QuantumKernel constructed from a ZZFeatureMap.
    """
    print("\n🚀 Running Pegasos Quantum SVC (PQSVC)")

    X_train = np.array(X_train); X_test = np.array(X_test)
    y_train = np.array(y_train); y_test = np.array(y_test)

    X_tr = X_train[:train_sub, :qubits]
    y_tr = y_train[:train_sub]
    X_te = X_test[:test_sub, :qubits]
    y_te = y_test[:test_sub]

    feature_map = ZZFeatureMap(feature_dimension=qubits, reps=reps)
    quantum_instance = get_quantum_instance(use_cloud)
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=num_steps)
    qsvc.fit(X_tr, y_tr)
    acc = qsvc.score(X_te, y_te)
    print(f"✅ PQSVC Accuracy: {acc:.4f}")
    return acc
'''
# qml_model.py
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Qiskit imports
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel

# BlueQubit (optional)
USE_BLUEQUBIT = False  # set True to use BlueQubit cloud
BLUEQUBIT_API_ENV = "BLUEQUBIT_API_TOKEN"

if USE_BLUEQUBIT:
    from bluequbit import BQClient


def get_quantum_instance(use_cloud=USE_BLUEQUBIT):
    if use_cloud:
        token = os.getenv(BLUEQUBIT_API_ENV)
        if not token:
            raise RuntimeError(f"Set {BLUEQUBIT_API_ENV} environment variable for BlueQubit.")
        bq = BQClient(api_token=token, execution_mode="cloud")
        return bq
    else:
        return Aer.get_backend("aer_simulator_statevector")


# --------------------------------------------------------------------
# Run VQC with two ansatz options
# --------------------------------------------------------------------
def run_vqc(X_train, X_test, y_train, y_test,
            qubits=4, reps=3, maxiter=50, use_cloud=USE_BLUEQUBIT,
            train_sub=300, test_sub=100):
    print("\n🚀 Running Variational Quantum Classifier (VQC)")

    X_train = np.array(X_train)[:, :qubits]
    X_test = np.array(X_test)[:, :qubits]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train[:train_sub]
    y_train = y_train[:train_sub]
    X_test = X_test[:test_sub]
    y_test = y_test[:test_sub]

    feature_map = ZZFeatureMap(feature_dimension=qubits, reps=1)
    backend = get_quantum_instance(use_cloud)
    optimizer = COBYLA(maxiter=maxiter)

    vqc_results = {}

    for ansatz_name, ansatz in {
        "RealAmplitudes": RealAmplitudes(num_qubits=qubits, reps=reps),
        "EfficientSU2": EfficientSU2(num_qubits=qubits, reps=reps)
    }.items():
        print(f"\n⚛️ VQC with ansatz: {ansatz_name}")
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=backend,
        )

        vqc.fit(X_train, y_train)
        preds = vqc.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        vqc_results[ansatz_name] = (acc, prec, rec, f1)
        print(f"✅ {ansatz_name} -> Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}")

    return vqc_results


# --------------------------------------------------------------------
# Run PQSVC for multiple C values + metrics + confusion matrix
# --------------------------------------------------------------------
def run_pqsvc(X_train, X_test, y_train, y_test,
              qubits=4, reps=1, num_steps=100, use_cloud=USE_BLUEQUBIT,
              C_values=(0.1, 1, 10, 100, 1000),
              train_sub=300, test_sub=100):
    print("\n🚀 Running Pegasos Quantum SVC (PQSVC)")

    X_train = np.array(X_train)[:, :qubits]
    X_test = np.array(X_test)[:, :qubits]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train[:train_sub]
    y_train = y_train[:train_sub]
    X_test = X_test[:test_sub]
    y_test = y_test[:test_sub]

    feature_map = ZZFeatureMap(feature_dimension=qubits, reps=reps)
    backend = get_quantum_instance(use_cloud)
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

    pqsvc_results = {}
    conf_mats = {}

    for C in C_values:
        print(f"\n⚛️ PQSVC with C={C}")
        qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=num_steps)
        qsvc.fit(X_train, y_train)
        preds = qsvc.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        pqsvc_results[C] = (acc, prec, rec, f1)
        conf_mats[C] = cm

        print(f"✅ C={C}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
        print("Confusion Matrix:\n", cm)

    return pqsvc_results, conf_mats

