import qiskit
import qiskit_aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC, VQC

print("Qiskit version:", qiskit.__version__)
print("Aer version:", qiskit_aer.__version__)
