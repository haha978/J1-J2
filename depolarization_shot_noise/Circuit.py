import sys
sys.path.insert(0, "../")
import qiskit
from qiskit import QuantumCircuit, Aer
import numpy as np
from depolarization_shot_noise.utils import get_nearest_neighbors, flatten_neighbor_l

def ALA(circ, N_qubits, var_params, h_l, n_layers):
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    if N_qubits % 2 == 0:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    else:
        for layer in range(n_layers):
            if layer % 2 == 0:
                for i in range(0, N_qubits-1, 2):
                    circ.cx(i, i+1)
                for i in range(N_qubits-1):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
            else:
                for i in range(1, N_qubits, 2):
                    circ.cx(i, i+1)
                for i in range(1, N_qubits):
                    circ.ry(var_params[param_idx], i)
                    param_idx += 1
    for h_idx in h_l:
        circ.h(h_idx)
    return circ

def Q_Circuit(m, n, var_params, h_l, n_layers, ansatz_type):
    N_qubits = m * n
    circ = QuantumCircuit(N_qubits, N_qubits)
    if ansatz_type == "ALA":
        N_qubits = m*n
        return ALA(circ, N_qubits, var_params, h_l, n_layers)
    elif ansatz_type == "HVA":
        return HVA(circ, m, n, var_params, h_l, n_layers)
    else:
        raise ValueError("No available ansatz")
