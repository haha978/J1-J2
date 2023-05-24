import sys
sys.path.insert(0, "../")
import qiskit
from qiskit import QuantumCircuit, Aer
import numpy as np
from noiseless.utils import get_nearest_neighbors, flatten_neighbor_l

def ALA(circ, N_qubits, var_params, n_layers):
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
    return circ

def get_nn_dict(m, n):
    nn_l = flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    dict_idx = 0
    nn_dict = {}
    while len(nn_l) > 0:
        seen_l = nn_l.pop(0)
        nn_dict[dict_idx] = [seen_l.copy()]
        new_nn_l = []
        for nn in nn_l:
            q1, q2 = nn
            if q1 in seen_l or q2 in seen_l:
                new_nn_l.append(nn)
            else:
                nn_dict[dict_idx].append(nn)
                seen_l.append(q1)
                seen_l.append(q2)
        nn_l = new_nn_l
        dict_idx += 1
    return nn_dict

def HVA(circ, m, n, var_params, n_layers):
    #NEED SOME CODE HERE
    param_idx = 0
    N_qubits = m * n
    nn_dict = get_nn_dict(m, n)
    for i in range(N_qubits):
        circ.h(i)
    for _ in range(n_layers):
        for i in range(N_qubits):
            circ.rx(var_params[param_idx], i)
            param_idx += 1
        for k in nn_dict:
            for nn in nn_dict[k]:
                q1, q2 = nn
                circ.rzz(var_params[param_idx], q1, q2)
                param_idx += 1
    return circ

def Q_Circuit(m, n, var_params, n_layers, ansatz_type):
    N_qubits = m * n
    circ = QuantumCircuit(N_qubits, N_qubits)
    if ansatz_type == "ALA":
        N_qubits = m*n
        return ALA(circ, N_qubits, var_params, n_layers)
    elif ansatz_type == "HVA":
        return HVA(circ, m, n, var_params, n_layers)
    else:
        raise ValueError("No available ansatz")
