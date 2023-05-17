import qiskit
from qiskit import QuantumCircuit, Aer
from utils import get_nearest_neighbors, flatten_neighbor_l, get_next_nearest_neighbors
import numpy as np

def get_nn_dict(m, n, next_nn = False):
    if next_nn == True:
        nn_l = flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)
    else:
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

def ALA(circ, N_qubits, var_params, h_l, n_layers):
    param_idx = 0
    for i in range(N_qubits):
        circ.h(i)
    assert N_qubits == 9, "number of qubits is 9"
    m, n = 3, 3
    nn_dict = get_nn_dict(m, n, next_nn = False)
    next_nn_dict = get_nn_dict(m, n, next_nn = True)
    print(nn_dict)
    print(next_nn_dict)
    for _ in range(n_layers):
        for i in range(N_qubits):
            circ.ry(var_params[param_idx], i)
            param_idx += 1
        for k in nn_dict:
            for nn in nn_dict[k]:
                q1, q2 = nn
                circ.cx(q1, q2)
        for i in range(N_qubits):
            circ.ry(var_params[param_idx], i)
            param_idx += 1
        for next_k in next_nn_dict:
            for next_nn in next_nn_dict[next_k]:
                q1, q2 = next_nn
                circ.cx(q1, q2)
    for h_idx in h_l:
        circ.h(h_idx)
    return circ


def HVA(circ, m, n, var_params, h_l, n_layers):
    #NEED SOME CODE HERE
    param_idx = 0
    N_qubits = m * n
    nn_dict = get_nn_dict(m, n, next_nn = False)
    next_nn_dict = get_nn_dict(m, n, next_nn = True)
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
        for next_k in next_nn_dict:
            for next_nn in next_nn_dict[next_k]:
                q1, q2 = next_nn
                circ.rzz(var_params[param_idx], q1, q2)
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
if __name__ == "__main__":
    Nparams = 2*(9+9)
    param = np.random.uniform( low = -np.pi, high = np.pi, size =Nparams)
    print(Q_Circuit(3, 3, param, [], 2, "ALA"))
