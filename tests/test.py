import sys
import pathlib
from qiskit.providers.aer import Aer
from qiskit import transpile
import numpy as np
sys.path.insert(0, "../")
from noiseless.Circuit import Q_Circuit as Q_Circuit_noiseless
from shot_noise.Circuit import Q_Circuit as Q_Circuit_noise
from noiseless.utils import get_Hx, expected_op1_op2, expected_op, create_partial_Hamiltonian
from noiseless.HR_J1_J2 import get_params, get_operations_l
from noiseless.HR_J1_J2 import get_HR_distance as get_HR_distance_noiseless
from shot_noise.HR_J1_J2 import get_measurement_index_l
from shot_noise.utils import expectation_X, get_NN_coupling, get_nNN_coupling, get_exp_cross
from shot_noise.utils import flatten_neighbor_l, get_nearest_neighbors, get_next_nearest_neighbors
from shot_noise.utils import distanceVecFromSubspace

"""
Before Running the test, please read the comments below

1. Make sure to copy "get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend)""
function from shot_noise/HR_J1_J2.py file before running the test

2. Because "get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)"
function in shot_noise/HR_J1_J2.py saves its output, the get_measurement(...) function needs to be
copied to this script, excluding the part where it loads and saves the measurement.

3. Comment out in main to run the test of interest
"""
def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    num_shots = hyperparam_dict["shots"]
    backendnm = hyperparam_dict["backend"]
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    circ = Q_Circuit_noise(m, n, var_params, h_l, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"])
    circ.measure(list(range(n_qbts)), list(range(n_qbts)))
    circ = transpile(circ, backend)
    job = backend.run(circ, shots = num_shots)
    if backendnm != "aer_simulator":
        job_id = job.id()
        job_monitor(job)
    result = job.result()
    measurement = dict(result.get_counts())
    return measurement

def get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend):
    cov_mat = np.zeros((3,3))
    m, n = hyperparam_dict["m"],  hyperparam_dict["n"]
    n_qbts = m * n
    z_l, x_l = [], [i for i in range(n_qbts)]
    var_params = get_params(params_dir_path, param_idx)
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, param_idx)
    exp_X, exp_NN, exp_nNN = expectation_X(x_m, 1), get_NN_coupling(z_m, m, n, 1), get_nNN_coupling(z_m, m, n, 1)

    #diagonal terms
    cov_mat[0, 0] = expectation_X(x_m, 2) - exp_X**2
    cov_mat[1, 1] = get_NN_coupling(z_m, m, n, 2) - exp_NN**2
    cov_mat[2, 2] = get_nNN_coupling(z_m, m, n, 2) - exp_nNN**2

    #cross terms
    NN_index_l = flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    nNN_index_l = flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)
    NN_nNN_val = - (exp_NN * exp_nNN)

    for NN_indices in NN_index_l:
        for nNN_indices in nNN_index_l:
            indices = NN_indices + nNN_indices
            NN_nNN_val += get_exp_cross(z_m, indices)

    cov_mat[1, 2], cov_mat[2, 1]= NN_nNN_val, NN_nNN_val
    X_NN_val = -(exp_X * exp_NN)
    X_nNN_val = -(exp_X * exp_nNN)

    for h_idx in range(n_qbts):
        h_l = [h_idx]
        cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
        X_NN_index_l = get_measurement_index_l(h_idx, NN_index_l)
        X_nNN_index_l = get_measurement_index_l(h_idx, nNN_index_l)
        for indices in X_NN_index_l:
            X_NN_val += get_exp_cross(cross_m, indices)
        for indices in X_nNN_index_l:
            X_nNN_val += get_exp_cross(cross_m, indices)
    cov_mat[0, 1] = X_NN_val
    cov_mat[0, 2] = X_nNN_val
    cov_mat[2, 0], cov_mat[1, 0] = cov_mat[0, 2], cov_mat[0, 1]
    val, vec = np.linalg.eigh(cov_mat)
    argsort = np.argsort(val)
    val, vec = val[argsort], vec[:, argsort]
    orig_H = np.array([1, hyperparam_dict["J1"], hyperparam_dict["J2"]])
    orig_H = orig_H/np.linalg.norm(orig_H)
    HR_dist = distanceVecFromSubspace(orig_H, vec[:, :1])
    return HR_dist

def get_statevector(m, n, var_params, n_layers, ansatz_type, backend):
    circ = Q_Circuit_noiseless(m, n, var_params, n_layers, "ALA")
    circ.save_statevector()

    result = backend.run(circ).result()
    statevector = np.array(result.get_statevector(circ))
    return statevector

def get_HR_hyperparam_dict(dir_path, shots):
    HR_hyperparam_dict = np.load(dir_path.joinpath('HR_hyperparam_dict.npy'), allow_pickle = True).item()
    HR_hyperparam_dict["shots"] = shots
    HR_hyperparam_dict["backend"] = "aer_simulator"
    return HR_hyperparam_dict

def test1():
    """
    Compares the HR distance values from noiseless simulation and values from simulation with shot-noise only.
    1. Please specify params_dir_path
    2. Make sure that parent directory path of params_dir_path contains HR_hyperparam_dict.npy
    3. Make sure that param_idx does not exceed the maximum index in params_dir_path
    4. Choose the number of shots used to test
    """
    params_dir_path = pathlib.Path("/root/research/HR/PAPER_FIGURES/J1-J2/tests/ALA_3layers_noiseless/params_dir")
    param_idx = 400
    shots = 10000
    hyperparam_dict = get_HR_hyperparam_dict(params_dir_path.parents[0], shots)
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    var_params = get_params(params_dir_path, param_idx)
    backend = Aer.get_backend(hyperparam_dict["backend"])
    statevector = get_statevector(m, n, var_params, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"], backend)
    ops_l = get_operations_l(m, n)
    HR_dist_noiseless = get_HR_distance_noiseless(hyperparam_dict,statevector, ops_l)
    print("This is noiseless HR distance: ", HR_dist_noiseless)
    for _ in range(10):
        HR_dist_noisy = get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend)
        print("This is noisy HR distance: ", HR_dist_noisy)

def test2():
    """
    Compares (0, 1)th and (0, 2)th entry of covariance matrix from the noiseless simulation
    vs those from simulation with shot noise
    """
    params_dir_path = pathlib.Path("/root/research/HR/PAPER_FIGURES/J1-J2/tests/ALA_3layers_noiseless/params_dir")
    param_idx = 400
    shots = 10000
    hyperparam_dict = get_HR_hyperparam_dict(params_dir_path.parents[0], shots)
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    n_qbts = m*n
    var_params = get_params(params_dir_path, param_idx)
    backend = Aer.get_backend(hyperparam_dict["backend"])
    statevector = get_statevector(m, n, var_params, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"], backend)
    ops_l = get_operations_l(m, n)
    Hx, H_zz, H_n_zz = ops_l[0], ops_l[1], ops_l[2]
    exp_cross_val_0_1 = expected_op(np.matmul(Hx, H_zz), statevector)
    print("This is the value of cov_mat[0, 1] when noiseless: ", exp_cross_val_0_1)
    NN_index_l = flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    nNN_index_l = flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)
    for _ in range(5):
        X_NN_val = 0
        for h_idx in range(n_qbts):
            h_l = [h_idx]
            cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
            X_NN_index_l = get_measurement_index_l(h_idx, NN_index_l)
            for indices in X_NN_index_l:
                X_NN_val = X_NN_val + get_exp_cross(cross_m, indices)
        print("This is value of cov_mat[0, 1] with shot noise: ", X_NN_val)
    print("===============================")
    exp_cross_val_0_2 = expected_op(np.matmul(Hx, H_n_zz), statevector)
    print("This is the value of cov_mat[0, 2] when noiseless: ", exp_cross_val_0_2)
    for _ in range(5):
        X_nNN_val = 0
        for h_idx in range(n_qbts):
            h_l = [h_idx]
            cross_m = get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx)
            X_nNN_index_l = get_measurement_index_l(h_idx, nNN_index_l)
            for indices in X_nNN_index_l:
                X_nNN_val += get_exp_cross(cross_m, indices)
        print("This is value of cov_mat[0, 2] with shot noise: ", X_nNN_val)

def test3():
    """
    Compares nearest neighbor coupling strength between noiseless simulation and simulation with shot noise.
    """
    params_dir_path = pathlib.Path("/root/research/HR/PAPER_FIGURES/J1-J2/tests/ALA_3layers_noiseless/params_dir")
    param_idx = 400
    shots = 10000
    hyperparam_dict = get_HR_hyperparam_dict(params_dir_path.parents[0], shots)
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    n_qbts = m*n
    var_params = get_params(params_dir_path, param_idx)
    backend = Aer.get_backend(hyperparam_dict["backend"])
    statevector = get_statevector(m, n, var_params, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"], backend)
    ops_l = get_operations_l(m, n)
    Hx, H_zz, H_n_zz = ops_l[0], ops_l[1], ops_l[2]
    exp_val_H_zz = expected_op(H_zz, statevector)
    print("This is value of H_Z_{i}Z_{j} from noiseless simulation, where (i,j) is nearest neighbor: ", exp_val_H_zz)
    for _ in range(10):
        z_l, x_l = [], [i for i in range(n_qbts)]
        z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
        print("This is value of H_Z_{i}Z_{j} from simulation wtih shot noise, where (i,j) is nearest neighbor: ", get_NN_coupling(z_m, m, n, 1))

def test4():
    """
    Compares <X0Z1Z3> value between noiseless simulation and simulations with shot noise.
    """
    params_dir_path = pathlib.Path("/root/research/HR/PAPER_FIGURES/J1-J2/tests/ALA_3layers_noiseless/params_dir")
    param_idx = 400
    shots = 10000
    hyperparam_dict = get_HR_hyperparam_dict(params_dir_path.parents[0], shots)
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    n_qbts = m*n
    var_params = get_params(params_dir_path, param_idx)
    backend = Aer.get_backend(hyperparam_dict["backend"])
    statevector = get_statevector(m, n, var_params, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"], backend)
    var_params = get_params(params_dir_path, param_idx)
    sig_z = np.array([[1., 0.], [0., -1.]])
    sig_x = np.array([[0, 1], [1, 0]])
    temp = [np.eye(2)] * n_qbts
    temp[0] = sig_x
    temp[1] = sig_z
    temp[3] = sig_z
    tempSum = temp[0]
    for i in range(1, n_qbts):
        tempSum = np.kron(temp[i], tempSum)
    val1 = expected_op(tempSum, statevector)
    print("This is <X0Z1Z3> value from noiseless simulation: ", val1)
    for _ in range(10):
        cross_m = get_measurement(n_qbts, var_params, backend, [0], hyperparam_dict, param_idx)
        #breakpoint()
        print("This is <X0Z1Z3> value from shot noise simulation: ", get_exp_cross(cross_m, [0,1,3]))

def main():
    #RUN TESTS
    test1()
    test2()
    test3()
    test4()

if __name__ == '__main__':
    main()
