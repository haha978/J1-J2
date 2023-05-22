import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
from utils import flatten_neighbor_l, get_nearest_neighbors, get_next_nearest_neighbors
from utils import distanceVecFromSubspace, get_Hamiltonian
# from utils import distanceVecFromSubspace, get_exp_cross, get_exp_X, get_exp_ZZ
import pickle
import matplotlib.pyplot as plt
import os
from Circuit import Q_Circuit

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where VQE_hyperparam_dict.npy exists and HR distances and plots will be stored")
    args = parser.parse_args()
    return args

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_HR_distance(op_l, param_idx, params_dir_path, backend):
    cov_mat = np.zeros((3,3))
    m, n = hyperparam_dict["m"],  hyperparam_dict["n"]
    n_qbts = m * n

    #first I need to get the statevector
    var_params = get_params(params_dir_path, param_idx)
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

def main(args):
    if not os.path.exists(os.path.join(args.input_dir,"VQE_hyperparam_dict.npy")):
        raise ValueError( "input directory must be a valid input path that contains VQE_hyperparam_dict.npy")

    #LOAD All the hyperparamter data from VQE here
    VQE_hyperparam_dict = np.load(os.path.join(args.input_dir, "VQE_hyperparam_dict.npy"), allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")
    backend = Aer.get_backend("aer_simulator")

    hyperparam_dict = {}
    hyperparam_dict["gst_E"] = VQE_hyperparam_dict["gst_E"]
    hyperparam_dict["J1"], hyperparam_dict["J2"] = VQE_hyperparam_dict["J1"], VQE_hyperparam_dict["J2"]
    hyperparam_dict["m"], hyperparam_dict["n"] = VQE_hyperparam_dict["m"], VQE_hyperparam_dict["n"]
    hyperparam_dict["n_layers"] = VQE_hyperparam_dict["n_layers"]
    hyperparam_dict["ansatz_type"] = VQE_hyperparam_dict["ansatz_type"]

    print("This is hyperparameter dictionary newly constructed: ", hyperparam_dict)
    np.save(os.path.join(args.input_dir, "HR_hyperparam_dict.npy"), hyperparam_dict)

    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    gst_E = hyperparam_dict["gst_E"]
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    J1, J2 = hyperparam_dict["J1"], hyperparam_dict["J2"]
    n_layers = hyperparam_dict["n_layers"]

    Hamiltonian = get_Hamiltonian(m, n, J1, J2)
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonian)
    argmin_idx = np.argmin(eigen_vals)
    gst_E, ground_state = np.real(eigen_vals[argmin_idx]), eigen_vecs[:, argmin_idx]

    NN_index_l= flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    nNN_index_l= flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)

    HR_dist_hist = []
    fid_hist = []

    for param_idx in range(len(E_hist)):
        var_params = get_params(params_dir_path, param_idx)
        circ = Q_Circuit(m, n, var_params, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"])
        circ.save_statevector()
        result = backend.run(circ).result()
        statevector = result.get_statevector(circ)
        HR_dist = get_HR_distance(statevector, op_l)
        print(f"This is HR distance: {HR_dist} for {param_idx}th param")
        HR_dist_hist.append(HR_dist)
        fid_sqrt = np.vdot(statevector, ground_state)
        fid = np.vdot(fid_sqrt,fid_sqrt)
        fid_hist.append(fid)
        with open(os.path.join(args.input_dir, f"HR_dist_hist", f"{args.shots}shots.pkl"), "wb") as fp:
            pickle.dump(HR_dist_hist, fp)
        with open(os.path.join(args.input_dir, "fid_hist.pkl"), "wb") as fp:
            pickle.dump(fid_hist, fp)

    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 2-D "+ f"J1-J2 {m} x {n} grid \n" + f"J1: {J1}, J2: {J2}, shots: {args.shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3))
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.scatter(VQE_steps, fid_hist, c = 'g', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(args.input_dir+'/'+  str(m*n)+"qubits_"+ str(n_layers)+f"layers_shots_{args.shots}_HR_dist.png", dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 2-D J1-J2 model")
    args = get_args(parser)
    main(args)
