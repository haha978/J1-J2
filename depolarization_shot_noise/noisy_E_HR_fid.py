import sys
sys.path.insert(0, "../")
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import transpile
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import os
from depolarization_shot_noise.Circuit import Q_Circuit
from depolarization_shot_noise.utils import expectation_X, get_NN_coupling, get_nNN_coupling, get_exp_cross
from depolarization_shot_noise.utils import flatten_neighbor_l, get_nearest_neighbors, get_next_nearest_neighbors
from depolarization_shot_noise.utils import distanceVecFromSubspace, get_Hamiltonian

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where HR_hyperparam_dict.npy, parameter directory, and measurement directory exists. HR distances and plots will be stored in this given path.")
    parser.add_argument('--HR_hyperparam_dict_nm', type = str, help = "FILENAME of the hyperparameter dictionary used during HR")
    parser.add_argument('--param_idx_l', action = 'store_true', help = "if there is param_idx_l, then use param_idx_l.npy in input_dir \
                                to load the parameter index list to measure corresponding HR distances")
    args = parser.parse_args()
    return args

def get_measurement(n_qbts, var_params, backend, h_l, hyperparam_dict, param_idx):
    num_shots = hyperparam_dict["shots"]
    backendnm = hyperparam_dict["backend"]
    p1, p2 = hyperparam_dict["p1"], hyperparam_dict["p2"]
    measurement_path = os.path.join(args.input_dir, "measurement",f"{num_shots}_shots_{backendnm}_p1_{p1}_p2_{p2}", f"{param_idx}th_param_{''.join([str(e) for e in h_l])}qbt_h_gate.npy")
    if os.path.exists(measurement_path):
        #no need to save as it is already saved
        measurement = np.load(measurement_path, allow_pickle = "True").item()
    else:
        raise ValueError("Doesn't have measurement for corresponding idx")
    return measurement

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_noisy_E(hyperparam_dict, param_idx, params_dir_path, backend):
    """
    Obtain Energy values obtained from hardware runs / simulations with depolarization and shot noise.
    """
    m, n = hyperparam_dict["m"],  hyperparam_dict["n"]
    n_qbts = m * n
    z_l, x_l = [], [i for i in range(n_qbts)]
    var_params = get_params(params_dir_path, param_idx)
    z_m = get_measurement(n_qbts, var_params, backend, z_l, hyperparam_dict, param_idx)
    x_m = get_measurement(n_qbts, var_params, backend, x_l, hyperparam_dict, param_idx)
    exp_X, exp_NN, exp_nNN = expectation_X(x_m, 1), get_NN_coupling(z_m, m, n, 1), get_nNN_coupling(z_m, m, n, 1)
    E = exp_X + hyperparam_dict["J1"]*exp_NN + hyperparam_dict["J2"]*exp_nNN
    print("This is noisy energy: ", E)
    return E

def main(args):
    HR_hyperparam_dict_path = os.path.join(args.input_dir, "HR_hyperparam_dict", args.HR_hyperparam_dict_nm)
    if not os.path.exists(HR_hyperparam_dict_path):
        raise ValueError( "args.input_dir with args.HR_hyperparam_dict_nm must be a valid input path that contains HR_hyperparam_dict.npy")

    #LOAD the hyperparamter data from HR here
    hyperparam_dict = np.load(HR_hyperparam_dict_path, allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")

    shots, backend, p1, p2 = hyperparam_dict["shots"], hyperparam_dict["backend"], hyperparam_dict["p1"], hyperparam_dict["p2"]

    print("This is hyperparameter dictionary newly constructed: ", hyperparam_dict)
    if not os.path.isdir(os.path.join(args.input_dir, "measurement", f"{shots}_shots_{backend}_p1_{p1}_p2_{p2}")):
        raise ValueError("measurement directory does not exist in args.input_dir")

    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    gst_E = hyperparam_dict["gst_E"]
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    J1, J2 = hyperparam_dict["J1"], hyperparam_dict["J2"]
    n_layers = hyperparam_dict["n_layers"]

    noisy_E_hist = []
    fid_hist = []
    if args.param_idx_l:
        param_idx_l_path = os.path.join(args.input_dir, "param_idx_l.npy")
        assert os.path.isfile(param_idx_l_path), "there is no param_idx_l.npy file in input_dir"
        param_idx_l = np.load(param_idx_l_path, allow_pickle = "True")
        with open(os.path.join(args.input_dir,  "HR_dist_hist", f"HR_param_idx_l_{shots}shots_{backend}_p1_{p1}_p2_{p2}.pkl"), "rb") as fp:
            HR_dist_hist = pickle.load(fp)
    else:
        param_idx_l = list(range(len(E_hist)))
        with open(os.path.join(args.input_dir,  "HR_dist_hist", f"HR_{shots}shots_{backend}_p1_{p1}_p2_{p2}.pkl"), "rb") as fp:
            HR_dist_hist = pickle.load(fp)

    #Define filename
    if args.param_idx_l:
        fid_hist_filename = f"fid_param_idx_l_p1_{p1}_p2_{p2}.pkl"
        noisy_E_hist_filename = f"noisy_E_param_idx_l_{shots}_shots_{backend}__p1_{p1}_p2_{p2}.pkl"
        img_name = f"layers_shots_param_idx_l_{shots}_shots_{backend}_p1_{p1}_p2_{p2}_noisy_HR_dist.svg"
    else:
        fid_hist_filename = f"fid_p1_{p1}_p2_{p2}.pkl"
        noisy_E_hist_filename = f"noisy_E_{shots}_shots_{backend}__p1_{p1}_p2_{p2}.pkl"
        img_name = f"layers_shots_{shots}_shots_{backend}_p1_{p1}_p2_{p2}_noisy_HR_dist.svg"


    if not os.path.isdir(os.path.join(args.input_dir, f"noisy_E_hist")):
        os.makedirs(os.path.join(args.input_dir, f"noisy_E_hist"))

    #calculate noisy_E_hist and save it
    for param_idx in param_idx_l:
        noisy_E_hist.append(get_noisy_E(hyperparam_dict, param_idx, params_dir_path, backend))
        with open(os.path.join(args.input_dir, f"noisy_E_hist", noisy_E_hist_filename), "wb") as fp:
            pickle.dump(noisy_E_hist, fp)

    #Load Fidelity
    fid_hist_path = os.path.join(args.input_dir, "fid_hist", fid_hist_filename)
    assert os.path.isfile(fid_hist_path), "corresponding fidelity history must exist"
    with open(fid_hist_path, "rb") as fp:
        fid_hist = pickle.load(fp)

    #create plots
    fig, ax = plt.subplots()
    ax.scatter(param_idx_l, noisy_E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 2-D "+ f"J1-J2 {m} x {n} grid \n" + f"J1: {J1}, J2: {J2}, shots: {shots}" + \
                    '\n' + 'True Ground energy: ' + str(round(gst_E, 3))
    if not (hyperparam_dict['p1'] == 0 and hyperparam_dict['p2'] == 0):
        title = title + '\n' + f"p1: {p1}, p2: {p2}"
    plt.title(title, fontdict = {'fontsize' : 15})
    ax2 = ax.twinx()
    ax2.scatter(param_idx_l, HR_dist_hist, c = 'r', alpha = 0.8, marker=".", label = "HR distance")
    ax2.scatter(param_idx_l, fid_hist, c = 'g', alpha = 0.8, marker=".", label = "Fidelity")
    ax2.set_ylabel("HR distance | Fidelity")
    ax2.legend(bbox_to_anchor=(1.28, 1.22), fontsize = 10)
    plt.savefig(args.input_dir+'/'+  str(m*n)+"qubits_"+ str(n_layers)+img_name, dpi = 300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "HR for 2-D J1-J2 model")
    args = get_args(parser)
    main(args)
