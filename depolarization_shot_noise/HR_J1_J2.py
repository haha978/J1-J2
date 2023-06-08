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
from depolarization_shot_noise.utils import distanceVecFromSubspace, get_Hamiltonian, get_fidelity

HR_dist_hist = []

def get_args(parser):
    parser.add_argument('--input_dir', type = str, help = "directory where VQE_hyperparam_dict.npy exists. HR distances and plots will be stored in the input_dir")
    parser.add_argument('--shots', type=int, default=1000, help = "number of shots during HamiltonianReconstuction (default: 1000)")
    parser.add_argument('--backend', type = str, default = "aer_simulator", help = "backend for ionq runs (aer_simulator, ionq.simulator, ionq.qpu, ionq.qpu.aria-1, default = aer_simulator)")
    parser.add_argument('--use_VQE_p1_p2', action = 'store_true', help = "Use VQE p1 and p2 values when simulating HR. Only compatible with aer_simulator backend")
    parser.add_argument('--param_idx_l', action = 'store_true', help = "if there is param_idx_l, then use param_idx_l.npy in input_dir \
                                to load the parameter index list to measure corresponding HR distances")
    parser.add_argument('--p1', type = float, default = 0.0, help = "one-qubit gate depolarization noise (default: 0.0)")
    parser.add_argument('--p2', type = float, default = 0.0, help = "two-qubit gate depolarization noise (default: 0.0)")
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
        m, n = hyperparam_dict["m"], hyperparam_dict["n"]
        circ = Q_Circuit(m, n, var_params, h_l, hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"])
        circ.measure(list(range(n_qbts)), list(range(n_qbts)))
        circ = transpile(circ, backend)
        job = backend.run(circ, shots = num_shots)
        if backendnm != "aer_simulator":
            job_id = job.id()
            job_monitor(job)
        result = job.result()
        measurement = dict(result.get_counts())
        np.save(measurement_path, measurement)
    return measurement

def get_params(params_dir_path, param_idx):
    var_params = np.load(os.path.join(params_dir_path, f"var_params_{param_idx}.npy"))
    return var_params

def get_measurement_index_l(h_idx, z_indices):
    m_index_l = []
    for zi, zj in z_indices:
        if h_idx != zi and h_idx != zj:
            m_index_l.append([h_idx, zi, zj])
    return m_index_l

def get_fid(hyperparam_dict, param_idx, params_dir_path, ground_state, backend):
    var_params = get_params(params_dir_path, param_idx)
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    circ = Q_Circuit(m, n, var_params, [], hyperparam_dict["n_layers"], hyperparam_dict["ansatz_type"])
    if hyperparam_dict['p1'] == 0 and hyperparam_dict['p2'] == 0:
        circ.save_statevector()
        result = backend.run(circ).result()
        statevector = result.get_statevector(circ)
        statevector = np.array(statevector)
        fid = np.vdot(statevector, ground_state)
    else:
        circ.save_density_matrix()
        result = backend.run(circ).result()
        den_mat = result.data(0)['density_matrix']
        fid = get_fidelity(ground_state, den_mat)
    return fid.real

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

def main(args):
    if not os.path.exists(os.path.join(args.input_dir,"VQE_hyperparam_dict.npy")):
        raise ValueError( "input directory must be a valid input path that contains VQE_hyperparam_dict.npy")
    if not os.path.isdir(os.path.join(args.input_dir, "HR_dist_hist")):
        os.makedirs(os.path.join(args.input_dir, "HR_dist_hist"))
    if not os.path.isdir(os.path.join(args.input_dir, "HR_hyperparam_dict")):
        os.makedirs(os.path.join(args.input_dir, "HR_hyperparam_dict"))

    #LOAD All the hyperparamter data from VQE here
    VQE_hyperparam_dict = np.load(os.path.join(args.input_dir, "VQE_hyperparam_dict.npy"), allow_pickle = True).item()
    params_dir_path = os.path.join(args.input_dir,"params_dir")

    #Initialize hyperparameter dictionary
    hyperparam_dict = {}
    hyperparam_dict["gst_E"] = VQE_hyperparam_dict["gst_E"]
    hyperparam_dict["J1"], hyperparam_dict["J2"] = VQE_hyperparam_dict["J1"], VQE_hyperparam_dict["J2"]
    hyperparam_dict["m"], hyperparam_dict["n"] = VQE_hyperparam_dict["m"], VQE_hyperparam_dict["n"]
    hyperparam_dict["p1"], hyperparam_dict["p2"] = args.p1, args.p2
    hyperparam_dict["n_layers"] = VQE_hyperparam_dict["n_layers"]
    hyperparam_dict["ansatz_type"] = VQE_hyperparam_dict["ansatz_type"]
    #Need a new number of shots for HR distance for cost purposes.
    hyperparam_dict["shots"] = args.shots
    hyperparam_dict["backend"] = args.backend

    if args.use_VQE_p1_p2:
        hyperparam_dict["p1"], hyperparam_dict["p2"] = VQE_hyperparam_dict["p1"], VQE_hyperparam_dict["p2"]


    print("This is hyperparameter dictionary newly constructed: ", hyperparam_dict)
    #set the most updated p1 and p2 for updated purposes
    p1, p2 = hyperparam_dict["p1"], hyperparam_dict["p2"]
    #create noise_model to use it when simulating fidelity
    noise_model = NoiseModel()
    p1_error = depolarizing_error(p1, 1)
    p2_error = depolarizing_error(p2, 2)
    noise_model.add_all_qubit_quantum_error(p1_error, ['h','ry'])
    noise_model.add_all_qubit_quantum_error(p2_error, ['cx'])

    if args.backend == "aer_simulator":
        if hyperparam_dict["p1"] == 0 and hyperparam_dict["p2"] == 0:
            backend = AerSimulator()
        else:
            backend = AerSimulator(noise_model = noise_model)
    else:
        provider = AzureQuantumProvider(resource_id = "/subscriptions/58687a6b-a9bd-4f79-b7af-1f8f76760d4b/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/HamiltonianReconstruction",\
                                        location = "West US")
        backend = provider.get_backend(args.backend)

    if not os.path.isdir(os.path.join(args.input_dir, "measurement", f"{args.shots}_shots_{args.backend}_p1_{p1}_p2_{p2}")):
        os.makedirs(os.path.join(args.input_dir, "measurement", f"{args.shots}_shots_{args.backend}_p1_{p1}_p2_{p2}"))

    np.save(os.path.join(args.input_dir, "HR_hyperparam_dict", f"{args.shots}_shots_{args.backend}_p1_{p1}_p2_{p2}.npy"), hyperparam_dict)

    with open(os.path.join(args.input_dir, "E_hist.pkl"), "rb") as fp:
        E_hist = pickle.load(fp)

    if args.param_idx_l:
        fid_hist_filename = f"fid_param_idx_l_p1_{p1}_p2_{p2}.pkl"
        HR_dist_hist_filename =  f"HR_param_idx_l_{args.shots}shots_{args.backend}_p1_{p1}_p2_{p2}.pkl"
        img_name = f"layers_shots_param_idx_l_{args.shots}_p1_{p1}_p2_{p2}_HR_dist.png"
    else:
        fid_hist_filename = f"fid_p1_{p1}_p2_{p2}.pkl"
        HR_dist_hist_filename =  f"HR_{args.shots}shots_{args.backend}_p1_{p1}_p2_{p2}.pkl"
        img_name = f"layers_shots_{args.shots}_p1_{p1}_p2_{p2}_HR_dist.png"

    gst_E = hyperparam_dict["gst_E"]
    m, n = hyperparam_dict["m"], hyperparam_dict["n"]
    J1, J2 = hyperparam_dict["J1"], hyperparam_dict["J2"]
    n_layers = hyperparam_dict["n_layers"]

    #get ground state NEED TO DELETE THIS LINE THO --> PROBABLY JUST GET IT WHEN VQE
    Hamiltonian = get_Hamiltonian(m, n, J1, J2)
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonian)
    argmin_idx = np.argmin(eigen_vals)
    gst_E, ground_state = np.real(eigen_vals[argmin_idx]), eigen_vecs[:, argmin_idx]

    NN_index_l= flatten_neighbor_l(get_nearest_neighbors(m, n), m, n)
    nNN_index_l= flatten_neighbor_l(get_next_nearest_neighbors(m, n), m, n)

    HR_dist_hist = []
    fid_hist = []
    if args.param_idx_l:
        param_idx_l_path = os.path.join(args.input_dir, "param_idx_l.npy")
        assert os.path.isfile(param_idx_l_path), "there is no param_idx_l.npy file in input_dir"
        param_idx_l = np.load(param_idx_l_path, allow_pickle = "True")
        print(param_idx_l)
    else:
        param_idx_l = list(range(len(E_hist)))

    for param_idx in param_idx_l:
        HR_dist = get_HR_distance(hyperparam_dict, param_idx, params_dir_path, backend)
        print(f"This is HR distance: {HR_dist} for {param_idx}th param")
        HR_dist_hist.append(HR_dist)
        with open(os.path.join(args.input_dir, f"HR_dist_hist", HR_dist_hist_filename), "wb") as fp:
            pickle.dump(HR_dist_hist, fp)

    #fid_hist
    if not os.path.isdir(os.path.join(args.input_dir, "fid_hist")):
        os.makedirs(os.path.join(args.input_dir, "fid_hist"))

    #backend initialization for fidelity
    if p1 == 0 and p2 == 0:
        fid_backend = AerSimulator()
    else:
        fid_backend = AerSimulator(method = 'density_matrix', noise_model = noise_model)

    for param_idx in param_idx_l:
        fid  = get_fid(hyperparam_dict, param_idx, params_dir_path, ground_state, fid_backend)
        print(f"This is fidelity: {fid} for {param_idx}th param")
        fid_hist.append(fid)
        with open(os.path.join(args.input_dir, "fid_hist", fid_hist_filename), "wb") as fp:
            pickle.dump(fid_hist, fp)

    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 2-D "+ f"J1-J2 {m} x {n} grid \n" + f"J1: {J1}, J2: {J2}, shots: {args.shots}" + \
                    '\n' + 'True Ground energy: ' + str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3))
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
    parser = argparse.ArgumentParser(description = "VQE for 2-D J1-J2 model")
    args = get_args(parser)
    main(args)
