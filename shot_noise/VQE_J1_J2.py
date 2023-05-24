import sys
sys.path.insert(0, "../")
import numpy as np
import os
from qiskit import QuantumCircuit, Aer
from qiskit.algorithms.optimizers import IMFIL
from qiskit import transpile
from qiskit.tools.monitor import job_monitor
import argparse
from functools import partial
import pickle
import matplotlib.pyplot as plt
from shot_noise.utils import get_Hamiltonian, expectation_X, get_NN_coupling, get_nNN_coupling
from shot_noise.utils import get_nearest_neighbors
from shot_noise.Circuit import Q_Circuit

E_hist = []

def get_args(parser):
    parser.add_argument('--m', type = int, help = "number of qubits in a row")
    parser.add_argument('--n', type = int, help = "number of qubits in a column")
    parser.add_argument('--J1', type = float, default = 0.5, help = "strength of nearest neighbor coupling(default J: 0.5)")
    parser.add_argument('--J2', type = float, default = 0.05, help = "strength of next nearest neighbor coupling(default J: 0.05)")
    parser.add_argument('--ansatz_type', type = str, default = "ALA", help = "Ansatz type (default: ALA)")
    parser.add_argument('--shots', type = int, default = 10000, help = "Number of shots (default: 10000)")
    parser.add_argument('--max_iter', type = int, default = 500, help = "maximum number of iterations (default: 10000)")
    parser.add_argument('--n_layers', type = int, default = 3, help = "number of ALA ansatz layers needed (default: 3)")
    parser.add_argument('--output_dir', type = str, default = ".", help = "output directory being used (default: .)")
    parser.add_argument('--init_param', type = str, default = "NONE", help = "parameters for initialization (default: NONE)")
    args = parser.parse_args()
    return args

def get_measurement(m, n, var_params, backend, shots, h_l):
    n_qbts = m * n
    circ = Q_Circuit(m, n, var_params, h_l, args.n_layers, args.ansatz_type)
    circ.measure(list(range(n_qbts)), list(range(n_qbts)))
    circ = transpile(circ, backend)
    job = backend.run(circ, shots = shots)
    result = job.result()
    measurement = dict(result.get_counts())
    return measurement

def get_E(var_params, m, n, shots, J1, J2, backend):
    """
    Get energy
    """
    n_qbts = m * n
    z_l, x_l = [], [i for i in range(n_qbts)]
    z_m = get_measurement(m, n, var_params, backend, shots, z_l)
    x_m = get_measurement(m, n, var_params, backend, shots, x_l)

    # Need to save energy here
    Hx, Hzz, Hz_z = expectation_X(x_m, 1), get_NN_coupling(z_m, m, n, 1), get_nNN_coupling(z_m, m, n, 1)
    # exp_X_sqr, exp_ZZ_sqr = get_exp_X(x_m, 2), get_exp_ZZ(z_m, 2)
    E = Hx + J1*Hzz + J2*Hz_z
    E_hist.append(E)
    with open(os.path.join(args.output_dir, "E_hist.pkl"), "wb") as fp:
        pickle.dump(E_hist, fp)
    np.save(os.path.join(args.output_dir, "params_dir", f"var_params_{len(E_hist)-1}.npy"), var_params)
    print("This is energy: ", E)
    return E

def main(args):
    # Dont save params yet
    if not os.path.exists(os.path.join(args.output_dir,"params_dir")):
        os.makedirs(os.path.join(args.output_dir,"params_dir"))
    n_qbts = args.m * args.n
    Nparams = 0
    if args.ansatz_type == "ALA":
        if n_qbts % 2 == 0:
            for i in range(args.n_layers):
                if i % 2 == 0:
                    Nparams += n_qbts
                else:
                    Nparams += (n_qbts - 2)
        else:
            for i in range(args.n_layers):
                Nparams += (n_qbts - 1)
    elif args.ansatz_type == "HVA":
        Nparams = args.n_layers * (n_qbts + len(get_nearest_neighbors(args.m, args.n)))
    else:
        raise ValueError("please type the correct ansatz type")

    Hamiltonian = get_Hamiltonian(args.m, args.n, args.J1, args.J2)
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonian)
    argmin_idx = np.argmin(eigen_vals)
    gst_E, ground_state = np.real(eigen_vals[argmin_idx]), eigen_vecs[:, argmin_idx]
    print("This ground state energy: ", gst_E)

    # Sets parameter initialization here.
    if args.init_param == "NONE":
        var_params = np.random.uniform(low = -np.pi, high = np.pi, size = Nparams)
    else:
        param_PATH = os.path.join(args.init_param)
        var_params = np.load(param_PATH)
        assert len(var_params) == Nparams, "loaded params needs to have the same length as the Nparams"

    bounds = np.tile(np.array([-np.pi, np.pi]), (Nparams,1))
    backend = Aer.get_backend('aer_simulator')
    imfil = IMFIL(maxiter = args.max_iter)
    get_E_func = partial(get_E, m = args.m, n = args.n, shots = args.shots, J1 = args.J1, J2 = args.J2, backend = backend)
    result = imfil.minimize(get_E_func, x0 = var_params, bounds = bounds)
    fig, ax = plt.subplots()
    VQE_steps = np.array(list(range(len(E_hist))))
    ax.scatter(VQE_steps, E_hist, c = 'b', alpha = 0.8, marker = ".", label = "Energy")
    ax.set_xlabel('VQE Iterations')
    ax.set_ylabel("Energy")
    ax.legend(bbox_to_anchor=(1.28, 1.30), fontsize = 10)
    title = "VQE 2-D "+ f"J1-J2 {args.m} x {args.n} grid \n" + f"J1: {args.J1}, J2: {args.J2}, shots: {args.shots}" + '\n' + 'True Ground energy: ' + \
            str(round(gst_E, 3)) + '\n' + 'Estimated Ground Energy: '+ str(round(float(min(E_hist)), 3))
    plt.title(title, fontdict = {'fontsize' : 15})
    plt.savefig(args.output_dir+'/'+  str(n_qbts)+"qubits_"+ str(args.n_layers)+f"layers_shots_{args.shots}.png", dpi = 300, bbox_inches='tight')
    # Create hyperparam_dict for Hamiltonian Reconstruction
    hyperparam_dict = {}
    hyperparam_dict["m"], hyperparam_dict["n"] = args.m, args.n
    hyperparam_dict["J1"], hyperparam_dict["J2"] = args.J1, args.J2
    hyperparam_dict["shots"], hyperparam_dict["n_layers"] = args.shots, args.n_layers
    hyperparam_dict["ansatz_type"] = args.ansatz_type
    hyperparam_dict["gst_E"] = gst_E
    np.save(os.path.join(args.output_dir, "VQE_hyperparam_dict.npy"), hyperparam_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "VQE for 1-D TFIM with non-periodic boundary condition")
    args = get_args(parser)
    main(args)
