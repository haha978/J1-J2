from HR_J1_J2 import get_HR_distance
from utils import get_Hamiltonian, create_partial_Hamiltonian, get_Hx
from utils import get_nearest_neighbors, get_next_nearest_neighbors
import numpy as np

def test1():
    """
    Check whether ground state gives zero HR distance
    """
    m, n = 3, 3
    J1, J2 = 0.5, 0.2
    Hamiltonian = get_Hamiltonian(m, n, J1, J2)
    hyperparam_dict = {}
    hyperparam_dict["J1"], hyperparam_dict["J2"] = J1, J2
    eigen_vals, eigen_vecs = np.linalg.eig(Hamiltonian)
    argmin_idx = np.argmin(eigen_vals)
    gst_E, ground_state = np.real(eigen_vals[argmin_idx]), eigen_vecs[:, argmin_idx]
    NN_index_l= get_nearest_neighbors(m, n)
    nNN_index_l= get_next_nearest_neighbors(m, n)
    ops_l = []
    ops_l.append(get_Hx(m*n))
    ops_l.append(create_partial_Hamiltonian(NN_index_l, m, n))
    ops_l.append(create_partial_Hamiltonian(nNN_index_l, m, n))
    HR_dist = get_HR_distance(hyperparam_dict, ground_state, ops_l)
    print("This is HR distance of the ground state: ", HR_dist)
    assert abs(HR_dist) <= 1e-12

def main():
    test1()

if __name__ == '__main__':
    main()
