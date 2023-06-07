import numpy as np

def get_num_mt(mt):
    num_mt_l = list(map(lambda x: 1 if x == '0' else -1, mt))
    num_mt_l.reverse()
    return num_mt_l

def diagonalize(mat):
    """
    diagonalize matrix
    return sorted eigenvalues and eigen vectors
    """
    val, vec = np.linalg.eigh(mat)
    argsort = np.argsort(val)
    return val[argsort], vec[:, argsort]

def distanceVecFromSubspace(w, A):
    """
    Get L2 norm of distance from w to subspace spanned by columns of A

    Args:
        w (numpy 1d vector): vector of interest
        A (numpy 2d matrix): columns of A

    Return:
        L2 norm of distance from w to subspace spanned by columns of A
    """
    Q, _ = np.linalg.qr(A)
    r = np.zeros(w.shape)
    #len(Q[0]) is number of eigenvectors
    for i in range(len(Q[0])):
        r += np.dot(w, Q[:,i])*Q[:,i]
    return np.linalg.norm(r-w)

def get_exp_cross(cross_m, indices):
    tot_val, tot_count = 0, 0
    for cross_mt, m_count in cross_m.items():
        cross_mt = get_num_mt(cross_mt)
        mul = 1
        for ind in indices:
            mul = mul * cross_mt[ind]
        tot_val += (mul * m_count)
        tot_count += m_count
    return tot_val/tot_count

def create_identity(m, n):
    row = [np.eye(2)]*n
    temp = []
    for _ in range(m):
        temp.append(row.copy())
    return temp

def create_partial_Hamiltonian(neighbor_l, m, n):
    """
    Returns neighbor-coupling Hamiltonian, using neighbor_l.
    This function is used to create nearest-neighbor and next-nearest Hamiltonian

    neighbor_l: List[List[Tuple(i1, j1), Tuple(i2, j2)]]: list of neighboring qubit pair
    m: number of qubits in a row
    n: number of qubits in column
    """
    sig_z = np.array([[1., 0.], [0., -1.]])
    Hzz = 0
    for coord1, coord2 in neighbor_l:
        temp = create_identity(m, n)
        for i in range(m):
            for j in range(n):
                if (i,j) == coord1 or (i,j) == coord2:
                    temp[i][j] = sig_z
                else:
                    temp[i][j] = np.eye(2)
        tempSum = temp[0][0]
        for i in range(m):
            for j in range(n):
                if i != 0 or j != 0:
                    tempSum = np.kron(temp[i][j], tempSum)
        Hzz += tempSum
    return Hzz

def get_nearest_neighbors(m, n):
    NN_coord_l = []
    for i in range(m):
        for j in range(n):
            if i + 1 < m:
                NN_coord_l.append([(i,j), (i+1,j)])
            if j + 1 < n:
                NN_coord_l.append([(i,j), (i,j+1)])
    return NN_coord_l

def get_next_nearest_neighbors(m, n):
    nNN_coord_l = []
    for i in range(m):
        for j in range(n):
            if i+1 < m and j+1 < n:
                nNN_coord_l.append([(i,j), (i+1, j+1)])
            if i+1 < m and j-1 >= 0:
                nNN_coord_l.append([(i,j), (i+1, j-1)])
    return nNN_coord_l

def flatten_neighbor_l(neighbor_l, m, n):
    flat_neighbor_l = []
    for coord1, coord2 in neighbor_l:
        i1, j1 = coord1
        i2, j2 = coord2
        n1 = n*i1 + j1
        n2 = n*i2 + j2
        flat_neighbor_l.append([n1, n2])
    return flat_neighbor_l

def expectation_X(x_m, expo):
    tot_val, tot_count = 0, 0
    for x_mt, m_count in x_m.items():
        x_mt = get_num_mt(x_mt)
        tot_val += ((sum(x_mt)**expo)*m_count)
        tot_count += m_count
    exp_val = tot_val/tot_count
    return exp_val

def get_NN_coupling(z_m, m, n, expo):
    tot_val, tot_count = 0, 0
    NN_l = flatten_neighbor_l(get_nearest_neighbors(m, n), m , n)
    for z_mt, m_count in z_m.items():
        z_mt = get_num_mt(z_mt)
        sum_zz = 0
        for i, j in NN_l:
            sum_zz += z_mt[i]*z_mt[j]
        tot_val += ((sum_zz**expo) * m_count)
        tot_count += m_count
    exp_val = tot_val / tot_count
    return exp_val

def get_fidelity(wf, mat):
    fid = np.sqrt(np.matmul(np.conj(wf),np.matmul(mat, wf)))
    return fid.real

def get_nNN_coupling(z_m, m, n, expo):
    tot_val, tot_count = 0, 0
    nNN_l = flatten_neighbor_l(get_next_nearest_neighbors(m, n), m , n)
    for z_mt, m_count in z_m.items():
        z_mt = get_num_mt(z_mt)
        sum_zz = 0
        for i, j in nNN_l:
            sum_zz += z_mt[i]*z_mt[j]
        tot_val += ((sum_zz**expo) * m_count)
        tot_count += m_count
    exp_val = tot_val / tot_count
    return exp_val

def get_Hamiltonian(m, n, J1, J2):
    """
    Returns J1-J2 Hamiltonian. Total number of qubits: m x n

    m: number of qubits in a row
    n: number of qubits in a column
    J1: strength of nearest neighbor coupling
    J2: strength of next-nearest neighbor coupling

    H = X + J1*ZZ_<i,j> + J2*ZZ_<<i,j>>
    """
    N_qubits = m * n
    sig_x = np.array([[0., 1.], [1., 0.]])

    Hx = 0
    for i in range(N_qubits):
        temp = temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for k in range(1, N_qubits):
            tempSum = np.kron(temp[k], tempSum)
        Hx += tempSum

    NN_coord_l = get_nearest_neighbors(m, n)
    Hzz_J1 = create_partial_Hamiltonian(NN_coord_l, m, n)
    nNN_coord_l = get_next_nearest_neighbors(m, n)
    Hzz_J2 = create_partial_Hamiltonian(nNN_coord_l, m, n)
    return Hx + J1*Hzz_J1 + J2*Hzz_J2
