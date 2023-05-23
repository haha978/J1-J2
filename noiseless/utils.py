import numpy as np

def create_identity(m, n):
    row = [np.eye(2)]*n
    temp = []
    for _ in range(m):
        temp.append(row.copy())
    return temp

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

def flatten_neighbor_l(neighbor_l, m, n):
    flat_neighbor_l = []
    for coord1, coord2 in neighbor_l:
        i1, j1 = coord1
        i2, j2 = coord2
        n1 = n*i1 + j1
        n2 = n*i2 + j2
        flat_neighbor_l.append([n1, n2])
    return flat_neighbor_l

def expected_op(op, wf):
    return np.vdot(wf, np.matmul(op, wf)).real

def expected_op1_op2(op1, op2, wf):

    return np.vdot(wf, np.matmul(op1, np.matmul(op2, wf))).real

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

def get_Hx(N_qubits):
    Hx = 0
    sig_x = np.array([[0., 1.], [1., 0.]])
    for i in range(N_qubits):
        temp = temp = [np.eye(2)]*N_qubits
        temp[i] = sig_x
        tempSum = temp[0]
        for k in range(1, N_qubits):
            tempSum = np.kron(tempSum, temp[k])
        Hx += tempSum
    return Hx

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
    Hx = get_Hx(N_qubits)
    NN_coord_l = get_nearest_neighbors(m, n)
    Hzz_J1 = create_partial_Hamiltonian(NN_coord_l, m, n)
    nNN_coord_l = get_next_nearest_neighbors(m, n)
    Hzz_J2 = create_partial_Hamiltonian(nNN_coord_l, m, n)
    return Hx + J1*Hzz_J1 + J2*Hzz_J2
