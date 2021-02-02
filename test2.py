import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

import pickle, random
import itertools as it
import multiprocessing
from multiprocessing import Array

import mp_globals
mp_globals.data = []

# just define some constants for easy reference
pi = np.pi

# mnist test dataset (10000 * 784)
def load_mnist():
    with open("data/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
        return mnist["test_images"], mnist["test_images"].shape


'''SETUP'''
def setup(vecs, K, seed=123):
    # reset the seed
    np.random.seed(seed)

    _, P = vecs.shape

    # open a dataset of vectors of size (N * P) (X)
    # normalise all of them (row-wise)
    X = normalize(vecs, norm='l2')

    # generate a random row vector with P elements E
    e = np.random.standard_normal(size=(1, P))
    # normalise it (for consistency)
    e = normalize(e, norm='l2')

    # store every N angles (against the random vector E) in an array A
    dots = np.matmul(X, e.T)
    A = np.arccos(dots).flatten()

    # generate a random matrix of size (P * K) (where k can be varied) (R)
    R = np.random.standard_normal(size=(P, K))

    # calc V = sign(X * R) of size (N * K)
    V = np.sign(np.matmul(X, R))

    # r = sign(E * R) of size (1 * K)
    r = np.sign(np.matmul(e, R)).flatten()
    
    return X, e, R, A, V, r


'''PROCEDURES'''
def cv_orig(A, V, r_cache, K, v1, v2):
    # then, for any two vectors (indexed as v1 and v2)
    # calc A = sum(V[v1] == V[v2]) / K
    # B = sum(V[v1] == r) / K
    # C = sum(V[v2] == r) / K
    A_est = sum(V[v1] == V[v2]) / K
    B_est = r_cache[v1] / K
    C_est = r_cache[v2] / K

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for theta_ij = pi * (1 - A)
    theta_ij = pi * (1 - A_est)

    # if theta_ij is 0, then c1 and c2 will result in division by 0
    # so just return theta_ij = 0
    if theta_ij == 0: return theta_ij

    # calculate the control variate coefficients as below:
    c1 = (-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)
    c2 = (2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)
    
    # calculate the true mean values for CV
    u_b = 1 - theta_i1 / pi
    u_c = 1 - theta_j1 / pi

    # calculate the new estimate for A
    final = A_est + c1 * (B_est - u_b) + c2 * (C_est - u_c)

    # hence calculate the angle
    return pi * (1 - final)

def cv_empirical(A, V, r, K, v1, v2):
    # then, for any two vectors (indexed as v1 and v2)
    # calc A = sum(V[v1] == V[v2]) / K
    # B = sum(V[v1] == r) / K
    # C = sum(V[v2] == r) / K
    A_est = sum(V[v1] == V[v2]) / K
    B_est = sum(V[v1] == r) / K
    C_est = sum(V[v2] == r) / K

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # calculate the control variate coefficients as below (using empirical observations):
    A_vec = V[v1] == V[v2]
    B_vec = V[v1] == r
    C_vec = V[v2] == r
    vecs = np.array([A_vec, B_vec, C_vec])

    cov_mat = np.cov(vecs, ddof=0)
    c_ab = cov_mat[0, 1]
    c_ac = cov_mat[0, 2]
    c_bc = cov_mat[1, 2]
    v_b = cov_mat[1, 1]
    v_c = cov_mat[2, 2]
    denom = c_bc ** 2 - v_b * v_c
    
    if denom == 0: return 0

    c1 = (c_ab * v_c - c_ac * c_bc) / denom
    c2 = (-c_ab * c_bc + c_ac * v_b) / denom
    
    # calculate the true mean values for CV
    u_b = 1 - theta_i1 / pi
    u_c = 1 - theta_j1 / pi

    # calculate the new estimate for A
    final = A_est + c1 * (B_est - u_b) + c2 * (C_est - u_c)

    # hence calculate the angle
    return pi * (1 - final)



def cv_cubic(A, V, r_cache, K, v1, v2, NR_iter=15):
    # then, for any two vectors (indexed as v1 and v2)
    # calc A = sum(V[v1] == V[v2]) / K
    # B = sum(V[v1] == r) / K
    # C = sum(V[v2] == r) / K
    A_est = sum(V[v1] == V[v2]) / K
    B_est = r_cache[v1] / K
    C_est = r_cache[v2] / K

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for theta_ij = pi * (1 - A)
    theta_ij = pi * (1 - A_est)

    if theta_ij == 0: return theta_ij

    # find the theta_ij that solves the cubic
    def f(theta_ij):
        return (theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A_est - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B_est - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C_est - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))
    
    def fp(theta_ij):
        return (-2*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1)*(theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A_est - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B_est - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C_est - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)) + (pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + 2*theta_ij*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) + pi*theta_j1**2 + 2*pi*(A_est - 1)*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) - 2*(theta_i1 + pi*(B_est - 1))*(pi*theta_i1 + 2*theta_ij*theta_j1 - pi*theta_ij - pi*theta_j1) - 2*(theta_j1 + pi*(C_est - 1))*(2*theta_i1*theta_ij - pi*theta_i1 - pi*theta_ij + pi*theta_j1)))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)**2)

    # use newton-raphson to find the root
    for _ in range(NR_iter):
        theta_ij = theta_ij - f(theta_ij) / fp(theta_ij)
    
    return theta_ij

def srp(V, K, v1, v2):
    # for any two vectors (indexed as v1, v2)
    # calculate the hamming distance between the two projections
    dist = sum(V[v1] != V[v2])
    
    # then, the estimate is given by dist * pi / K
    return dist * pi / K

def calc_one_pair(pair):
    X = mp_globals.X
    A = mp_globals.A
    V = mp_globals.V
    r_cache = mp_globals.r_cache

    v1, v2 = pair

    t_act = np.arccos(np.dot(X[v1], X[v2]))

    t_cv_o = cv_orig(A, V, r_cache, K, v1, v2)
    e_o = (t_act - t_cv_o) ** 2

    # t_cv_e = cv_empirical(A, V, r, K, v1, v2)
    # err_cv_e += (t_act - t_cv_e)** 2
    
    t_cv_c = cv_cubic(A, V, r_cache, K, v1, v2)
    e_c = (t_act - t_cv_c) ** 2

    t_srp = srp(V, K, v1, v2)
    e_s = (t_act - t_srp)** 2
    
    return e_o, e_c, e_s


'''ANALYSIS'''
# possibly repeat this for different values of k
# possibly repeat this for different pairs of v1, v2
#   - alternatively, repeat this for every pair, then calculate the sum of squared errors between the actual angles and the estimated ones
#   - also, calculate the variance
# maybe calculate estimates using the cubic as well (just to compare rmse and variances)
#   - actually, because we know that this is equivalent to the mle approach, this also works as a way to show variance reduction using this cv approach (i think?)

# store every pairwise angle in an array (store only cosine for speed)
# G = np.einsum('ij,kj->ik', X, X)
# G = np.arccos(cos_thetas)

SEED = 100
K = 256

def init_pool(X_s, A_s, V_s, r_s):
    mp_globals.X = X_s
    mp_globals.A = A_s
    mp_globals.V = V_s
    mp_globals.r_cache = r_s


def main():
    # global X, A, V, r_cache

    # test this for the mnist test dataset first
    mnist, (N, _) = load_mnist()
    X, _, _, A, V, r = setup(mnist, K, SEED)

    print("Caching counts with respect to extra vector (simulation speed up)")
    r_cache = []
    for v in tqdm(range(N)):
        r_cache.append(sum(V[v] == r))

    # mp_globals.X = X
    # mp_globals.A = A
    # mp_globals.V = V
    # mp_globals.r_cache = r_cache

    # print(X.dtype)
    # print(A.dtype)
    # print(V.dtype)

    X_share = Array('f', X, lock=False)
    A_share = Array('f', A, lock=False)
    V_share = Array('f', V, lock=False)
    r_share = Array('i', r_cache, lock=False)

    err_cv_o = 0
    # err_cv_e = 0
    err_cv_c = 0
    err_srp = 0

    # every possible pair (gonna take a day)
    # total = (N * (N - 1) // 2)
    # iters = it.combinations(list(range(N)), 2)

    # a_dict = np.load('angles_dict.npy', allow_pickle=True).item()

    # take a subset
    total = 10000
    iters = np.random.randint(N, size=(total, 2))

    print(f"Calculating estimates for {total} pairs")

    with multiprocessing.Pool(initializer=init_pool, initargs=(X_share, A_share, V_share, r_share)) as p:
        results = list(tqdm(p.imap(calc_one_pair, iters), total=total))
    
    for (e_o, e_c, e_s) in results:
        err_cv_o += e_o
        err_cv_c += e_c
        err_srp += e_s

    # for pair in tqdm(iters, total=total):
    #     # calculate the actual angle
    #     # t_act = np.arccos(np.dot(X[v1], X[v2]))
    #     # # if v1 > v2:
    #     # #     v1, v2 = v2, v1
    #     # # t_act = a_dict[(v1, v2)]

    #     # # obtain the estimates using the various procedures
    #     # # calculate the errors associated with each estimate
    #     # t_cv_o = cv_orig(A, V, r_cache, K, v1, v2)
    #     # err_cv_o += (t_act - t_cv_o)** 2

    #     # # t_cv_e = cv_empirical(A, V, r, K, v1, v2)
    #     # # err_cv_e += (t_act - t_cv_e)** 2
        
    #     # t_cv_c = cv_cubic(A, V, r_cache, K, v1, v2)
    #     # err_cv_c += (t_act - t_cv_c) ** 2

    #     # t_srp = srp(V, K, v1, v2)
    #     # err_srp += (t_act - t_srp) ** 2
        # calc_one_pair(pair)

    print(err_cv_o / total)
    # print(err_cv_e / total)
    print(err_cv_c / total)
    print(err_srp / total)

if __name__ == "__main__":
    main()
