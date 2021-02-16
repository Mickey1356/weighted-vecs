import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

import pickle, random
import itertools as it

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

def cv_cubic(A, V, r_cache, K, v1, v2, NR_iter=10):
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

def mle_cubic(A, V, r, r_cache, K, v1, v2, NR_iter=20):
    A_est = sum(V[v1] == V[v2])
    B_est = r_cache[v1]
    C_est = r_cache[v2]

    n1 = sum(((V[v1] != V[v2]) * (V[v2] == r)))
    n2 = sum((V[v1] == V[v2]) * (V[v1] != r))
    n3 = sum((V[v1] == V[v2]) * (V[v1] == r))
    n4 = sum((V[v1] != V[v2]) * (V[v1] == r))

    # A (i==j): (n2 + n3) / k
    # B (i==1): (n3 + n4) / k
    # C (j==1): (n1 + n3) / k

    # print(n1, n2, n3, n4)
    # print(A_est, n2 + n3)
    # print(B_est, n3 + n4)
    # print(C_est, n1 + n3)
    # return

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    p_i1 = 1 - theta_i1 / pi
    p_j1 = 1 - theta_j1 / pi

    # we need an initial estimate for p
    p3 = n3 / K

    # adapted from matlab code
    # a3 = n1 + n2 + n3 + n4
    # a2 = n3 + n4 - n2 * p_i1 - 2 * n3 * p_i1 - n4 * p_i1 - n2 * p_j1 - 2 * n3 * p_j1 - 2 * n4 * p_j1 - n1 * (-1 + 2 * p_i1 + p_j1)
    # a1 = n1 * p_i1 * (-1 + p_i1 + p_j1) + p_j1 * (n2 * p_i1 + n4 * (-1 + p_i1 + p_j1)) + n3 * (p_i1 ** 2 + (-1 + p_j1) * p_j1 + p_i1 * (-1 + 3 * p_j1))
    # a0 = -n3 * p_i1 * p_j1 * (-1 + p_i1 + p_j1)

    # def f(p):
    #     return a3 * (p ** 3) + a2 * (p ** 2) + a1 * p + a0

    # def fp(p):
    #     return 3 * a3 * (p ** 2) + 2 * a2 * p + a1

    # find the theta_ij that solves the cubic
    # def f(p):
    #     return pi*n1*p*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1))
    
    # def fp(p):
    #     return pi*(pi*n1*p*(theta_i1 + pi*(p - 1)) + pi*n1*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n1*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1)) + pi*n2*p*(theta_j1 + pi*(p - 1)) + n2*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + n3*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n4*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)))

    def f(x):
        # p1 = n1 * np.pi * (angle_i1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1)) * x
        # p2 = n2 * np.pi * (angle_i1 + np.pi(x - 1)) * (angle_j1 + np.pi(x - 1)) * x
        # p3 = n3 * (angle_i1 + np.pi(x - 1)) * (angle_j1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1))
        # p4 = n4 * np.pi * (angle_j1 + np.pi(x - 1)) * (angle_i1 + angle_j1 + np.pi(x - 1)) * x

        p1 = n1 * np.pi / (theta_j1 + np.pi * (x - 1))
        p2 = n4 * np.pi / (theta_i1 + np.pi * (x - 1))
        p3 = n2 * np.pi / (theta_i1 + theta_j1 + np.pi * (x - 1))
        p4 = n3 / x
        return p1 + p2 + p3 + p4

    def fp(x):
        p1 = n1 * np.pi * np.pi / (theta_j1 + np.pi * (x - 1)) ** 2
        p2 = n4 * np.pi * np.pi / (theta_i1 + np.pi * (x - 1)) ** 2
        p3 = n2 * np.pi * np.pi / (theta_i1 + theta_j1 + np.pi * (x - 1)) ** 2
        p4 = n3 / x ** 2
        return - p1 - p2 - p3 - p4


    # # use newton-raphson to find the root
    for _ in range(NR_iter):
        p3 = p3 - f(p3) / fp(p3)

    if p3 > 1:
        p3 = 1
    if p3 < 0:
        p3 = 0
    if p3 != p3:
        p3 = 0
    # clamp to 0 and 1
    # p3[p3 != p3] = p3_orig[p3 != p3]
    # p3[p3 > 1] = 1
    # p3[p3 < 0] = 0

    p2 = p3 - 1 + (theta_i1 + theta_j1) / pi

    p2_p3 = p2 + p3
    # p2_p3[p2_p3 > 1] = 1
    # p2_p3[p2_p3 < 0] = 0

    theta_ij = pi * (1 - p2_p3)

    # theta_ij[theta_ij != theta_ij] = 0
    # theta_ij[torch.isinf(theta_ij)] = 0

    return theta_ij

def srp(V, K, v1, v2):
    # for any two vectors (indexed as v1, v2)
    # calculate the hamming distance between the two projections
    dist = sum(V[v1] != V[v2])
    
    # then, the estimate is given by dist * pi / K
    return dist * pi / K


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
K = 60

# def init_pool(X_s, A_s, V_s, r_s):
#     mp_globals.X = X_s
#     mp_globals.A = A_s
#     mp_globals.V = V_s
#     mp_globals.r_cache = r_s

def main():
    # test this for the mnist test dataset first
    mnist, (N, _) = load_mnist()
    X, _, _, A, V, r = setup(mnist, K, SEED)

    print("Caching counts with respect to extra vector (simulation speed up)")
    r_cache = []
    for v in tqdm(range(N)):
        r_cache.append(sum(V[v] == r))

    err_cv_o = 0
    err_cv_c = 0
    err_mle = 0
    err_srp = 0

    # every possible pair (gonna take a day)
    # total = (N * (N - 1) // 2)
    # iters = it.combinations(list(range(N)), 2)

    # take a subset
    total = 10000
    iters = np.random.randint(N, size=(total, 2))

    print("Calculating estimates for {} pairs".format(total))

    for (v1, v2) in tqdm(iters, total=total):
        # calculate the actual angle
        t_act = np.arccos(np.dot(X[v1], X[v2]))

        # obtain the estimates using the various procedures
        # calculate the errors associated with each estimate
        t_cv_o = cv_orig(A, V, r_cache, K, v1, v2)
        err_cv_o += (t_act - t_cv_o) ** 2
        
        t_cv_c = cv_cubic(A, V, r_cache, K, v1, v2)
        err_cv_c += (t_act - t_cv_c) ** 2
        
        t_mle = mle_cubic(A, V, r, r_cache, K, v1, v2)
        err_mle += (t_act - t_mle) ** 2

        t_srp = srp(V, K, v1, v2)
        err_srp += (t_act - t_srp) ** 2

    print(err_cv_o / total)
    print(err_cv_c / total)
    print(err_mle / total)
    print(err_srp / total)

if __name__ == "__main__":
    main()
