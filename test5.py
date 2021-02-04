import torch
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

import pickle
import random
import itertools as it

pi = np.pi

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

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

    X = torch.from_numpy(X).to(DEVICE)
    A = torch.from_numpy(A).to(DEVICE)
    V = torch.from_numpy(V).to(DEVICE)
    r = torch.from_numpy(r).to(DEVICE)

    return X, e, R, A, V, r

'''PROCEDURES'''
def cv_orig(A, V, r_cache, K, v1, v2):
    # then, for any two vectors (indexed as v1 and v2)
    # calc A = sum(V[v1] == V[v2]) / K
    # B = sum(V[v1] == r) / K
    # C = sum(V[v2] == r) / K

    A_est = torch.true_divide((V[v1] == V[v2]).sum(dim=1), K)
    B_est = torch.true_divide(r_cache[v1], K)
    C_est = torch.true_divide(r_cache[v2], K)

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for theta_ij = pi * (1 - A)
    theta_ij = pi * (1 - A_est)

    # calculate the control variate coefficients as below:    
    c1 = (-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)
    c2 = (2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)/(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)
    
    # calculate the true mean values for CV
    u_b = 1 - torch.true_divide(theta_i1, pi)
    u_c = 1 - torch.true_divide(theta_j1, pi)

    # three = 1 - (theta_ij + theta_i1 + theta_j1) / (2 * pi)
    # cov_ab = three - A_est * u_b
    # cov_ac = three - A_est * u_c
    # cov_bc = three - u_b * u_c
    # var_b = u_b * (1 - u_b)
    # var_c = u_c * (1 - u_c)

    # cm = torch.stack((var_b, cov_bc, cov_bc, var_c), 0).view(2, 2)
    # ys = torch.stack((cov_ab, cov_ac), 0).view(2, 1)
    # cs = (-1 * torch.bmm(torch.inverse(cm), ys)).view(2)
    # final = A_est + cs[0] * (B_est - u_b) + cs[1] * (C_est - u_c)

    final = A_est + c1 * (B_est - u_b) + c2 * (C_est - u_c)

    # hence calculate the angle
    return pi * (1 - final)

def cv_cubic(A, V, r_cache, K, v1, v2, NR_iter=10):
    # then, for any two vectors (indexed as v1 and v2)
    # calc A = sum(V[v1] == V[v2]) / K
    # B = sum(V[v1] == r) / K
    # C = sum(V[v2] == r) / K
    A_est = torch.true_divide((V[v1] == V[v2]).sum(dim=1), K)
    B_est = torch.true_divide(r_cache[v1], K)
    C_est = torch.true_divide(r_cache[v2], K)

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for theta_ij = pi * (1 - A)
    theta_ij = pi * (1 - A_est)

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
    dist = (V[v1] != V[v2]).sum(dim=1)
    
    # then, the estimate is given by dist * pi / K
    return dist * pi / K


def batch(iterable, total, batch_size=1):
    for ndx in range(0, total, batch_size):
        yield iterable[ndx:min(ndx + batch_size, total)]

# def batch(iterable, total, batch_size=1):
#     args = [iter(iterable)] * batch_size
#     return it.zip_longest(*args)

SEED = 100

def main():
    # test this for the mnist test dataset first
    mnist, (N, _) = load_mnist()
    K = 2048
    X, _, _, A, V, r = setup(mnist, K, SEED)

    print("Caching counts with respect to extra vector (simulation speed up)")
    r_cache = []
    for v in tqdm(range(N)):
        r_cache.append((V[v] == r).sum())
    r_cache = torch.tensor(r_cache).to(DEVICE)

    err_cv_o = 0
    err_cv_c = 0
    err_srp = 0

    # every possible pair
    total = (N * (N - 1) // 2)
    iters = torch.combinations(torch.tensor(list(range(N))).to(DEVICE), 2).to(DEVICE)

    # take a subset
    # total = 10000
    # iters = torch.from_numpy(np.random.randint(N, size=(total, 2))).to(DEVICE).long()

    batch_size = 5000

    for bat in tqdm(batch(iters, total, batch_size), total=total // batch_size):
        t_bat = bat.T

        dots = torch.bmm(X[t_bat[0]].view(batch_size, 1, -1), X[t_bat[1]].view(batch_size, -1, 1)).flatten()
        t_act = torch.acos(dots)

        t_cv_o = cv_orig(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_o += ((t_act - t_cv_o) ** 2).sum()
        
        t_cv_c = cv_cubic(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_c += ((t_act - t_cv_c)** 2).sum()

        t_srp = srp(V, K, t_bat[0], t_bat[1])
        err_srp += ((t_act - t_srp) ** 2).sum()


    # for (v1, v2) in tqdm(iters, total=total):
    # for (v1, v2) in iters:
        # calculate the actual angle
        # t_act = torch.acos(torch.dot(X[v1], X[v2]))

        # obtain the estimates using the various procedures
        # calculate the errors associated with each estimate
        # t_cv_o = cv_orig(A, V, r_cache, K, v1, v2)
        # err_cv_o += (t_act - t_cv_o) ** 2
        # break
        
        # t_cv_c = cv_cubic(A, V, r_cache, K, v1, v2)
        # err_cv_c += (t_act - t_cv_c) ** 2

        # t_srp = srp(V, K, v1, v2)
        # err_srp += (t_act - t_srp) ** 2
    
    print(err_cv_o.item() / total)
    print(err_cv_c.item() / total)
    print(err_srp.item() / total)


if __name__ == "__main__":
    main()