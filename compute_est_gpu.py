import torch
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

import pickle

pi = np.pi

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# mnist test dataset (10000 * 784)
def load_mnist():
    with open('data/mnist.pkl','rb') as f:
        mnist = pickle.load(f)
        return mnist['test_images'], mnist['test_images'].shape

# gisette dataset (13500 * 5000)
def load_gisette():
    with open('data/gisette.npy', 'rb') as f:
        data = np.load(f)
        return data, data.shape

'''SETUP'''
def setup(vecs, K):
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

    final = A_est + c1 * (B_est - u_b) + c2 * (C_est - u_c)
    final = pi * (1 - final)

    final[final != final] = 0
    final[torch.isinf(final)] = 0

    return final


def cv_cubic(A, V, r_cache, K, v1, v2, NR_iter=5):
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

    theta_ij[theta_ij != theta_ij] = 0
    theta_ij[torch.isinf(theta_ij)] = 0

    return theta_ij


def mle_cubic(A, V, r, K, v1, v2, NR_iter=100):
    n1 = ((V[v1] != V[v2]) * (V[v2] == r)).sum(dim=1)
    n2 = ((V[v1] == V[v2]) * (V[v1] != r)).sum(dim=1)
    n3 = ((V[v1] == V[v2]) * (V[v1] == r)).sum(dim=1)
    n4 = ((V[v1] != V[v2]) * (V[v1] == r)).sum(dim=1)

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for p
    p3_orig = torch.true_divide(n3, K).double()
    p3 = torch.true_divide(n3, K).double()

    # find the theta_ij that solves the cubic
    # def f(p):
    #     return pi*n1*p*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1))
    
    # def fp(p):
    #     return pi*(pi*n1*p*(theta_i1 + pi*(p - 1)) + pi*n1*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n1*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n2*p*(theta_i1 + pi*(p - 1)) + pi*n2*p*(theta_j1 + pi*(p - 1)) + n2*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_j1 + pi*(p - 1)) + n3*(theta_i1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + n3*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_j1 + pi*(p - 1)) + pi*n4*p*(theta_i1 + theta_j1 + pi*(p - 1)) + n4*(theta_j1 + pi*(p - 1))*(theta_i1 + theta_j1 + pi*(p - 1)))

    def f(x):
        p1 = n1 * pi / (theta_j1 + pi * (x - 1))
        p2 = n4 * pi / (theta_i1 + pi * (x - 1))
        p3 = n2 * pi / (theta_i1 + theta_j1 + pi * (x - 1))
        p4 = n3 / x
        return p1 + p2 + p3 + p4

    def fp(x):
        p1 = n1 * pi * pi / (theta_j1 + pi * (x - 1)) ** 2
        p2 = n4 * pi * pi / (theta_i1 + pi * (x - 1)) ** 2
        p3 = n2 * pi * pi / (theta_i1 + theta_j1 + pi * (x - 1)) ** 2
        p4 = n3 / x ** 2
        return - p1 - p2 - p3 - p4


    # # use newton-raphson to find the root
    for _ in range(NR_iter):
        p3 = p3 - f(p3) / fp(p3)

    # clamp to 0 and 1
    p3[p3 != p3] = p3_orig[p3 != p3]
    # p3[p3 > 1] = 1
    # p3[p3 < 0] = 0

    p2 = p3 - 1 + (theta_i1 + theta_j1) / pi

    p2_p3 = p2 + p3
    p2_p3[p2_p3 > 1] = 1
    p2_p3[p2_p3 < 0] = 0

    theta_ij = pi * (1 - p2_p3)

    # theta_ij[theta_ij != theta_ij] = 0
    # theta_ij[torch.isinf(theta_ij)] = 0

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


def sim_one_K(K, mnist, iters, total):
    X, _, _, A, V, r = setup(mnist, K)

    r_cache = (V == r).sum(dim=1)

    err_cv_o = 0
    err_cv_c = 0
    err_mle = 0
    err_srp = 0

    for bat in tqdm(batch(iters, total, BATCH_SIZE), total=total // BATCH_SIZE):
        t_bat = bat.T

        dots = torch.bmm(X[t_bat[0]].view(BATCH_SIZE, 1, -1), X[t_bat[1]].view(BATCH_SIZE, -1, 1)).flatten()
        t_act = torch.acos(dots)

        t_cv_o = cv_orig(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_o += ((t_act - t_cv_o) ** 2).sum()
        
        t_cv_c = cv_cubic(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_c += ((t_act - t_cv_c) ** 2).sum()

        t_mle = mle_cubic(A, V, r, K, t_bat[0], t_bat[1])
        err_mle += ((t_act - t_mle) ** 2).sum()

        t_srp = srp(V, K, t_bat[0], t_bat[1])
        err_srp += ((t_act - t_srp) ** 2).sum()

    mse_cv_o = (err_cv_o.item() / total)
    mse_cv_c = (err_cv_c.item() / total)
    mse_mle = (err_mle.item() / total)
    mse_srp = (err_srp.item() / total)

    return mse_cv_o, mse_cv_c, mse_mle, mse_srp


def get_iterator(N, get_all=True):
    if get_all:
        # every possible pair
        total = (N * (N - 1) // 2)
        iters = torch.combinations(torch.from_numpy(np.arange(N)), 2).to(DEVICE).long()
    else:
        total = 10000
        iters = torch.from_numpy(np.random.randint(N, size=(total, 2))).to(DEVICE).long()

    return total, iters

# constants that should be manually changed
SEED = 1356
CALC_ALL = False
K_REPS = 10

if CALC_ALL:
    BATCH_SIZE = 101000
else:
    BATCH_SIZE = 10000

def main():
    # reset the seed
    np.random.seed(SEED)

    print(f'Using device: {DEVICE}')

    # test this for the mnist test dataset first
    mnist, (N, _) = load_mnist()
    total, iters = get_iterator(N, get_all=CALC_ALL)

    # set the size of dimensions to check
    Ks = list(range(10, 101, 10))

    with open('out.csv', 'w') as f:
        f.write('K,iter,CV_SUB,CV_CUBIC,MLE,SRP\n')
        for K in Ks:
            print(f'K = {K}')
            for i_num in range(K_REPS):
                errs = sim_one_K(K, mnist, iters, total)
                f.write(f'{K},{i_num},{",".join(str(e) for e in errs)}\n')


if __name__ == "__main__":
    main()