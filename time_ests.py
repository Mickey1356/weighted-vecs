import torch
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

import pickle
from timeit import default_timer as timer

pi = np.pi

# if torch.cuda.is_available():
#     DEVICE = 'cuda'
# else:
#     DEVICE = 'cpu'
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

def setup(vecs, K):
    _, P = vecs.shape

    # open a dataset of vectors of size (N * P) (X)
    # normalise all of them (row-wise)
    X = normalize(vecs, norm='l2')

    # generate a row vector with P elements E
    # random vector
    # e = np.random.standard_normal(size=(1, P))
    # mean of dataset
    e = np.mean(X, axis=0).reshape(1, -1)
    # first singular vector
    # _, _, u = np.linalg.svd(X, full_matrices=False)
    # e = u[0, :].reshape(1, -1)
    # normalise it (for consistency)
    e = normalize(e, norm='l2')

    # store every N angles (against the random vector E) in an array A
    dots = np.matmul(X, e.T)
    A = np.arccos(dots).flatten()

    # generate a random matrix of size (P * K) (where K can be varied) (R)
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

    # set out of bounds values to be their original estimates
    # final[final < 0] = theta_ij[final < 0].double()
    # final[final > pi] = theta_ij[final > pi].double()

    # # set nan values to be their original estimates
    # final[final != final] = theta_ij[final != final].double()
    # final[torch.isinf(final)] = theta_ij[torch.isinf(final)].double()

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
    # theta_ij_orig = pi * (1 - A_est)
    theta_ij = pi * (1 - A_est)

    # find the theta_ij that solves the cubic
    def f(theta_ij):
        return (theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A_est - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B_est - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C_est - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2))
    
    def fp(theta_ij):
        return (-2*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1)*(theta_ij*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + pi*(A_est - 1)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2) + (theta_i1 + pi*(B_est - 1))*(-2*theta_i1**2*theta_j1 + pi*theta_i1**2 - 2*pi*theta_i1*theta_ij + 2*pi*theta_i1*theta_j1 - 2*theta_ij**2*theta_j1 + pi*theta_ij**2 + 2*pi*theta_ij*theta_j1 + 2*theta_j1**3 - 3*pi*theta_j1**2) + (theta_j1 + pi*(C_est - 1))*(2*theta_i1**3 - 3*pi*theta_i1**2 - 2*theta_i1*theta_ij**2 + 2*pi*theta_i1*theta_ij - 2*theta_i1*theta_j1**2 + 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)) + (pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + 2*theta_ij*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) + pi*theta_j1**2 + 2*pi*(A_est - 1)*(2*theta_i1*theta_j1 - pi*theta_i1 + pi*theta_ij - pi*theta_j1) - 2*(theta_i1 + pi*(B_est - 1))*(pi*theta_i1 + 2*theta_ij*theta_j1 - pi*theta_ij - pi*theta_j1) - 2*(theta_j1 + pi*(C_est - 1))*(2*theta_i1*theta_ij - pi*theta_i1 - pi*theta_ij + pi*theta_j1)))/(pi*(pi*theta_i1**2 + 4*theta_i1*theta_ij*theta_j1 - 2*pi*theta_i1*theta_ij - 2*pi*theta_i1*theta_j1 + pi*theta_ij**2 - 2*pi*theta_ij*theta_j1 + pi*theta_j1**2)**2)

    # use newton-raphson to find the root
    for _ in range(NR_iter):
        theta_ij = theta_ij - f(theta_ij) / fp(theta_ij)

    # set out of bounds values to be their original estimates
    # theta_ij[theta_ij < 0] = theta_ij_orig[theta_ij < 0].double()
    # theta_ij[theta_ij > pi] = theta_ij_orig[theta_ij > pi].double()

    # # set nan values to be their original estimates
    # theta_ij[theta_ij != theta_ij] = theta_ij_orig[theta_ij != theta_ij].double()
    # theta_ij[torch.isinf(theta_ij)] = theta_ij_orig[torch.isinf(theta_ij)].double()

    return theta_ij


def mle_cubic(A, V, r, K, v1, v2, NR_iter=50):
    n1 = ((V[v1] != V[v2]) * (V[v2] == r)).sum(dim=1)
    n2 = ((V[v1] == V[v2]) * (V[v1] != r)).sum(dim=1)
    n3 = ((V[v1] == V[v2]) * (V[v1] == r)).sum(dim=1)
    n4 = ((V[v1] != V[v2]) * (V[v1] == r)).sum(dim=1)

    # theta_i1 = ANGLES[v1], theta_j1 = ANGLES[v2]
    theta_i1 = A[v1]
    theta_j1 = A[v2]

    # we need an initial estimate for p
    # p3_orig = torch.true_divide(n3, K).double()
    p3 = torch.true_divide(n3, K).double()

    # adapted from matlab code
    p_i1 = 1 - theta_i1 / pi
    p_j1 = 1 - theta_j1 / pi

    a3 = n1 + n2 + n3 + n4
    a2 = n3 + n4 - n2 * p_i1 - 2 * n3 * p_i1 - n4 * p_i1 - n2 * p_j1 - 2 * n3 * p_j1 - 2 * n4 * p_j1 - n1 * (-1 + 2 * p_i1 + p_j1)
    a1 = n1 * p_i1 * (-1 + p_i1 + p_j1) + p_j1 * (n2 * p_i1 + n4 * (-1 + p_i1 + p_j1)) + n3 * (p_i1 ** 2 + (-1 + p_j1) * p_j1 + p_i1 * (-1 + 3 * p_j1))
    a0 = -n3 * p_i1 * p_j1 * (-1 + p_i1 + p_j1)

    def f(p):
        return a3 * (p ** 3) + a2 * (p ** 2) + a1 * p + a0

    def fp(p):
        return 3 * a3 * (p ** 2) + 2 * a2 * p + a1

    # def f(x):
    #     p1 = n1 * pi / (theta_j1 + pi * (x - 1))
    #     p2 = n4 * pi / (theta_i1 + pi * (x - 1))
    #     p3 = n2 * pi / (theta_i1 + theta_j1 + pi * (x - 1))
    #     p4 = n3 / x
    #     return p1 + p2 + p3 + p4

    # def fp(x):
    #     p1 = n1 * pi * pi / (theta_j1 + pi * (x - 1)) ** 2
    #     p2 = n4 * pi * pi / (theta_i1 + pi * (x - 1)) ** 2
    #     p3 = n2 * pi * pi / (theta_i1 + theta_j1 + pi * (x - 1)) ** 2
    #     p4 = n3 / x ** 2
    #     return - p1 - p2 - p3 - p4

    # # use newton-raphson to find the root
    for _ in range(NR_iter):
        p3 = p3 - f(p3) / fp(p3)

    # clamp to 0 and 1
    # p3[p3 != p3] = p3_orig[p3 != p3]
    # p3[p3 > 1] = p3_orig[p3 > 1]
    # p3[p3 < 0] = p3_orig[p3 < 0]

    p2 = p3 - 1 + (theta_i1 + theta_j1) / pi

    p2_p3 = p2 + p3
    # p2_p3[p2_p3 > 1] = 1
    # p2_p3[p2_p3 < 0] = 0

    theta_ij = pi * (1 - p2_p3)

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


def sim_one_K(K, data, iters, total, batch_size):
    X, _, _, A, V, r = setup(data, K)

    r_cache = (V == r).sum(dim=1)

    err_cv_o = 0
    err_cv_c = 0
    err_mle = 0
    err_srp = 0

    global time_srp
    global time_mle
    global time_cv_cubic
    global time_cv_sub
    global time_calc

    for bat in tqdm(batch(iters, total, batch_size), total=int(np.ceil(total / batch_size))):
        length = len(bat)
        t_bat = bat.T

        s = timer()
        dots = torch.bmm(X[t_bat[0]].view(length, 1, -1), X[t_bat[1]].view(length, -1, 1)).flatten()
        t_act = torch.acos(dots)
        e = timer()
        time_calc += (e - s)

        s = timer()
        t_cv_o = cv_orig(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_o += ((t_act - t_cv_o) ** 2).sum()
        e = timer()
        time_cv_sub += (e - s)

        s = timer()
        t_cv_c = cv_cubic(A, V, r_cache, K, t_bat[0], t_bat[1])
        err_cv_c += ((t_act - t_cv_c) ** 2).sum()
        e = timer()
        time_cv_cubic += (e - s)

        s = timer()
        t_mle = mle_cubic(A, V, r, K, t_bat[0], t_bat[1])
        err_mle += ((t_act - t_mle) ** 2).sum()
        e = timer()
        time_mle += (e - s)

        s = timer()
        t_srp = srp(V, K, t_bat[0], t_bat[1])
        err_srp += ((t_act - t_srp)** 2).sum()
        e = timer()
        time_srp += (e - s)

        
    mse_cv_o = (err_cv_o.item() / total)
    mse_cv_c = (err_cv_c.item() / total)
    mse_mle = (err_mle.item() / total)
    mse_srp = (err_srp.item() / total)

    mse_cv_o = (err_cv_o / total)
    mse_cv_c = (err_cv_c / total)
    mse_mle = (err_mle / total)
    mse_srp = (err_srp / total)

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
CALC_ALL = True
K_REPS = 5

time_srp = 0
time_mle = 0
time_cv_cubic = 0
time_cv_sub = 0
time_calc = 0

def main():
    # reset the seed
    np.random.seed(SEED)

    print(f'Using device: {DEVICE}')

    # batch size depends on gpu memory
    # higher is better, to avoid the overhead from transferring from cpu to gpu as much as possible
    # for a 6gb gpu, mnist: 120000, gisette: 32000
    batch_size = 100000
    
    # set the size of dimensions to check
    # Ks = range(10, 101, 10)
    # Ks = range(800, 1001, 50)
    Ks = [50, 500, 1000]

    # mnist dataset
    data, (N, _) = load_mnist()
    
    # gisette dataset
    # data, (N, _) = load_gisette()
    
    total, iters = get_iterator(N, get_all=CALC_ALL)

    # load rng_state (if needed)
    # rng_s = pickle.load(open('rng_state.out', 'rb'))
    # np.random.set_state(rng_s)

    global time_srp
    global time_mle
    global time_cv_cubic
    global time_cv_sub
    global time_calc

    for K in Ks:
        print(f'K = {K}')
        time_srp = 0
        time_mle = 0
        time_cv_cubic = 0
        time_cv_sub = 0
        time_calc = 0

        for i_num in range(K_REPS):
            errs = sim_one_K(K, data, iters, total, batch_size)
        
        print('srp', time_srp)
        print('mle', time_mle)
        print('cv-cubic', time_cv_cubic)
        print('cv-sub', time_cv_sub)
        print('calc', time_calc)
    


    # if pausing: just save numpy rng state so can restore from this point in the future
    # pickle.dump(np.random.get_state(), open('rng_state.out', 'wb'))


if __name__ == "__main__":
    main()