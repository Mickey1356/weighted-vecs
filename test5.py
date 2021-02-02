import torch
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

import pickle

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

SEED = 100

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


def main():
    # test this for the mnist test dataset first
    mnist, (N, _) = load_mnist()
    K = 256
    X, _, _, A, V, r = setup(mnist, K, SEED)

    print("Caching counts with respect to extra vector (simulation speed up)")
    r_cache = []
    for v in tqdm(range(N)):
        r_cache.append(sum(V[v] == r))


if __name__ == "__main__":
    main()