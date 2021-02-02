import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

import json, pickle
import itertools as it

def load_mnist():
    with open("data/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
        return mnist["test_images"], mnist["test_images"].shape

def main():
    mnist, (N, _) = load_mnist()
    X = normalize(mnist, norm='l2')
    a_dict = {}
    total = (N * (N - 1) // 2)
    iters = it.combinations(list(range(N)), 2)
    for (v1, v2) in tqdm(iters, total=total):
        angle = np.arccos(np.dot(X[v1], X[v2]))
        a_dict[(v1, v2)] = angle
    np.save('angles_dict.npy', a_dict)

main()