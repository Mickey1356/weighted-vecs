import numpy as np
d1 = np.genfromtxt('data/gisette_train.data')
d2 = np.genfromtxt('data/gisette_valid.data')
d3 = np.genfromtxt('data/gisette_test.data')
data = np.concatenate((d1, d2, d3), axis=0)

with open('data/gisette.npy', 'wb') as f:
    np.save(f, data)