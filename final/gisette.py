import numpy as np
d1 = np.genfromtxt('gisette_train.data')
d2 = np.genfromtxt('gisette_valid.data')
d3 = np.genfromtxt('gisette_test.data')
data = np.concatenate((d1, d2, d3), axis=0)

with open('gisette.npy', 'wb') as f:
    np.save(f, data)