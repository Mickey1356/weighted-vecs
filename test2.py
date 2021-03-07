import numpy as np
import pickle

# st = np.random.get_state()
# print(np.random.normal(size=(2,2)))
# pickle.dump(st, open('o.out', 'wb'))

st = pickle.load(open('o.out', 'rb'))
np.random.set_state(st)
print(np.random.normal(size=(2,2)))
