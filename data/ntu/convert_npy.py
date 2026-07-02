import numpy as np

data = np.load('NTU60_CS.npz')

np.save('x_train.npy', data['x_train'])
np.save('y_train.npy', data['y_train'])
np.save('x_test.npy', data['x_test'])
np.save('y_test.npy', data['y_test'])