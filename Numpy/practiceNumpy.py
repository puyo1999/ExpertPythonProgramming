import numpy as np

x = np.array([1,2,3])
print(x.__class__)
print(x.shape)
print('x:{}'.format(x))

np.array([[1,2], [2,3,4]])

a = np.array([1,2,3])
b = np.array([4,5,6])

a = b
a[...] = b