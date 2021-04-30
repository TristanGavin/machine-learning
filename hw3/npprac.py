import numpy as np

x1 = np.arange(9.0).reshape((3,3))
print(x1)
x2 = np.arange(3.0)
print(x2)

print(x1.shape, x2.shape)

print(np.multiply(x1,x2))
