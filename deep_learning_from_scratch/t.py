import util
import numpy as np

a = np.ones((5,3,5))
b = np.ones((3,3))
print(b @ a)
print((b @ a).shape)

