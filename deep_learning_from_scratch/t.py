import util
import numpy as np

a = np.ones((4,5))
b = np.ones((5,3))
print((a.reshape((4,1,5)) @ b).shape)