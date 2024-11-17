from pathlib import Path
import sys
PATH_PY = Path(__file__)
PATH_PROJECT = PATH_PY.parent.parent
sys.path.append(PATH_PROJECT.__str__())

import util

import numpy as np

X, y = util.datasets.load_mnist()
print(y[0])
#X.shape  (70000,784)
#y.shape  (70000,)

#from PIL import Image
#img = Image.fromarray(np.uint8(X[4]).reshape(28,28))
#img.show()


from matplotlib import pyplot as plt
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i].reshape(28,28))
plt.show()