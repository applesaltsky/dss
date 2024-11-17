from pathlib import Path
import sys

PATH_PY = Path(__file__)
PATH_PROJECT = PATH_PY.parent.parent
sys.path.append(PATH_PROJECT.__str__())

import matplotlib.pyplot as plt
import numpy as np
import util

f = lambda x : 0.4 * (x**2) + 8 * x - 1
#print(util.trainer.numerical_diff(f,3))
#print(util.trainer.numerical_diff(f,6))

f2 = lambda x :(x[0]**2) + (x[1]**2)
#print(util.trainer.numerical_gradient(f2, np.array([2,3],dtype=float)))

x_arr = np.arange(-2, 2.5, 0.5)
y_arr = np.arange(-2, 2.5, 0.5)
x_p = []
y_p = []
x_v = []
y_v = []
for _x_p in x_arr:
    for _y_p in y_arr:
        _x_v, _y_v = util.trainer.numerical_gradient(f2, np.array([_x_p,_y_p],dtype=float))
        x_p.append(_x_p)
        y_p.append(_y_p)
        x_v.append(_x_v)
        y_v.append(_y_v)


plt.quiver(x_p,y_p,x_v,y_v)
plt.show()