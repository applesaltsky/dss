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


init_x = np.array([-3.,4.])
w = util.trainer().gradient_descent(f2, init_x, lr=0.05, step_num = 100)
x_arr = [x for x,y in w]
y_arr = [y for x,y in w]
plt.scatter(x_arr,y_arr)


x_t = []
y_t = []
for r in range(5):
    for theta in np.arange(0,360,step=30/(r+1)):
        x_p = np.sin(theta * np.pi / 180) * r
        x_t.append(x_p)
        y_t.append(np.sqrt(r ** 2 - x_p ** 2))

        x_t.append(x_p)
        y_t.append(-1*np.sqrt(r ** 2 - x_p ** 2))

plt.scatter(x_t, y_t,s=1,c='gray',marker=',')
plt.show()