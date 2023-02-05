# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import odeutils as ode

def fx(x, y):
    return y

def fy(x, y):
    return -x

def f(x):
    return np.array([fx(x[0], x[1]), fy(x[0], x[1])])

def H(x, y):
    return x**2 + y**2


x0 = np.array([1/2,0])
h = 0.05
nt = 200

x = ode.integrate(ode.forward_euler, x0, f, nt, h)

# %%
ode.get_quiver(fx, fy, 16, 1)
# plt.savefig("gni/pendulum")
plt.plot(x[0, :], x[1, :], 'r-')
plt.show()

# %%


