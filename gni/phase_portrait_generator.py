# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def fx(x, y):
    return y


def fy(x, y):
    return -x

def f(x):
    return np.array([fx(x[0], x[1]), fy(x[0], x[1])])

def H(x, y):
    return x**2 + y**2


def get_quiver(fx, fy, n, lims):
    range = np.linspace(-lims, lims, n)
    X, Y = np.meshgrid(range, range)
    Fx, Fy = fx(X, Y), fy(X, Y)
    R = np.power(Fx**2 + Fy**2, 1 / 2)
    plt.quiver(X, Y, Fx / R, Fy / R, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("v")

# plt.show()

def forward_euler(x, f, h):
    return x + h*f(x)

x0 = np.array([1/2,0])
h = 0.05
nt = 200

def numerical_trajectory(phi, x0, f, nt=100, h=0.1):
    trj = np.empty((len(x0), nt), np.float16)
    trj[:, 0] = x0
    for it in range(nt-1):
        trj[:, it+1] = phi(trj[:, it], f, h)
    return trj 

x = numerical_trajectory(forward_euler, x0, f, nt, h)

# %%
get_quiver(fx, fy, 16, 1)
# plt.savefig("gni/pendulum")
plt.plot(x[0, :], x[1, :], 'r-')
plt.show()

# %%


