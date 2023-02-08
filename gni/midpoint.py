import numpy as np
import scipy


def fun(x):
    return x**2 - 2


x0 = 1


def midpoint(f, x, h):
    return scipy.optimize.root(
        lambda x1: - x1 + x + h * f(x1 / 2 + x / 2), x0, method="hybr", tol=1e-8
    ).x

def f(x):
    return np.array([x[0]+1,x[1]+1])

midpoint(f, np.array([1,1]), 0.3)