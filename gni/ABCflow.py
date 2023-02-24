import numpy as np

A = 0.9
B = 0.4
C = 0.15


def fx(x, y, z):
    return C * np.sin(y) + B * np.cos(z)


def fy(x, y, z):
    return B * np.sin(z) + A * np.cos(x)


def fz(x, y, z):
    return A * np.sin(x) + C * np.cos(y)


def f(x):
    return np.array([fx(x[0], x[1], x[2]), fx(x[0], x[1], x[2]), fz(x[0], x[1], x[2])])
