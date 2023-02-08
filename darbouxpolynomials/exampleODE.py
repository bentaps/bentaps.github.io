import numpy as np

def fx(x, y):
    return x**2 + 2 * x * y + 3 * y**2


def fy(x, y):
    return 2 * y * (2 * x + y)


def f(X):
    return np.array([fx(X[0], X[1]), fy(X[0], X[1])])


def p(x, y):
    return x + y


def p_ls(x):  # zero level set
    return -x


def q(x, y):
    return x - y


def q_ls(x):
    return x


def r(x, y):
    return y


def r_ls(x):
    return np.full(np.shape(x), 0)


def H(x, y):
    return p(x, y) * q(x, y) ** 3 * r(x, y) ** -1