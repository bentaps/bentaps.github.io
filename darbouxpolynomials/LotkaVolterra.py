import numpy as np

B = 1
P = 1
R = 1
D = 1


def fx(x, y):
    return x * (B - P * y)


def fy(x, y):
    return y * (R * x - D)


def f(X):
    return np.array([fx(X[0], X[1]), fy(X[0], X[1])])


def p(x, y):
    return x


def p_ls(x):  # zero level set
    return np.full(np.shape(x), 0)


def q(x, y):
    return y


def q_ls(x):
    return 1e16 * x


def H(x, y):
    return B * np.log(y) - P * y - R * x + D * np.log(x)
