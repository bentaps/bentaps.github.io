import numpy as np
ALPHA = 1
def fx(x, y):
    return -ALPHA*y

def fy(x, y):
    return x

def f(x):
    return np.array([fx(x[0], x[1]), fy(x[0], x[1])])

def H(x, y):
    return x**2 + ALPHA*y**2