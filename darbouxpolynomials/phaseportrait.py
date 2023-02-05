# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import os
import sys
from matplotlib.animation import PillowWriter
import random

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

alpha = 0.0


def fx(x, y):
    return x**2 + 2 * x * y + 3 * y ** 2


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
    return y + alpha


def r_ls(x):
    return np.full(np.shape(x), -alpha)


def H(x, y):
    return p(x, y) * q(x, y) ** 3 * r(x, y) ** -1


h = 0.05
nsols = 100
lim = 1
nt = 50
X = np.linspace(-lim, lim, 15)
x0, y0 = np.meshgrid(X, X)
initial_conditions = np.array([
    random.choices(x0.flatten(), k=nsols),
    random.choices(y0.flatten(), k=nsols),
]).T

fig = plt.figure()
plt.xlim([-lim, lim])
plt.ylim([-lim, lim])


gifname = "PhaseLines"
writer = PillowWriter(fps=30)
with writer.saving(fig, f"{sys.path[0]}/GIFs/{gifname}.gif", 100):
    for i, x0 in enumerate(initial_conditions):
        print(f"{i+1}/{nsols}")
        x = ode.integrate(ode.forward_euler, x0, f, nt=nt, h=h)
        for it in range(len(x[0, :])-1):
            plt.plot(x[0, 0], x[1, 0], "ko")
            plt.plot(x[0, :it], x[1, :it], "b-")
            if (
                any(abs(x[:, it]) > lim)
                or abs(np.linalg.norm(x[:, it+1]) - np.linalg.norm(x[:, it])) < 0.01*2*lim/100
            ):
                break
            if it % 2 == 0:
                writer.grab_frame()

gifname = "PhaseLines2"
with writer.saving(fig, f"{sys.path[0]}/GIFs/{gifname}.gif", 100):
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, p_ls(X), 'k--', linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, q_ls(X), 'k--', linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, r_ls(X), 'k--', linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    ode.get_quiver(fx, fy, lim=lim)
    [writer.grab_frame() for _ in range(130)]


# n1 = 16
# range1 = np.linspace(-5, 5, n1)
# X1, Y1 = np.meshgrid(range1, range1)
# Fx, Fy = fx(X1, Y1), fy(X1, Y1)
# R = np.power(Fx**2 + Fy**2, 1 / 2)
# plt.quiver(X1, Y1, Fx / R, Fy / R, alpha=0.5)

# n2 = 64
# range2 = np.linspace(-1, 1, n2)
# X2, Y2 = np.meshgrid(range2, range2)
# Z1 = np.power(H(X2, Y2), 1/2)
# Z2 = -np.power(-H(X2, Y2), 1/2)

# plt.imshow(Z1, cmap=cm.coolwarm)
# plt.imshow(Z2, cmap=cm.coolwarm)
# # # plt.quiver(X1, Y1, np.power(np.abs(Fx), 1/4)*Fx, np.power(np.abs(Fy), 1/4)*Fy, edgecolor="k", facecolor="None", linewidth=0.5)
# plt.show()
