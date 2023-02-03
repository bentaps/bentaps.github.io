import numpy as np
import matplotlib.pyplot as plt


def fx(x, y):
    return x**2 + 2 * x * y + 3 * y**2


def fy(x, y):
    return 2 * (y - 1 / 2) * (2 * x + y)

def f(X):
    return np.array([fx(X[0], X[1]), fy(X[0], X[1])])

def f(X):
    return np.array([-X[1], X[0]])

def p(x, y):
    return x + y


def q(x, y):
    return x - y


def r(x, y):
    return y - 1 / 2


def H(x, y):
    return p(x, y) * q(x, y) ** 3 * r(x, y) ** -1


def FE(X, f, h):
    return X + h*f(X)

def integrate_ode(f, integrator, X0, h, T):
    nt = round(T/h)
    X = np.empty((nt,2))
    X[0, :] = X0

    for it in range(nt-1):
        X[it+1, :] = integrator(X[it, :], f, h)
    return X[:,0], X[:,1]

h = 0.01
T = 2
X0 = [0, 1]
for _ in range(10):
    X0 = (np.random.rand(2)-0.5)/0.5
    x, y = integrate_ode(f, FE, X0, h, T)
    plt.plot(x, y)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()









# n1 = 16
# range1 = np.linspace(-5, 5, n1)
# X1, Y1 = np.meshgrid(range1, range1)
# Fx, Fy = fx(X1, Y1), fy(X1, Y1)
# R = np.power(Fx**2 + Fy**2, 1 / 2)
# plt.quiver(X1, Y1, Fx / R, Fy / R, alpha=0.5)

# # n2 = 64
# # range2 = np.linspace(-1, 1, n2)
# # X2, Y2 = np.meshgrid(range2, range2)
# # Z1 = np.power(H(X2, Y2), 1/2)
# # Z2 = -np.power(-H(X2, Y2), 1/2)

# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# # surf = ax.plot_surface(X2, Y2, Z1, cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # surf = ax.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm,
# #    linewidth=0, antialiased=False)
# # plt.quiver(X1, Y1, np.power(np.abs(Fx), 1/4)*Fx, np.power(np.abs(Fy), 1/4)*Fy, edgecolor="k", facecolor="None", linewidth=0.5)
# plt.show()
