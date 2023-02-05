import matplotlib.pyplot as plt
import numpy as np



def get_quiver(fx, fy, lim=1, n=20):
    range = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(range, range)
    Fx, Fy = fx(X, Y), fy(X, Y)
    R = np.power(Fx**2 + Fy**2, 1 / 2)
    plt.quiver(X, Y, Fx / R, Fy / R, alpha=0.5)
    

def forward_euler(x, f, h):
    return x + h*f(x)

def integrate(phi, x0, f, nt=100, h=0.1):
    trj = np.empty((len(x0), nt), np.float16)
    trj[:, 0] = x0
    for it in range(nt-1):
        trj[:, it+1] = phi(trj[:, it], f, h)
    return trj 



