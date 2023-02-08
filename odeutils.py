import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
import os


def quiver(fx, fy, lim=1, n=20):
    range = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(range, range)
    Fx, Fy = fx(X, Y), fy(X, Y)
    R = np.power(Fx**2 + Fy**2, 1 / 2)
    plt.quiver(X, Y, Fx / R, Fy / R, alpha=0.5)


def forward_euler(x, f, h):
    return x + h * f(x)


def symplectic_euler(x, f, h):
    xstep1 = x + h * f(x)
    xstep2 = x + h * f([xstep1[0], x[1]])
    return np.array([xstep1[0], xstep2[1]]).reshape(np.shape(x))


def integrate(phi, x0, f, nt=100, h=0.1):
    trj = np.empty((len(x0), nt), np.float16)
    trj[:, 0] = x0
    for it in range(nt - 1):
        trj[:, it + 1] = phi(trj[:, it], f, h)
    return trj


def gif(
    fig,
    initial_conditions,
    f,
    nt,
    h,
    integrator,
    filename,
    lim=None,
    frame_res=2,
    dpi=100,
    fps=30,
    writer=PillowWriter(fps=30),
    marker="",
    linestyle="b-",
):
    """creates a gif of numerical solutions to f"""
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with writer.saving(fig, filename, dpi=dpi):
        for i, x0 in enumerate(initial_conditions):
            print(x0)
            print(f"{i+1}/{len(initial_conditions)}")
            x = integrate(integrator, x0, f, nt=nt, h=h)
            # print(np.shape(x), type(x), x.view())
            for it in range(len(x[0, :]) - 1):
                plt.plot(x[0, :it], x[1, :it], linestyle)
                plt.plot(x[0, 0], x[1, 0], "k*")
                l = plt.plot(x[0, it - 1], x[1, it - 1], marker)
                if lim:
                    if (
                        any(abs(x[:, it]) > lim)
                        or abs(np.linalg.norm(x[:, it + 1]) - np.linalg.norm(x[:, it]))
                        < 0.01 * 2 * lim / 100
                    ):
                        print("Warning: premeture loop break point reached")
                        break
                if it % frame_res == 0:
                    writer.grab_frame()
                line = l.pop(0)
                line.remove()
