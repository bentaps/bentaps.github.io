import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
import random
import os


def quiver(fx, fy, lim=1, n=20):
    range = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(range, range)
    Fx, Fy = fx(X, Y), fy(X, Y)
    R = np.power(Fx**2 + Fy**2, 1 / 2)
    plt.quiver(X, Y, Fx / R, Fy / R, alpha=0.5)


def forward_euler(x, f, h):
    return x + h * f(x)


def backward_euler_sho(x, f, h):
    A = np.array([[1, h], [-h, 1]], dtype=np.float32)
    return np.linalg.inv(A) @ x


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


def random_grid(lim, gridpoints, nsols):
    """returns an nsols by 2 array with random points on taken from a gridpoints by gridpoints grid"""
    X = np.linspace(-lim, lim, gridpoints)
    x0, y0 = np.meshgrid(X, X)
    return np.array(
        [
            random.choices(x0.flatten(), k=nsols),
            random.choices(y0.flatten(), k=nsols),
        ]
    ).T


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


def gif_centrifuge(
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
    nsol = len(initial_conditions)
    lines = [None] * nsol
    print(len(lines))
    if not os.path.exists(dir):
        os.makedirs(dir)
    x = np.empty((2, nt, nsol))

    assert len(integrator) == len(
        marker
    ), "Length of intergrator must be equal to length of markers"

    nmeth = len(integrator)

    for i, x0 in enumerate(initial_conditions):
        x[:, :, i] = integrate(integrator[i % nmeth], x0, f, nt=nt, h=h)
    with writer.saving(fig, filename, dpi=dpi):
        for it in range(nt):
            if it % frame_res == 0:
                for k in range(nsol):
                    lines[k] = plt.plot(x[0, it, k], x[1, it, k], marker[k % nmeth])
                writer.grab_frame()

                [line.pop(0).remove() for line in lines]


def gif_particles(
    fig,
    filename,
    X,
    imeth,
    dpi=100,
    writer=None,
    marker="bo",
    plane="x-y",
    fps=30,
):
    if writer is None:
        writer=PillowWriter(fps)
    PLANES = {
        "x-y": [0, 1],
        "y-z": [1, 2],
        "z-x": [2, 0],
    }
    PLANES[plane][0]
    """creates a gif of numerical solutions to f"""
    dir = os.path.dirname(filename)
    nt, _, nm, nparticles = X.shape
    lines = [None] * nparticles

    plt.xlim(X[:, PLANES[plane][0], :, :].min(), X[:, PLANES[plane][0], :, :].max())
    plt.ylim(X[:, PLANES[plane][1], :, :].min(), X[:, PLANES[plane][1], :, :].max())

    with writer.saving(fig, filename, dpi=dpi):
        for it in range(nt):
            plt.cla()
            plt.scatter(
                X[it, PLANES[plane][0], 0, :],
                X[it, PLANES[plane][1], 0, :],
                marker="o",
                alpha=0.3,
            )  # re
            plt.scatter(
                X[it, PLANES[plane][0], imeth, :],
                X[it, PLANES[plane][1], imeth, :],
                marker="o",
                alpha=0.3,
            )  # ref
            # plt.scatter(X[it, PLANES[plane][0], imeth, :], X[it, PLANES[plane][1], imeth, :], marker='x')
            writer.grab_frame()
