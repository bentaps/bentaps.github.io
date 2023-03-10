import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from matplotlib.animation import PillowWriter
import random
import os
from scipy.spatial.transform import Rotation


def quiver(fx, fy, lim=1, n=20):
    range = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(range, range)
    Fx, Fy = fx(X, Y), fy(X, Y)
    R = np.power(Fx**2 + Fy**2, 1 / 2)
    plt.quiver(X, Y, Fx / R, Fy / R, alpha=0.5)


def forward_euler(x, f, h):
    return x + h * f(x)


def R(axis, deg):
    """rotation matrix"""
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def rotate(initial_conditions, theta, phi, psi):
    initial_conditions = initial_conditions @ R("x", theta) @ R("y", phi) @ R("z", psi)


def volume_preserving(x, f, h):
    xstep1 = x + h * f(x)
    xstep2 = x + h * f([xstep1[0], x[1], x[2]])
    xstep3 = x + h * f([xstep1[0], xstep2[1], x[2]])
    return np.array([xstep1[0], xstep2[1], xstep3[2]]).reshape(np.shape(x))


def backward_euler_sho(x, f, h):
    A = np.array([[1, h], [-h, 1]], dtype=np.float32)
    return np.linalg.inv(A) @ x


def midpoint_sho(x, f, h):
    Ap = np.array([[1, h / 2], [-h / 2, 1]], dtype=np.float32)
    Am = np.array([[1, -h / 2], [h / 2, 1]], dtype=np.float32)
    return np.linalg.inv(Ap) @ Am @ x


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


def random_grid(lim, nsols, gridpoints=100):
    """returns an nsols by 2 array with random points on taken from a gridpoints by gridpoints grid"""
    X = np.linspace(-lim, lim, gridpoints)
    x0, y0 = np.meshgrid(X, X)
    return np.array(
        [
            random.choices(x0.flatten(), k=nsols),
            random.choices(y0.flatten(), k=nsols),
        ]
    ).T


def rand_cube(nsols, x0=None, width=1, stretch=(1, 1, 1), a=1):
    """returns a cube of width width, centered at x0 with nsols point in it"""
    # set a=0 to not center cube
    x0 = x0 if x0 else (0, 0, 0)
    out = (
        width
        * (
            np.array(
                [
                    (stretch[0] * (rand(1, nsols) - a * 1 / 2) + x0[0])[0],
                    (stretch[1] * (rand(1, nsols) - a * 1 / 2) + x0[1])[0],
                    (stretch[2] * (rand(1, nsols) - a * 1 / 2) + x0[2])[0],
                ]
            )
        ).T
    )
    return out


def ring(nsols, r=1, x0=None):
    """returns a cube of width width, centered at x0 with nsols point in it"""
    t = np.linspace(0, 2 * np.pi, num=nsols, endpoint=True)
    x, y, z = r * np.sin(t), r * np.cos(t), np.zeros(nsols)
    return np.array([x, y, z]).T + x0


def random_circle(radius, nsols, gridpoints=100):
    """returns an nsols by 2 array with random points within circle of radius R"""
    polarcoords = random_grid(lim=1, nsols=nsols, gridpoints=gridpoints)
    theta, r = 2 * np.pi * polarcoords[:, 0], radius * np.sqrt(
        np.abs(polarcoords[:, 1])
    )
    return np.array([r * np.cos(theta), r * np.sin(theta)]).T


def gif(
    fig,
    initial_conditions,
    f,
    nt,
    h,
    integrator,
    filename,
    lim=None,
    frame_res=4,
    dpi=100,
    fps=30,
    writer=PillowWriter(fps=30),
    marker="",
    linestyle="b-",
):
    """creates a gif of numerical solutions to f in 2d"""
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
                        # or abs(np.linalg.norm(x[:, it + 1]) - np.linalg.norm(x[:, it]))
                        # < 0.01 * 2 * lim / 100
                    ):
                        print("Warning: premeture loop break point reached")
                        break
                if it % frame_res == 0:
                    writer.grab_frame()
                line = l.pop(0)
                line.remove()


def gif3d(
    fig,
    initial_conditions,
    f,
    nt,
    h,
    integrator,
    filename,
    color="b",
    lim=None,
    frame_res=4,
    dpi=100,
    fps=30,
    writer=PillowWriter(fps=30),
    marker="",
):
    """creates a gif of numerical solutions to f in 3d"""
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    ax = fig.add_subplot(projection="3d")
    with writer.saving(fig, filename, dpi=dpi):
        for i, x0 in enumerate(initial_conditions):
            print(x0)
            print(f"{i+1}/{len(initial_conditions)}")
            x = integrate(integrator, x0, f, nt=nt, h=h)
            # print(np.shape(x), type(x), x.view())
            for it in range(len(x[0, :]) - 1):
                ax.view_init(azim=it / nt * 3.14159)
                plt.plot(x[0, :it], x[1, :it], x[2, :it], color=color, linestyle="-")
                plt.plot(x[0, 0], x[1, 0], x[2, 0], "k*")
                l = plt.plot(x[0, it - 1], x[1, it - 1], marker)
                if it % frame_res == 0:
                    writer.grab_frame()
                line = l.pop(0)
                line.remove()


def gif3d_2(
    initial_conditions,
    f,
    nt,
    h,
    integrator,
    filename,
    ax=None,
    color=None,
    dpi=100,
    fps=30,
    alpha=0.4,
    lim=None,
    animtype="lines",
    elev_start=0,
    elev_stop=None,
    azim_start=0,
    azim_stop=None,
    linewidth=1,
):
    """creates a gif of numerical solutions to f"""
    assert animtype in ["lines", "ring", "dots"]

    elev_stop = elev_stop if elev_stop else elev_start
    azim_stop = azim_stop if azim_stop else azim_start

    writer = PillowWriter(fps=fps)
    dir = os.path.dirname(filename)
    nsol = len(initial_conditions)
    lines = [None] * nsol
    if not os.path.exists(dir):
        os.makedirs(dir)
    x = np.empty((3, nt, nsol))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = plt.gcf()

    for i, x0 in enumerate(initial_conditions):
        x[:, :, i] = integrate(integrator, x0, f, nt=nt, h=h)

    for i in range(5):
        if lim is not None:  # periodic boundary (doesn't work for lines)
            for ix in [0, 2, 4]:
                x[ix, :, :][x[ix, :, :] < lim[ix]] += lim[ix + 1] - lim[ix]
                x[ix, :, :][x[ix, :, :] > lim[ix + 1]] += lim[ix] - lim[ix + 1]

    with writer.saving(fig, filename, dpi=dpi):
        for it in range(nt):
            ax.view_init(
                azim=azim_stop * (it / nt) + azim_start * (1 - (it / nt)),
                elev=elev_stop * (it / nt) + elev_start * (1 - (it / nt)),
            )
            if animtype == "lines":
                for k in range(nsol):
                    lines[k] = plt.plot(
                        x[0, 0:it, k],
                        x[1, 0:it, k],
                        x[2, 0:it, k],
                        "-",
                        # markersize=1,
                        color=color[k % len(color)] if type(color) == list else color,
                        alpha=alpha,
                        linewidth=linewidth,
                    )
            elif animtype == "ring":
                lines[0] = plt.plot(
                    x[0, it, :],
                    x[1, it, :],
                    x[2, it, :],
                    "-",
                    color=color,
                    alpha=alpha,
                )
            elif animtype == "dots":
                for k in range(nsol):
                    lines[k] = plt.plot(
                        x[0, it, k],
                        x[1, it, k],
                        x[2, it, k],
                        ".",
                        color=color,
                        alpha=alpha,
                    )

            writer.grab_frame()
            print(f"frame {it}/{int(nt)}")
            if it != nt - 1:
                [line.pop(0).remove() for line in lines if line]


def gif_centrifuge(
    fig,
    initial_conditions,
    f,
    nt,
    h,
    integrator,
    filename,
    colors=None,
    lim=None,
    frame_res=2,
    dpi=100,
    fps=30,
    writer=PillowWriter(fps=30),
    alpha=0.4,
):
    """creates a gif of numerical solutions to f"""
    dir = os.path.dirname(filename)
    nsol = len(initial_conditions)
    lines = [None] * nsol
    if not os.path.exists(dir):
        os.makedirs(dir)
    x = np.empty((2, nt, nsol))

    if colors is not None:
        assert len(integrator) == len(
            colors
        ), "Length of intergrator must be equal to length of markers"

    nmeth = len(integrator)

    for i, x0 in enumerate(initial_conditions):
        x[:, :, i] = integrate(integrator[i % nmeth], x0, f, nt=nt, h=h)
    with writer.saving(fig, filename, dpi=dpi):
        for it in range(nt):
            if it % frame_res == 0:
                for k in range(nsol):
                    lines[k] = plt.plot(
                        x[0, it, k],
                        x[1, it, k],
                        color=colors[k % nmeth],
                        marker=".",
                        alpha=alpha,
                    )
                writer.grab_frame()

                [line.pop(0).remove() for line in lines]
                print(f"frame {it}/{int(nt/frame_res)}")


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
    title="",
    alpha=0.2,
    color="r",
):
    """creates a gif of numerical solutions to f"""
    if writer is None:
        writer = PillowWriter(fps)
    PLANES = {
        "x-y": [0, 1],
        "y-z": [1, 2],
        "z-x": [2, 0],
    }
    PLANES[plane][0]
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
                marker=".",
                alpha=alpha,
            )  # re
            plt.scatter(
                X[it, PLANES[plane][0], imeth, :],
                X[it, PLANES[plane][1], imeth, :],
                marker=".",
                alpha=alpha,
                color=color,
            )  # ref
            # plt.scatter(X[it, PLANES[plane][0], imeth, :], X[it, PLANES[plane][1], imeth, :], marker='x')
            plt.title(title)
            writer.grab_frame()
        [writer.grab_frame() for _ in range(int(3 * fps))]


COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "grey": "#7f7f7f",
    "yellow": "#bcbd22",
    "cyan": "#17becf",
}
