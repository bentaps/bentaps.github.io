import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from harmonicOscillator import fx, fy, f, H
import sys
import os
from matplotlib.animation import PillowWriter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

font = {'size'   : 22}

plt.rc('font', **font)

STEPSIZE = 0.05
N_TIMESTEPS = 350
LIM = 1
FPS = 20
DPI = 100
INITIAL_CONDITION = np.array([[0, 0.5]])
FILENAME = sys.path[0] + "/GIFs" + "/{gifname}.gif"

METHODS = {"Standard method": ode.forward_euler, "Geometric method": ode.symplectic_euler}

x = np.linspace(-LIM, LIM, 50, endpoint=True)
mesh = np.meshgrid(x, x)
writer = PillowWriter(fps=FPS)


def initialise_figure(title=""):
    fig = plt.figure()
    plt.xlim([-LIM, LIM])
    plt.ylim([-LIM, LIM])
    plt.title(title)
    ode.quiver(fx, fy, lim=LIM)
    plt.contour(*mesh, H(*mesh))
    return fig


gif_params = dict(
    initial_conditions=INITIAL_CONDITION,
    f=f,
    nt=N_TIMESTEPS,
    h=STEPSIZE,
    writer=writer,
    dpi=DPI,
    marker="ko",
    linestyle="r-",
    frame_res=1,
)

if __name__ == "__main__":
    for name, integrator in METHODS.items():
        ode.gif(
            integrator=integrator,
            filename=FILENAME.format(gifname=name),
            fig=initialise_figure(name),
            **gif_params
        )
