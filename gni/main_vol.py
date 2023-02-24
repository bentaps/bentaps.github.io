import numpy as np
import matplotlib.pyplot as plt
from ABCflow import fx, fy, fy, f
import sys
import os


sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

NSOLS = 1500
N_TIMESTEPS = 600
STEPSIZE = 0.2
FPS = 16
ALPHA = 0.12
DPI = 250
LENGTH = 40
ZEXTRA = 1.2
ZASPECT = 1.5
LINEWIDTH = 0.5
width=2
gif_kwargs = dict(
    filename="./GIFs/ABC_flow_temp.gif",
    elev_start=0,
    azim_start=-90,
    linewidth=LINEWIDTH,
    # elev_stop=0,
    # azim_stop=0,
    color=[
        ode.COLORS["blue"],
        ode.COLORS["blue"],
        ode.COLORS["blue"],
        'k',
        ode.COLORS["purple"],
    ],
    alpha=ALPHA,
    nt=N_TIMESTEPS,
    h=STEPSIZE,
    fps=FPS,
    dpi=DPI,
    animtype="lines",
)

np.random.seed(seed=1)
initial_conditions = ode.rand_cube(
    NSOLS, width=width, x0=(0, 1.6+LENGTH, 1.6), stretch=(LENGTH*2/width, 1, 1)
)


def get_fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # margins
    ax.set_xlim(-LENGTH, LENGTH)
    ax.set_ylim(-LENGTH, LENGTH)
    ax.set_zlim(-ZASPECT * ZEXTRA * LENGTH, ZASPECT * LENGTH)
    ax.set_box_aspect((2, 2, 2 * ZASPECT))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    plt.axis("off")
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    return fig, ax


fig, ax = get_fig_ax()

ode.gif3d_2(
    ax=ax,
    f=f,
    initial_conditions=initial_conditions,
    integrator=ode.volume_preserving,
    **gif_kwargs,
)

plt.axis("on")

plt.show()

# initial_conditions = ode.ring(NSOLS, r=1, x0=(2, 1, 0))
# initial_conditions = ode.rand_cube(NSOLS, width=0.25, x0=(3, 2, 1))