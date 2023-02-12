import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from harmonicOscillator import fx, fy, f, H
import sys
import os
from matplotlib.animation import PillowWriter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

font = {"size": 22}

plt.rc("font", **font)

STEPSIZE = 0.075
N_TIMESTEPS = 500
LIM = 1
FPS = 20
DPI = 100
NSOLS = 1000
INITIAL_CONDITION = ode.random_circle(radius=0.7, nsols=NSOLS)
FILENAME = sys.path[0] + "/GIFs" + "/{gifname}.gif"
print(INITIAL_CONDITION.shape)
METHODS = {
    "Heavy particles": ode.forward_euler,
    "Light particles": ode.backward_euler_sho,
}

x = np.linspace(-LIM, LIM, 50, endpoint=True)
mesh = np.meshgrid(x, x)
writer = PillowWriter(fps=FPS)


def initialise_figure(title=""):
    fig = plt.figure()
    plt.xlim([-LIM, LIM])
    plt.ylim([-LIM, LIM])
    plt.title(title)
    ode.quiver(fx, fy, lim=LIM)
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis="y", which="both", left=False, labelleft=False)

    return fig


gif_params = dict(
    initial_conditions=INITIAL_CONDITION,
    f=f,
    nt=N_TIMESTEPS,
    h=STEPSIZE,
    writer=writer,
    dpi=DPI,
    frame_res=1,
)

if __name__ == "__main__":
    ode.gif_centrifuge(
        integrator=[ode.forward_euler, ode.backward_euler_sho],
        colors=[ode.COLORS['purple'], ode.COLORS['orange']],
        filename=FILENAME.format(gifname="Centrifuge effect"),
        fig=initialise_figure("Centrifuge effect"),
        **gif_params
    )
