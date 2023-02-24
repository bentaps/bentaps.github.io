# %%
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from exampleODE import f, fx, fy, p_ls, q_ls, r_ls
from matplotlib.animation import PillowWriter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

FPS = 30
NSOLS = 10
DPI = 100
TEMP = "_temp"  # for testing change to None when creating a GIF for the website
N_TIMESTEPS = 10
STEPSIZE = 0.1
LIM = 1
FILENAME = sys.path[0] + "/GIFs" + TEMP + "/{gifname}.gif"

print(FILENAME)

INITIAL_CONDITIONS = ode.random_grid()

# Create GIF of random trajectories 
fig = plt.figure()
plt.xlim([-LIM, LIM])
plt.ylim([-LIM, LIM])

writer = PillowWriter(fps=FPS)
gifname = "PhaseLines"
ode.gif(
    fig=fig,
    initial_conditions=INITIAL_CONDITIONS,
    f=f,
    nt=N_TIMESTEPS,
    h=STEPSIZE,
    integrator=ode.forward_euler,
    filename=FILENAME.format(gifname=gifname),
    lim=LIM,
    writer=writer,
    dpi=DPI,
)

# Create GIF of random trajectories plus level sets and vector plot 
gifname = "PhaseLines2"
X = np.linspace(-LIM,LIM,100)
with writer.saving(fig, FILENAME.format(gifname=gifname), dpi=DPI):
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, p_ls(X), "k--", linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, q_ls(X), "k--", linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, r_ls(X), "k--", linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    ode.quiver(fx, fy, lim=LIM)
    [writer.grab_frame() for _ in range(130)]
