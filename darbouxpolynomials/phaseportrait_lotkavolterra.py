# %%
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from LotkaVolterra import f, fx, fy, p_ls, q_ls
from matplotlib.animation import PillowWriter

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

FPS = 40
NSOLS = 100
DPI = 100
TEMP = "_temp"  # for testing change to None when creating a GIF for the website
N_TIMESTEPS = 300
STEPSIZE = 0.025
LIM = 5
FILENAME = sys.path[0] + "/GIFs" + TEMP + "/{gifname}.gif"

print(FILENAME)

INITIAL_CONDITIONS = ode.random_grid(lim=LIM, nsols=NSOLS, gridpoints=13)

# Create GIF of random trajectories 
fig = plt.figure()
plt.xlim([-LIM, LIM])
plt.ylim([-LIM, LIM])
plt.grid()
plt.title("Predator-prey system")

writer = PillowWriter(fps=FPS)
gifname = "PhaseLinesLV"
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
gifname = "PhaseLines2LV"
X = np.linspace(-LIM,LIM,100)
with writer.saving(fig, FILENAME.format(gifname=gifname), dpi=DPI):
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, p_ls(X), "k--", linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    plt.plot(X, q_ls(X), "k--", linewidth=3)
    [writer.grab_frame() for _ in range(30)]
    ode.quiver(fx, fy, lim=LIM)
    [writer.grab_frame() for _ in range(130)]
