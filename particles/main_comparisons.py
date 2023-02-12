import os

from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import sys


sys.path.insert(1, os.path.join(sys.path[0], ".."))
import odeutils as ode

PATH = "data/"
LENGTH = None
INTERP_FACTOR = 30

FILENAME = sys.path[0] + "/GIFs" + "/{gifname}.gif"
FPS = 15
N_PARTICLES = 10000
try:
    matfiles = [PATH + file for file in os.listdir(PATH) if file.endswith(".mat")]
except FileNotFoundError:
    raise Exception("Run this code from the same directory it's in you doofus")

for idata, matfile in enumerate(matfiles):
    print(matfile)

    # Get data
    data = loadmat(matfile)
    print(data.keys())
    # print("Legend", data["Legend"])
    legend = ["Ref."] + list(l[0] for l in data["Legend"][0])
    print("Legend", [l for l in legend])
    print("T", data["T"])
    print("dt", data["dt"])
    print("lam", data["lam"])
    print("St", data["St"])

    simname = data.get("SIMULATIONNAME", f"missing_name-{idata}")[0]
    positions = np.array(data["YY"][:, 10:13, :, 0:N_PARTICLES])
    positions = np.remainder(positions, LENGTH) if LENGTH else positions
    nt, nd, nm, nparticles = positions.shape

    # Interpolate
    t_axis = np.arange(nt)
    nt_interp = nt * INTERP_FACTOR
    t_axis_interp = np.linspace(0, nt, nt_interp)
    positions_interp = np.empty((nt_interp, nd, nm, nparticles))
    for im in range(nm):
        for i in range(nparticles):
            interpolator = CubicSpline(t_axis, positions[:, :, im, i])
            positions_interp[:, :, im, i] = interpolator(t_axis_interp)

    def get_fig(title):
        fig = plt.figure()
        plt.title(title)
        return fig

    plane = "y-z" if idata == 2 else "z-x"

    if idata == 0:
        imeths = [0, 1]
    elif idata == 1:
        imeths = [0, 4]
    else:
        imeths = [2, 4]

    names = [
        [
            f"Geometric method O1 {idata} {plane}",
            "Geometric method $O(h)$",
        ],  # MRBF1+CP1
        [f"Standard method O1 {idata} {plane}", "Standard method $O(h)$"],  # TP1+FE1
        [
            f"Geometric method O2 {idata} {plane}",
            "Geometric method $O(h^2)$",
        ],  # MRBF2+CP2
        [None],  # TP2+CP2
        [f"Standard method O2 {idata} {plane}", "Standard method $O(h^2)$"],  # TP2+AB2
    ]
    for imeth in imeths:
        name = names[imeth][0]
        title = names[imeth][1]
        print(imeth, title)
        ode.gif_particles(
            X=positions_interp,
            imeth=imeth + 1,
            filename=FILENAME.format(gifname=name),
            dpi=100,
            fig=plt.figure(),
            fps=FPS,
            plane=plane,
            title=title,
            color=ode.COLORS["pink"] if "Geometric" in title else ode.COLORS["orange"],
            alpha=0.1,
        )
