import os

from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

PATH = "data/"
LENGTH = None
INTERP_FACTOR = 2
PLANES = {
    "x-y": [0, 1],
    # "y-z": [1, 2],
    # "z-x": [2, 0],
}
METHOD_INDS = [0, 1, 2, 6]

matfiles = [PATH + file for file in os.listdir(PATH) if file.endswith(".mat")]

for i, matfile in enumerate(matfiles):
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

    simname = data.get("SIMULATIONNAME", f"missing_name-{i}")[0]
    positions = np.array(data["YY"][:, 10:13, :, :])
    positions = np.remainder(positions, LENGTH) if LENGTH else positions
    nt, nd, nm, nparticles = positions.shape
    im = 0

    # Interpolate
    t_axis = np.arange(nt)
    nt_interp = nt * INTERP_FACTOR
    t_axis_interp = np.linspace(0, nt, nt_interp)
    positions_interp = np.empty((nt_interp, nd, nm, nparticles))
    for i in range(nparticles):
        interpolator = CubicSpline(t_axis, positions[:, :, im, i])
        positions_interp[:, :, im, i] = interpolator(t_axis_interp)
    
    # write GIFs
    for plane, axes in PLANES.items():
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax = plt.axes(
            xlim=(positions[:, axes[0], :, :].min(), positions[:, axes[0], :, :].max()),
            ylim=(positions[:, axes[1], :, :].min(), positions[:, axes[1], :, :].max()),
        )

        (d,) = ax.plot(positions[0, axes[0], 0, :], positions[0, axes[1], 0, :], "r.")

        plt.xlabel("xyz"[axes[0]])
        plt.xlabel("xyz"[axes[1]])
        plt.grid(visible=True, which="major")
        plt.title(f"{plane}")

        def animate_positions(i):
            return (
                d.set_data(
                    positions_interp[i, axes[0], 0, :],
                    positions_interp[i, axes[1], 0, :],
                ),
            )

        anim = animation.FuncAnimation(
            fig,
            animate_positions,
            frames=nt_interp,
            repeat_delay=1000,
        )
        writergif = animation.PillowWriter(fps=3 * INTERP_FACTOR)
        filename = f"./GIFs/particles-{simname}-{plane}.gif"
        anim.save(filename, writer=writergif)
