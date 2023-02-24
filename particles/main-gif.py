import os

from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

PATH = "data/"
LENGTH = None
INTERP_FACTOR = 6
PLANES = {
    "x-y": [0, 1],
    # "y-z": [1, 2],
    # "z-x": [2, 0],
}
METHOD_INDS = [0, 1, 2, 6]

matfiles = [PATH + file for file in os.listdir(PATH) if file.endswith(".mat")]

for i, matfile in enumerate(matfiles):
    if i==0 or i==1:
        continue
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
    positions = np.array(data["YY"][:, 10:13, :, 0:4000])
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
    cmap = np.random.rand(nparticles) * 256
    metadata = dict(title="Movie", artist="bentaps")
    writer = PillowWriter(fps=15, metadata=metadata)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=10., azim=-20)            
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    with writer.saving(fig, f"GIFs/{simname}.gif", 150):
        for it in range(nt_interp):
            plt.cla()
            # ax = plt.axes(
            #     xlim=(positions[:, 0, :, :].min(), positions[:, 0, :, :].max()),
            #     ylim=(positions[:, 1, :, :].min(), positions[:, 1, :, :].max()),
            #     zlim=(positions[:, 2, :, :].min(), positions[:, 2, :, :].max()),
            # )

            ax.scatter(
                positions_interp[it, 0, 0, :],
                positions_interp[it, 1, 0, :],
                positions_interp[it, 2, 0, :],
                c=cmap,
                s=5,
                alpha=0.5
            )

            # plt.grid(visible=True, which="major")
            plt.title(f"")

            ax.set_xlim(
                positions_interp[:, 0, :, :].min(), 
                positions_interp[:, 0, :, :].max()
            )
            ax.set_ylim(
                positions_interp[:, 1, :, :].min(), 
                positions_interp[:, 1, :, :].max()
            )
            ax.set_zlim(
                positions_interp[:, 2, :, :].min(), 
                positions_interp[:, 2, :, :].max()
            )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

            plt.axis("off")
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])


            writer.grab_frame()

