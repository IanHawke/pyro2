#!/usr/bin/env python3

# Plot same realization at different resolutions.
# A lot of this is taken from plot.py and io.py

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import argparse
import glob
import util.io as io
import compressible
import util.plot_tools as plot_tools
from scipy import stats
from scipy.integrate import trapz
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid

mpl.rcParams["text.usetex"] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# font sizes
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


def makeplot(outfile, width, height):
    """ Get the data, average, then plot """

    realizations = list(range(1,129))
    # realizations = list(range(1,513))

    fig = plt.figure(figsize=(width, height))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2**3, 2**4),
                    # nrows_ncols=(2**4, 2**5),
                    axes_pad=0.01,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    densities = []
    max_rho = -numpy.inf
    min_rho = numpy.inf
    for r in realizations:
        f = glob.glob(f'kh_random_outputs/256/kh_random_256_{r}_[1-9]*.h5')
        assert len(f)==1
        s = io.read(f[0])
        ivars = compressible.Variables(s.cc_data)
        myg = s.cc_data.grid
        gamma = s.cc_data.get_aux("gamma")
        q = compressible.cons_to_prim(s.cc_data.data, gamma, ivars,
                                      s.cc_data.grid)
        densities.append(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi, ivars.irho])
        max_rho = max(max_rho, numpy.max(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi, ivars.irho]))
        min_rho = min(min_rho, numpy.min(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi, ivars.irho]))

    for ax, rho in zip(grid, densities):
        ax.set_axis_off()
        im = ax.imshow(numpy.transpose(rho),
                       interpolation="nearest", origin="lower",
                       extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                       vmin = min_rho, vmax=max_rho,
                       cmap=s.cm)

        # needed for PDF rendering
#        cb = axes.cbar_axes[n].colorbar(img)
#        cb.solids.set_rasterized(True)
#        cb.solids.set_edgecolor("face")

        # if size in sizes[:2]:
        #     ax.set_title(fr"{size}${{}}^2$")
        # else:
        #     ax.set_title(fr"{size}${{}}^2$", y=-0.1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    cbar = ax.cax.colorbar(im)
#    fig.subplots_adjust(wspace=0.01, hspace=0.01)
#    fig.tight_layout()
    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile)
#    plt.show()

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=20.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=10.0,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.o, args.W, args.H)