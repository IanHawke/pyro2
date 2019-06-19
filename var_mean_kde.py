#!/usr/bin/env python3

# Compute the average over an ensemble of data
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
import matplotlib.gridspec as gsp


mpl.rcParams["text.usetex"] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# font sizes
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


def makeplot(basename, outfile, width, height):
    """ Get the data, average, then plot """

    files = glob.glob(basename+'_[0-9]*_[1-9][0-9]*.h5')
    print(files)

    sim_first = io.read(files[0])
    ivars = compressible.Variables(sim_first.cc_data)
    myg = sim_first.cc_data.grid
    files_size = (len(files), myg.ihi-myg.ilo, myg.jhi-myg.jlo)
    print(files_size)
    all_densities = numpy.zeros(files_size)
    for i, f in enumerate(files):
        print(f)
        s = io.read(f)
        gamma = s.cc_data.get_aux("gamma")
        dx = s.cc_data.grid.dx
        dy = s.cc_data.grid.dy
        q = compressible.cons_to_prim(s.cc_data.data, gamma, ivars,
                                      s.cc_data.grid)
        all_densities[i, :, :] = q[myg.ilo:myg.ihi, myg.jlo:myg.jhi, ivars.irho]

    mean_rho = all_densities.mean(axis=0)
    var_rho = all_densities.var(axis=0)

  # Now try doing the KDE plot
  # See https://seaborn.pydata.org/tutorial/distributions.html
  # We're doing a slice along the middle of x.
    max_density = numpy.max(all_densities)
    min_density = numpy.min(all_densities)
    d_density = max_density - min_density
    support = numpy.linspace(min_density - d_density, max_density + d_density,
                            500)
    kdensity = numpy.zeros((len(support), files_size[2]))
    for j_y in range(files_size[2]):
        print("kde", j_y)
        densities = numpy.mean(all_densities[:, :, j_y], axis=1)
        bandwidth = 1.06 * densities.std() * densities.size ** (-1 / 5.)
        kernels = []
        for x_i in densities:
           kernel = stats.norm(x_i, bandwidth).pdf(support)
           kernels.append(kernel)
        kdensity[:, j_y] = numpy.sum(kernels, axis=0)
        kdensity[:, j_y] /= trapz(kdensity[:, j_y], support)

    # plt.figure()
    # plt.imshow(numpy.transpose(kdensity),
    #           interpolation="nearest", origin="lower",
    #           extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
    #           cmap=s.cm)
    #
    fig = plt.figure(constrained_layout=False)
    widths = [1, 0.25]
    heights = [1, 1, 1, 1]
    gs = fig.add_gridspec(4, 2, width_ratios=widths, height_ratios=heights,
                          hspace=0.001, wspace=0.1)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(numpy.transpose(mean_rho),
                   interpolation="nearest", origin="lower",
                   extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                   cmap=s.cm)
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")
    j_indexes = [myg.jhi//8, myg.jhi//4, 3*myg.jhi//8, myg.jhi//2]
    for i, j_i in enumerate(j_indexes):
        ax_main.axhline(myg.y[j_i], myg.xmin, myg.xmax, 'w--', lw=3)
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(support, kdensity[:, j_i])
        ax.set_xlim(min_density, max_density)
        ax.set_xlabel(r"$\rho$")
        ax.set_yticks([])

#    fig.tight_layout()
    if outfile.endswith(".pdf"):
      plt.savefig(outfile, bbox_inches="tight")
    else:
      plt.savefig(outfile)
    plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=10.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=10.0,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("basename", type=str, nargs=1,
                        help="the basename you wish to average")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.basename[0], args.o, args.W, args.H)
