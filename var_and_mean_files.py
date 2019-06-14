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
    files_size = (len(files), *sim_first.cc_data.get_vars().shape[:2])
    print(files_size)
    all_densities = numpy.zeros(files_size)
    all_vorticities = numpy.zeros_like(all_densities)
    for i, f in enumerate(files):
        print(f)
        s = io.read(f)
        gamma = s.cc_data.get_aux("gamma")
        dx = s.cc_data.grid.dx
        dy = s.cc_data.grid.dy
        q = compressible.cons_to_prim(s.cc_data.data, gamma, ivars,
                                      s.cc_data.grid)
        all_densities[i, :, :] = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        all_vorticities[i, 1:-1, 1:-1] = ((v[2:, 1:-1] - v[:-2, 1:-1]) / dx -
                                          (u[1:-1, 2:] - u[1:-1, :-2]) / dy)

    mean_rho = all_densities.mean(axis=0)
    var_rho = all_densities.var(axis=0)
    mean_vorticity = all_vorticities.mean(axis=0)
    var_vorticity = all_vorticities.var(axis=0)

    myg = s.cc_data.grid

    fields = [mean_rho, var_rho, mean_vorticity, var_vorticity]
    field_names = [r"mean($\rho$)", r"var($\rho$)",
                   r"mean($\omega$)", r"var($\omega$)"]

    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')
    plt.clf()
    plt.rc("font", size=16)
    fig, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

    for n, ax in enumerate(axes):
        v = fields[n]

        img = ax.imshow(numpy.transpose(v),
                        interpolation="nearest", origin="lower",
                        extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                        cmap=s.cm)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # needed for PDF rendering
        cb = axes.cbar_axes[n].colorbar(img)
        cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")

        ax.set_title(field_names[n])

#    fig.tight_layout()
    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile)
#    plt.show()

#   Now try doing the KDE plot
#   See https://seaborn.pydata.org/tutorial/distributions.html
#   We're doing a slice along the middle of x.
#    max_density = numpy.max(all_densities)
#    min_density = numpy.min(all_densities)
#    d_density = max_density - min_density
#    support = numpy.linspace(min_density - d_density, max_density + d_density,
#                             500)
#    kdensity = numpy.zeros((len(support), files_size[2]))
#    i_c = files_size[1]//2
#    for j_y in range(files_size[2]):
#        densities = all_densities[:, i_c, j_y]
#        bandwidth = 1.06 * densities.std() * densities.size ** (-1 / 5.)
#        kernels = []
#        for x_i in densities:
#            kernel = stats.norm(x_i, bandwidth).pdf(support)
#            kernels.append(kernel)
#        kdensity[:, j_y] = numpy.sum(kernels, axis=0)
#        kdensity[:, j_y] /= trapz(kdensity[:, j_y], support)
#
#    plt.figure()
#    plt.imshow(numpy.transpose(kdensity),
#               interpolation="nearest", origin="lower",
#               extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
#               cmap=s.cm)
#    plt.show()
#    Surface plot doesn't look good?
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    X, S = numpy.meshgrid(myg.x, support)
#    ax.plot_surface(X, S, kdensity,
#                    cmap=s.cm)


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
