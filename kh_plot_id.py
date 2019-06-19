#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:35:38 2019

@author: ih3
"""

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


def makeplot(outfile, width, height):
    """ Get the data, average, then plot """

    files = ['kh_random_outputs/256/kh_random_256_1234_0000.h5',
             'kh_random_outputs/512/kh_random_512_1234_0000.h5',
             'kh_random_outputs/1024/kh_random_1024_1234_0000.h5',
             'kh_random_outputs/2048/kh_random_2048_1234_0000.h5']
    sizes = [256, 512, 1024, 2048]

    densities = []
    max_rho = -numpy.inf
    min_rho = numpy.inf
    for f in files:
        s = io.read(f)
        ivars = compressible.Variables(s.cc_data)
        myg = s.cc_data.grid
        gamma = s.cc_data.get_aux("gamma")
        q = compressible.cons_to_prim(s.cc_data.data, gamma, ivars,
                                      s.cc_data.grid)
        densities.append(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi, ivars.irho])
        max_rho = max(max_rho, numpy.max(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi,
                                           ivars.irho]))
        min_rho = min(min_rho, numpy.min(q[myg.ilo:myg.ihi, myg.jlo:myg.jhi,
                                           ivars.irho]))

    fig = plt.figure(constrained_layout=True)
    widths = [1, 0.25]
    heights = [1, 1, 1, 1]
    gs = fig.add_gridspec(4, 2, width_ratios=widths, height_ratios=heights,
                          hspace=0.001, wspace=0.1)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(numpy.transpose(densities[-1]),
                   interpolation="nearest", origin="lower",
                   extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                   cmap=s.cm)
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")
    zoom_axes = []
    zoom_width = 0.2
    for i in range(4):
        zoom_axes.append(fig.add_subplot(gs[i, 1]))
        zoom_axes[i].imshow(numpy.transpose(densities[i]),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=s.cm)
        zoom_axes[i].set_xlim(0.5-zoom_width/2, 0.5+zoom_width/2)
        zoom_axes[i].set_ylim(0.75-zoom_width/8, 0.75+zoom_width/8)
        # zoom_axes[i].set_axis_off()
        zoom_axes[i].set_xticklabels([])
        zoom_axes[i].set_yticklabels([])
        zoom_axes[i].set_ylabel(fr"{sizes[i]}${{}}^2$")
        zoom_axes[i].yaxis.set_label_position("right")

#    ax_main.indicate_inset_zoom(zoom_axes[0])

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
    parser.add_argument("-H", type=float, default=8.0,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.o, args.W, args.H)
