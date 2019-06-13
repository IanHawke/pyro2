#!/usr/bin/env python3

# Compute the average over an ensemble of data
# A lot of this is taken from plot.py and io.py

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import argparse
import h5py
import glob
import util.io as io
import compressible
import util.plot_tools as plot_tools


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
    for i, f in enumerate(files):
        print(f)
        s = io.read(f)
        gamma = s.cc_data.get_aux("gamma")

        q = compressible.cons_to_prim(s.cc_data.data, gamma, ivars, s.cc_data.grid)

        all_densities[i, :, :] = q[:, :, ivars.irho]
        
    mean_rho = all_densities.mean(axis=0)
    var_rho = all_densities.var(axis=0)
    
    myg = s.cc_data.grid

    fields = [mean_rho, var_rho]
    field_names = [r"mean($\rho$)", r"var($\rho$)"]
    
    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')
    plt.clf()
    plt.rc("font", size=18)
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
    plt.show()



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=9.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=16.0,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("basename", type=str, nargs=1,
                        help="the basename you wish to average")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.basename[0], args.o, args.W, args.H)
