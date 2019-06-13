#!/usr/bin/env python3

# Compute the average over an ensemble of data
# A lot of this is taken from plot.py and io.py

import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import h5py
import glob
import util.io as io


mpl.rcParams["text.usetex"] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# font sizes
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

def makeplot(basename, outfile, width, height):
    """ plot the data in a plotfile using the solver's vis() method """

    files = glob.glob(basename+'_[0-9]*_[1-9][0-9]*.h5')
    print(files)

    sim_average = io.read(files[0])
    sim_average.cc_data.get_vars()[:, :, :] = 0
    for f in files:
        print(f)
        s = io.read(f)
        sim_average.cc_data.get_vars()[:, :, :] += s.cc_data.get_vars()[:, :, :] / len(files)


    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')

    sim_average.dovis()
    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile)
    plt.show()



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=8.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=4.5,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("basename", type=str, nargs=1,
                        help="the basename you wish to average")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()

    makeplot(args.basename[0], args.o, args.W, args.H)
