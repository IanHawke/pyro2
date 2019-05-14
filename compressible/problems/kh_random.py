from __future__ import print_function

import numpy as np

import mesh.patch as patch
from util import msg


def init_data(my_data, rp):
    """ initialize the Kelvin-Helmholtz problem """

    msg.bold("initializing the Kelvin-Helmholtz problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print(my_data.__class__)
        msg.fail("ERROR: patch invalid in kh.py")

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    np.random.seed(rp.get_param("kh.seed"))

    interface_as = np.random.rand(2, 10)
    interface_as = (interface_as.T/np.sum(interface_as, axis=1)).T # Normalize so sum = 1
    interface_bs = -np.pi + 2*np.pi*np.random.rand(2, 10)

    print("Generated random numbers:")
    print("a:", interface_as)
    print("b:", interface_bs)

    epsilon = rp.get_param("kh.epsilon")

    def interface1(x):
        interface_y = 0.25 * np.ones_like(x)
        for n in range(10):
            interface_y += epsilon * interface_as[0, n] * np.cos(interface_bs[0, n] + 2*n*np.pi*x)
        return interface_y

    def interface2(x):
        interface_y = 0.75 * np.ones_like(x)
        for n in range(10):
            interface_y += epsilon * interface_as[1, n] * np.cos(interface_bs[1, n] + 2*n*np.pi*x)
        return interface_y

    rho_1 = rp.get_param("kh.rho_1")
    v_1 = rp.get_param("kh.v_1")
    rho_2 = rp.get_param("kh.rho_2")
    v_2 = rp.get_param("kh.v_2")

    gamma = rp.get_param("eos.gamma")

    myg = my_data.grid


    idx1 = myg.y2d < interface1(myg.x2d)
    idx2 = np.logical_and(myg.y2d >= interface1(myg.x2d),
                          myg.y2d <  interface2(myg.x2d))
    idx3 = myg.y2d >= interface2(myg.x2d)

    # we will initialize momentum as velocity for now

    # lower quarter
    dens[idx1] = rho_1
    xmom[idx1] = v_1

    # second quarter
    dens[idx2] = rho_2
    xmom[idx2] = v_2

    # third quarter
    dens[idx3] = rho_1
    xmom[idx3] = v_1

    # upper half
    xmom[:, :] *= dens
    ymom[:, :] = 0

    p = 2.5
    ener[:, :] = p/(gamma - 1.0) + 0.5*(xmom[:, :]**2 + ymom[:, :]**2)/dens[:, :]


def finalize():
    """ print out any information to the user at the end of the run """
    pass
