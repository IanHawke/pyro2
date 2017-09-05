from __future__ import print_function

import importlib
import numpy as np
import matplotlib.pyplot as plt

import advection
import advection_weno.fluxes as flx
import mesh.patch as patch
import mesh.integration as integration
import mesh.array_indexer as ai
from simulation_null import NullSimulation, grid_setup, bc_setup

from util import profile


class Simulation(advection.Simulation):

    def initialize(self):
        """
        Initialize the grid and variables for advection and set the initial
        conditions for the chosen problem.

        Over-ridden from the base class as the number of ghosts depends on
        the WENO order
        """

        weno_order = self.rp.get_param("advection.weno_order")
        assert(weno_order in (2, 3, 4, 5)), "Currently only implemented weno_order=2, 3, 4, 5"

        my_grid = grid_setup(self.rp, ng=weno_order+1)

        # create the variables
        my_data = patch.CellCenterData2d(my_grid)
        bc = bc_setup(self.rp)[0]
        my_data.register_var("density", bc)
        my_data.create()

        self.cc_data = my_data

        # now set the initial conditions for the problem
        problem = importlib.import_module("advection.problems.{}".format(self.problem_name))
        problem.init_data(self.cc_data, self.rp)

    def substep(self, myd):
        """
        take a single substep in the RK timestepping starting with the
        conservative state defined as part of myd
        """

        myg = myd.grid

        k = myg.scratch_array()

        flux_x, flux_y =  flx.fluxes(myd, self.rp, self.dt)

        F_x = ai.ArrayIndexer(d=flux_x, grid=myg)
        F_y = ai.ArrayIndexer(d=flux_y, grid=myg)

        k.v()[:,:] = \
            (F_x.v() - F_x.ip(1))/myg.dx + \
            (F_y.v() - F_y.jp(1))/myg.dy

        return k


    def method_compute_timestep(self):
        """
        Compute the advective timestep (CFL) constraint.  We use the
        driver.cfl parameter to control what fraction of the CFL
        step we actually take.
        """

        cfl = self.rp.get_param("driver.cfl")

        u = self.rp.get_param("advection.u")
        v = self.rp.get_param("advection.v")

        # the timestep is 1/sum{|U|/dx}
        xtmp = max(abs(u),self.SMALL)/self.cc_data.grid.dx
        ytmp = max(abs(v),self.SMALL)/self.cc_data.grid.dy

        self.dt = cfl/(xtmp + ytmp)


    def evolve(self):
        """
        Evolve the linear advection equation through one timestep.  We only
        consider the "density" variable in the CellCenterData2d object that
        is part of the Simulation.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myg = self.cc_data.grid
        myd = self.cc_data

        method = self.rp.get_param("advection.temporal_method")

        rk = integration.RKIntegrator(myd.t, self.dt, method=method)
        rk.set_start(myd)

        for s in range(rk.nstages()):
            ytmp = rk.get_stage_start(s)
            ytmp.fill_BC_all()
            k = self.substep(ytmp)
            rk.store_increment(s, k)

        rk.compute_final_update()

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
