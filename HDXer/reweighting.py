#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import pickle

from .errors import HDX_Error
from .reweighting_functions import read_contacts_hbonds, read_kints_segments

class MaxEnt():
    """Class for Maximum Entropy reweighting of a predicted
       HDX ensemble to fit target data. Initialises with a 
       dictionary of default parameters foe analysis, accessible
       as MaxEnt.params

       Default parameters can either be updated directly in the MaxEnt.params
       dictionary or by supplying extra parameters as kwargs during
       initialisation, e.g. MaxEnt(gamma=10**-2) or MaxEnt(**param_dict)

       Perform a Maximum Entropy reweighting of an ensemble of
       predicted HDX data using the MaxEnt.run method"""

    def __init__(self, **extra_params):
        """Initialize a Maximum Entropy reweighting object with the following parameters.
           Syntax of this documentation = Parameter (default value) : Description
               do_reweight (True) : Flag to turn on reweighting
               do_params (True) : Flag to turn on model parameter optimisation (e.g. Bc and Bh)
               do_mcmin (False) : Flag to turn on a Monte Carlo minimisation of model parameters, rather than a gradient-based optimisation
               do_mcsampl (False) : Flag to turn on a Monte Carlo sampling of model parameters, rather than a minimisation
               mc_refvar (0.03) : Reference variance of the MC sampling distribution (~estimated error on deuterated fractions)
               mc_equilsteps (-1) : Equilibration steps and subsequent smoothing (as 1/sqrt(number of steps)) for the initial stage of MC sampling. Default value (-1) switches off equilibration
               radou_bc (0.35) : Initial value of beta_C for the Radou-style HDX forward model
               radou_bh (2.00) : Initial value of beta_H for the Radou-style HDX forward model
               radou_bcrange (1.5) : Initial range (affects model parameter step size) for the beta_C parameter for MC minimization
               radou_bhrange (16.0) : Initial range (affects model parameter step size) for the beta_H parameter for MC minimization
               tolerance (10**-10) : Convergence criterion for reweighting, taken as: ( sum(abs(lambdas_new) - sum(abs(lambdas_old)) ) / sum(abs(lambdas_new))
               maxiters (10**6) : Maximum number of iterations for reweighting
               param_maxiters (10**2) : Maximum number of iterations for model parameter minimisation
               stepfactor (10**-5) : Initial rate (affects lambdas step size) for reweighting
               stepfactor_scaling (1.005) : Ratio for scaling down of initial step size if oscillations are observed
               param_stepfactor (10**-1) Initial rate (affects model parameter step size) for MC parameter optimisation
               temp (300.) : Temperature for HDX predictions in Kelvin
               random_initial (False) : Randomize the initial frame weights. Otherwise, all frames will initially be weighted equally"""

        maxentparams = { 'do_reweight' : True,
                         'do_params' : True,
                         'do_mcmin' : False,
                         'do_mcsampl' : False,
                         'mc_refvar' : 0.03,
                         'mc_equilsteps' : -1,
                         'radou_bc' : 0.35,
                         'radou_bh' : 2.00,
                         'radou_bcrange' : 1.5,
                         'radou_bhrange' : 16.0,
                         'tolerance' : 10**-10,
                         'maxiters' : 10**6,
                         'param_maxiters' : 10**2,
                         'stepfactor' : 10**-5,
                         'stepfactor_scaling' : 1.005,
                         'param_stepfactor' : 10**-1,
                         'temp' : 300.,
                         'random_initial' : False
                         }
        maxentparams.update(extra_params)
        maxentparams.update(kT = maxentparams['temp'] * 0.008314598)
        self.methodparams = maxentparams



    # Setup functions
    def setup_no_runobj(self, folderlist, kint_file, expt_file_path, times):
        """Setup initial variables when a run object file is NOT read in"""
        self.runvalues = {}

        maxentvalues = { 'contacts' : None,
                         'hbonds' : None,
                         'kint' : None,
                         'exp_dfrac' : None,
                         'segfilters' : None,
                         'lambdas' : None,
                         'nframes' : None,
                         'iniweights' : None,
                         'minuskt_filtered' : None,
                         'exp_dfrac_filtered' : None,
                         'ndatapoints' : None,
                         'nsegs' : None,
                         'lambdamod' : None,
                         'deltalambdamod' : None,
                         'curriter' : None }
        _contacts, _hbonds, _sorted_resids = read_contacts_hbonds(folderlist, 
                                                                  self.runparams['contacts_prefix'], 
                                                                  self.runparams['hbonds_prefix'])
        _nresidues = len(_hbonds)
        _minuskt, _exp_dfrac, _segfilters = read_kints_segments(kint_file, expt_file_path, _nresidues, times, _sorted_resids)

        # Starting lambda values
        _lambdas = np.zeros(_nresidues)

    #    if mc_sample_params:
        if self.methodparams['do_mcsampl']:
            self.mcsamplvalues = {}
            mcsamplvalues = { 'newlambdas' : np.zeros(_nresidues),
                              'newlambdas_h' : np.zeros(_nresidues),
                              'newlambdas_c' : np.zeros(_nresidues),
                              'newlambdas_ave_h' : np.zeros(_nresidues),
                              'newlambdas_ave_c' : np.zeros(_nresidues),
                              'lambdas_c' : np.zeros(_nresidues),
                              'lambdas_h' : np.zeros(_nresidues),
                              'chisquare_ave' : 0.}
            self.mcsamplvalues.update(mcsamplvalues)

        # Write initial parameter values
        _nframes = _contacts.shape[1]
        with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("Temp, kT, convergence tolerance, BetaC, BetaH, gamma, update rate (step size) factor, nframes\n")
            f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n"
                    % (self.methodparams['temp'],
                       self.methodparams['kT'],
                       self.methodparams['tolerance'],
                       self.methodparams['radou_bc'],
                       self.methodparams['radou_bh'],
                       self.runparams['gamma'],
                       self.methodparams['stepfactor'],
                       _nframes))

        # Random initial weights perturb the ensemble - e.g. can we recreate the final ensemble obtained with equally weighted frames?
        # Seed for random state will be printed out in logfile for reproducibility
        if self.methodparams['random_initial']:
            np.random.seed(None)
            statenum = np.random.randint(2**32)
            state = np.random.RandomState(statenum)
            with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'a') as f:
                f.write('Initial weights were randomized, seed for np.random.RandomState = %d\n' % statenum)
            _iniweights = state.rand(_nframes)
            np.savetxt("initial_weights_RandomState%d.dat" % statenum, _iniweights)
            np.random.seed(None)
        else:
            _iniweights = np.ones(_nframes)

        # Write some headers in output files
        with open("%swork.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# gamma, chisquare, work(kJ/mol)\n")
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc \n")
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc, work \n")

        # Set some constants & initial values for the iteration loop
        _minuskt_filtered = -_minuskt * _segfilters
        _exp_dfrac_filtered = _exp_dfrac * _segfilters
        _ndatapoints = np.sum(_segfilters)
        _nsegs = _segfilters.shape[0]
        _lambdamod = 0.0
        _deltalambdamod = 0.0
        _curriter = 0

        maxentvalues.update(contacts = _contacts,
                            hbonds = _hbonds,
                            minuskt = _minuskt,
                            exp_dfrac = _exp_dfrac,
                            segfilters = _segfilters,
                            lambdas = _lambdas,
                            nframes = _nframes,
                            iniweights = _iniweights,
                            minuskt_filtered = _minuskt_filtered,
                            exp_dfrac_filtered = _exp_dfrac_filtered,
                            ndatapoints = _ndatapoints,
                            nsegs = _nsegs,
                            lambdamod = _lambdamod,
                            deltalambdamod = _deltalambdamod,
                            curriter = _curriter)

        self.runvalues.update(maxentvalues)

    def setup_runobj(self, runobj):
        """Setup initial variables when a run object file IS read in"""

    def setup_restart(self, rstfile):
        """Setup initial variables when a restart file IS read in"""
        # List of pickle variables:
        global contacts, hbonds, kint, exp_dfrac, segfilters, lambdas, \
               lambdasc, lambdash, _lambdanewaverh, _lambdanewaverc, \
               _lambdanew, _lambdanewh, _lambdanewc, chisquareav, \
               nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, \
               Bc, Bh, gamma, ratef, avesigmalnpi, currcount
        contacts, hbonds, kint, exp_dfrac, segfilters, lambdas,lambdasc ,lambdash , _lambdanewaverh, _lambdanewaverc, \
        _lambdanew, _lambdanewh, _lambdanewc, chisquareav, \
        nframes, iniweights, num, exp_df_segf, normseg, numseg, lambdamod, deltalambdamod, \
        Bc, Bh, gamma, ratef, avesigmalnpi, currcount = pickle.load(open(rstfile, 'rb'))
        # Append to existing files
        with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("Temp, kT, convergence tolerance, BetaC, BetaH, gamma, update rate (step size) factor, nframes\n")
            f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n" % (T, kT, tol, Bc, Bh, gamma, ratef, nframes))
        with open("%swork.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# gamma, chisquare, work(kJ/mol)\n")
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc \n")
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# Iteration, avehdxdev, chisquare, lambdamod, deltalambdamod, rate, Bh, Bc, work \n")
        print("Restart file %s read" % rstfile)

    def set_run_params(self, gamma, runobj, restart, paramdict):
        """Set basic run parameters if not being read from calc_hdx object or a restart file"""
        
        self.runparams = {}
        self.runparams['gamma'] = gamma
        
        if restart is not None:
            self.runparams['from_restart'] = True
            return
        if runobj is not None:
            self.runparams['from_calchdx'] = True
            return

        self.runparams.update(paramdict) # update with any provided options
        
        # If options aren't provided, set defaults
        try:
            self.runparams['hbonds_prefix']
        except KeyError:
            self.runparams['hbonds_prefix'] = 'Hbonds_'
        try:
            self.runparams['contacts_prefix']
        except KeyError:
            self.runparams['contacts_prefix'] = 'Contacts_'
        try:
            self.runparams['out_prefix']
        except KeyError:
            self.runparams['out_prefix'] = 'reweighting_'

    def update_lnpi_and_weights(self):
        _contacts = self.methodparams['radou_bc'] * self.runvalues['contacts']
        _hbonds = self.methodparams['radou_bh'] * self.runvalues['hbonds']
        self.runvalues['lnpi'] = _hbonds + _contacts

        if self.methodparams['do_mcsampl']:
            biasfactor = np.sum(self.mcsamplvalues['lambdas_c'][:, np.newaxis] * self.runvalues['contacts']
                                + self.mcsamplvalues['lambdas_h'][:, np.newaxis] * self.runvalues['hbonds'],
                                axis=0)  # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
        else:
            biasfactor = np.sum(self.runvalues['lambdas'][:, np.newaxis] * self.runvalues['lnpi'],
                                axis=0)  # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D

        self.runvalues['currweights'] = self.runvalues['iniweights'] * np.exp(biasfactor)
        self.runvalues['currweights'] = self.runvalues['currweights'] / np.sum(self.runvalues['currweights'])
        self.runvalues['ave_lnpi'] = np.sum(self.runvalues['currweights'] * self.runvalues['lnpi'], axis=1)


    def run(self, gamma=10**-2, runobj=None, restart=None, **run_params):
        self.set_run_params(gamma, runobj, restart, run_params)

        # Choose which setup to do. Restart > Runobj > Normal
        if restart is None:
            if runobj is None:
                try:
                    self.setup_no_runobj(self.runparams['data_folders'],
                                         self.runparams['kint_file'],
                                         self.runparams['exp_file'],
                                         self.runparams['times'])
                except KeyError:
                    raise HDX_Error("Missing parameters to set up a reweighting run.\n"
                                    "Please ensure a restart or calc_hdx object is provided,"
                                    "or provide the following arguments to the run() call:"
                                    "data_folders, kint_file, exp_file, times")
            else:
                self.setup_runobj(runobj)
        else:
            self.setup_restart(restart)

        # 1) Do setup (restart or no restart)
        # 2) Do run 
        # 3) Do final save/cleanup


