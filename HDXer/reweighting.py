#!/usr/bin/env python

import numpy as np
import pickle

from .errors import HDX_Error
from .reweighting_functions import read_contacts_hbonds, \
                                   subsample_contacts_hbonds, \
                                   read_kints_segments, \
                                   generate_trial_betas, \
                                   calc_trial_ave_lnpi, \
                                   calc_trial_dfracs, \
                                   calc_work

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
               bv_bc (0.35) : Initial value of beta_C for the Best/Vendruscolo-style HDX forward model
               bv_bh (2.00) : Initial value of beta_H for the Best/Vendruscolo-style HDX forward model
               bv_bcrange (1.5) : Initial range (affects model parameter step size) for the beta_C parameter for MC minimization
               bv_bhrange (16.0) : Initial range (affects model parameter step size) for the beta_H parameter for MC minimization
               tolerance (10**-10) : Convergence criterion for reweighting, taken as: ( sum(abs(lambdas_new) - sum(abs(lambdas_old)) ) / sum(abs(lambdas_new))
               maxiters (10**6) : Maximum number of iterations for reweighting
               param_maxiters (10**2) : Maximum number of iterations for model parameter minimisation
               stepfactor (10**-5) : Initial rate (affects lambdas step size) for reweighting
               stepfactor_scaling (1.005) : Ratio for scaling down of initial step size if oscillations are observed
               param_stepfactor (10**-1) : Initial rate (affects model parameter step size) for MC parameter optimisation
               temp (300.) : Temperature for HDX predictions in Kelvin
               random_initial (False) : Randomize the initial frame weights. Otherwise, all frames will initially be weighted equally"""

        maxentparams = { 'do_reweight' : True,
                         'do_params' : True,
                         'do_mcmin' : False,
                         'do_mcsampl' : False,
                         'mc_refvar' : 0.03,
                         'mc_equilsteps' : -1,
                         'bv_bc' : 0.35,
                         'bv_bh' : 2.00,
                         'bv_bcrange' : 1.5,
                         'bv_bhrange' : 16.0,
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
        """Setup initial variables when calc_hdx object files are NOT provided"""
        self.runvalues = {}

        maxentvalues = { 'contacts' : None,
                         'hbonds' : None,
                         'minuskt' : None,
                         'exp_dfrac' : None,
                         'segfilters' : None,
                         'lambdas' : None,
                         'nframes' : None,
                         'minuskt_filtered' : None,
                         'exp_dfrac_filtered' : None,
                         'n_datapoints' : None,
                         'n_segs' : None,
                         'lambda_mod' : None,
                         'delta_lambda_mod' : None,
                         'curr_lambda_stepsize' : None,
                         'curriter' : None,
                         'is_converged' : None }
        _contacts, _hbonds, _sorted_resids = read_contacts_hbonds(folderlist,
                                                                  self.runparams['contacts_prefix'],
                                                                  self.runparams['hbonds_prefix'])
        if self.runparams['do_subsample']:
            _contacts, _hbonds = subsample_contacts_hbonds(_contacts, _hbonds,
                                                           self.runparams['sub_start'],
                                                           self.runparams['sub_end'],
                                                           self.runparams['sub_interval'])

        _nresidues = len(_hbonds)
        _minuskt, _exp_dfrac, _segfilters = read_kints_segments(kint_file, expt_file_path, _nresidues, times, _sorted_resids)

        # Starting lambda values
        _lambdas = np.zeros(_nresidues)

    #    if mc_sample_params:
        if self.methodparams['do_mcsampl']:
            self.mcsamplvalues = {}
            mcsamplvalues = {'final_MClambdas': np.zeros(_nresidues),
                             'final_MClambdas_h': np.zeros(_nresidues),
                             'final_MClambdas_c': np.zeros(_nresidues),
                             'gamma_MClambdas_h': np.zeros(_nresidues),
                             'gamma_MClambdas_c': np.zeros(_nresidues),
                             'gamma_MClambdas': np.zeros(_nresidues),
                             'ave_MClambdas_h': np.zeros(_nresidues),
                             'ave_MClambdas_c': np.zeros(_nresidues),
                             'lambdas_c': np.zeros(_nresidues),
                             'lambdas_h': np.zeros(_nresidues),
                             'MC_MSE_ave': 0.,
                             'MC_resfracs_ave': 0.,
                             'smoothing_rate': 1.0}
            self.mcsamplvalues.update(mcsamplvalues)

        # Write initial parameter values
        _nframes = _contacts.shape[1]
        with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("Temp, kT, Convergence criterion, Beta_C, Beta_H, Gamma, Update rate (step size) factor, N frames\n")
            f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.4e, %8.6f, %d\n"
                    % (self.methodparams['temp'],
                       self.methodparams['kT'],
                       self.methodparams['tolerance'],
                       self.methodparams['bv_bc'],
                       self.methodparams['bv_bh'],
                       self.runparams['gamma'],
                       self.methodparams['stepfactor'],
                       _nframes))

        # Random initial weights perturb the ensemble - e.g. can we recreate the final ensemble obtained with equally weighted frames?
        # Seed for random state will be printed out in logfile for reproducibility
        if self.runparams['iniweights'] is not None:
            try:
                _iniweights = np.array(self.runparams['iniweights'])
            except:
                raise HDX_Error("You supplied initial weights in a form that can't be converted to a numpy array, please check your inputs")
            try:
                assert(np.isclose(sum(self.runparams['iniweights']), _nframes))
            except AssertionError:
                raise HDX_Error("You supplied initial weights but the weights don't add up to the number of trajectory frames")
            try:
                assert(len(self.runparams['iniweights']) == _nframes)
            except AssertionError:
                raise HDX_Error("You supplied initial weights but the number of weights doesn't equal the number of trajectory frames")
        elif self.methodparams['random_initial']:
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
            f.write("# gamma, weighted MSE to target, weighted RMSE to target, work (kJ/mol by default)\n")
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H \n")
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H, Curr. work \n")

        # Set some constants & initial values for the iteration loop
        _minuskt_filtered = _minuskt * _segfilters
        _exp_dfrac_filtered = _exp_dfrac * _segfilters
        _n_datapoints = np.sum(_segfilters)
        _n_segs = _segfilters.shape[0]
        _lambda_mod = 0.0
        _delta_lambda_mod = 0.0
        _curr_lambda_stepsize = 0.0
        _curriter = 0
        _converged = False

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
                            n_datapoints = _n_datapoints,
                            n_segs = _n_segs,
                            lambda_mod = _lambda_mod,
                            delta_lambda_mod = _delta_lambda_mod,
                            curr_lambda_stepsize = _curr_lambda_stepsize,
                            curriter = _curriter,
                            is_converged = _converged)

        self.runvalues.update(maxentvalues)

    def setup_calc_hdx(self, resultsobj, analysisobj):
        """Setup initial variables when calc_hdx object files ARE provided
           So far this reads the by-residue Contacts, H-bonds, and residue IDs from the results object,
           and times, expt. dfracs, and segments from the analysis object"""
        self.runvalues = {}

        maxentvalues = { 'contacts' : None,
                         'hbonds' : None,
                         'minuskt' : None,
                         'exp_dfrac' : None,
                         'segfilters' : None,
                         'lambdas' : None,
                         'nframes' : None,
                         'minuskt_filtered' : None,
                         'exp_dfrac_filtered' : None,
                         'n_datapoints' : None,
                         'n_segs' : None,
                         'lambda_mod' : None,
                         'delta_lambda_mod' : None,
                         'curr_lambda_stepsize' : None,
                         'curriter' : None,
                         'is_converged' : None }
        _contacts, _hbonds, _sorted_resids = resultsobj.contacts, resultsobj.hbonds, np.array([ resultsobj.top.residue(residx).resSeq for residx in resultsobj.reslist ], dtype=np.int16)
        if self.runparams['do_subsample']:
            _contacts, _hbonds = subsample_contacts_hbonds(_contacts, _hbonds,
                                                           self.runparams['sub_start'],
                                                           self.runparams['sub_end'],
                                                           self.runparams['sub_interval'])

        _nresidues = len(_hbonds)

        ### This duplicates the function of reweighting_functions.read_kints_segments locally but using the calc_hdx objects
        _kint = resultsobj.rates
        _kint = np.repeat(_kint[:, np.newaxis], len(self.runparams['times']), axis=1) * self.runparams['times']

        _exp_dfrac, _segments = analysisobj.expfracs, analysisobj.segres['segres']

        # convert expt to (segments, residues, times)
        _exp_dfrac = _exp_dfrac[:, np.newaxis, :].repeat(_nresidues, axis=1)
        # convert kint to (segments, residues, times)
        _kint = _kint[np.newaxis, :, :].repeat(len(_segments), axis=0)

        # Make a set of filters that defines the residues in each segment & timepoint
        _segfilters = []
        for seg in _segments:
            seg_resids = range(seg[0], seg[1] + 1)
            _segfilters.append(np.in1d(_sorted_resids, seg_resids[1:]))  # Filter but skipping first residue in segment
        _segfilters = np.array(_segfilters, dtype=np.int16)
        _segfilters = np.repeat(_segfilters[:, :, np.newaxis], len(self.runparams['times']),
                               axis=2)  # Repeat to shape (n_segments, n_residues, n_times)

        assert all((_segfilters.shape == _exp_dfrac.shape,
                    _segfilters.shape == _kint.shape,
                    _nresidues == len(_sorted_resids)))  # Check we've at least read in the right number!

        print("Segments and experimental dfracs read from calc_hdx objects")
        _minuskt = -_kint
        ###

        # Starting lambda values
        _lambdas = np.zeros(_nresidues)

        #    if mc_sample_params:
        if self.methodparams['do_mcsampl']:
            self.mcsamplvalues = {}
            mcsamplvalues = {'final_MClambdas': np.zeros(_nresidues),
                             'final_MClambdas_h': np.zeros(_nresidues),
                             'final_MClambdas_c': np.zeros(_nresidues),
                             'gamma_MClambdas_h': np.zeros(_nresidues),
                             'gamma_MClambdas_c': np.zeros(_nresidues),
                             'gamma_MClambdas': np.zeros(_nresidues),
                             'ave_MClambdas_h': np.zeros(_nresidues),
                             'ave_MClambdas_c': np.zeros(_nresidues),
                             'lambdas_c': np.zeros(_nresidues),
                             'lambdas_h': np.zeros(_nresidues),
                             'MC_MSE_ave': 0.,
                             'MC_resfracs_ave': 0.,
                             'smoothing_rate': 1.0}
            self.mcsamplvalues.update(mcsamplvalues)

        # Write initial parameter values
        _nframes = _contacts.shape[1]
        with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("Temp, kT, Convergence criterion, Beta_C, Beta_H, Gamma, Update rate (step size) factor, N frames\n")
            f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n"
                    % (self.methodparams['temp'],
                       self.methodparams['kT'],
                       self.methodparams['tolerance'],
                       self.methodparams['bv_bc'],
                       self.methodparams['bv_bh'],
                       self.runparams['gamma'],
                       self.methodparams['stepfactor'],
                       _nframes))

        # Random initial weights perturb the ensemble - e.g. can we recreate the final ensemble obtained with equally weighted frames?
        # Seed for random state will be printed out in logfile for reproducibility
        if self.runparams['iniweights'] is not None:
            try:
                _iniweights = np.array(self.runparams['iniweights'])
            except:
                raise HDX_Error("You supplied initial weights in a form that can't be converted to a numpy array, please check your inputs")
            try:
                assert(np.isclose(sum(self.runparams['iniweights']), _nframes))
            except AssertionError:
                raise HDX_Error("You supplied initial weights but the weights don't add up to the number of trajectory frames")
            try:
                assert(len(self.runparams['iniweights']) == _nframes)
            except AssertionError:
                raise HDX_Error("You supplied initial weights but the number of weights doesn't equal the number of trajectory frames")
        elif self.methodparams['random_initial']:
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
            f.write("# gamma, weighted MSE to target, weighted RMSE to target, work (kJ/mol by default)\n")
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H \n")
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'w') as f:
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H, Curr. work \n")

        # Set some constants & initial values for the iteration loop
        _minuskt_filtered = _minuskt * _segfilters
        _exp_dfrac_filtered = _exp_dfrac * _segfilters
        _n_datapoints = np.sum(_segfilters)
        _n_segs = _segfilters.shape[0]
        _lambda_mod = 0.0
        _delta_lambda_mod = 0.0
        _curr_lambda_stepsize = 0.0
        _curriter = 0
        _converged = False

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
                            n_datapoints = _n_datapoints,
                            n_segs = _n_segs,
                            lambda_mod = _lambda_mod,
                            delta_lambda_mod = _delta_lambda_mod,
                            curr_lambda_stepsize = _curr_lambda_stepsize,
                            curriter = _curriter,
                            is_converged = _converged)

        self.runvalues.update(maxentvalues)

    def setup_restart(self, rstfile):
        """Setup initial variables when a restart file IS read in"""
        # Update self from pickle file
        with open(rstfile, 'rb') as fpkl:
            tmp_dict = pickle.load(fpkl)
        self.__dict__.update(tmp_dict)

        # Append to existing files
        with open("%sinitial_params.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("Temp, kT, Convergence criterion, Beta_C, Beta_H, Gamma, Update rate (step size) factor, N frames\n")
            f.write("%s, %6.3f, %5.2e, %5.2f, %5.2f, %5.2e, %8.6f, %d\n"
                    % (self.methodparams['temp'],
                       self.methodparams['kT'],
                       self.methodparams['tolerance'],
                       self.methodparams['bv_bc'],
                       self.methodparams['bv_bh'],
                       self.runparams['gamma'],
                       self.methodparams['stepfactor'],
                       self.runvalues['contacts'].shape[1]))
        with open("%swork.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# gamma, weighted MSE to target, weighted RMSE to target, work (kJ/mol by default)\n")
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H \n")
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("# RESTARTED FROM FILE %s :\n" % rstfile)
            f.write("# Iteration, weighted MSE to target, weighted RMSE to target, Total lambdas, Fractional change in lambda, Curr. stepsize, Beta_C, Beta_H, Curr. work \n")
        print("Restart file %s read" % rstfile)

    def set_run_params(self, gamma, resultsobj, analysisobj, restart, paramdict):
        """Set basic run parameters if not being read from calc_hdx object or a restart file"""

        # By default we read in all frames
        self.runparams = { 'do_subsample': False,
                           'sub_start': 0,
                           'sub_end': -1,
                           'sub_interval': 1,
                           'iniweights': None }
        self.runparams['gamma'] = gamma

        if restart is not None:
            self.runparams['from_restart'] = True
            return
        else:
            self.runparams['from_restart'] = False
        if resultsobj is not None:
            if analysisobj is not None:
                self.runparams['from_calchdx'] = True
                self.runparams['times'] = analysisobj.params['times']
                return
            else:
                raise HDX_Error("You've supplied a BV results object from calc_hdx but not an Analysis object.\n"
                                "Both are required if you wish to continue a calc_hdx run.")
        if analysisobj is not None:
            raise HDX_Error("You've supplied an Analysis object from calc_hdx but not a BV results object.\n"
                            "Both are required if you wish to continue a calc_hdx run.")

        self.runparams['from_calchdx'] = False

        self.runparams.update(paramdict) # update with any provided options

        # If options aren't provided, set defaults
        try:
            self.runparams['hbonds_prefix']
        except KeyError:
            self.runparams['hbonds_prefix'] = 'Hbonds_chain_0_res_'
        try:
            self.runparams['contacts_prefix']
        except KeyError:
            self.runparams['contacts_prefix'] = 'Contacts_chain_0_res_'
        try:
            self.runparams['out_prefix']
        except KeyError:
            self.runparams['out_prefix'] = 'reweighting_'

    def update_lnpi_and_weights(self):
        """Update the current values of ensemble-averaged protection factor for each residue,
           and current weight for each trajectory frame,
           based on the current lambda (biasing) value for each residue.

           Updates 'lnpi', 'avelnpi' and 'currweights' entries in the MaxEnt.runvalues dictionary"""

        _contacts = self.methodparams['bv_bc'] * self.runvalues['contacts']
        _hbonds = self.methodparams['bv_bh'] * self.runvalues['hbonds']
        self.runvalues['lnpi'] = _hbonds + _contacts

        # Calculate by-frame bias from lambda values (per residue) and contacts/hbonds (per residue and frame)
        if self.methodparams['do_mcsampl']:
            biasfactor = np.sum(self.mcsamplvalues['lambdas_c'][:, np.newaxis] * self.runvalues['contacts']
                                + self.mcsamplvalues['lambdas_h'][:, np.newaxis] * self.runvalues['hbonds'],
                                axis=0)  # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D
        else:
            biasfactor = np.sum(self.runvalues['lambdas'][:, np.newaxis] * self.runvalues['lnpi'],
                                axis=0)  # Sum over all residues, = array of len(nframes). lambdas is 1D array broadcast to 2D

        # Calculate current weight of each frame (from initial weight & bias applied), and weighted-average protection factors (by-residue)
        self.runvalues['currweights'] = self.runvalues['iniweights'] * np.exp(biasfactor)
        self.runvalues['currweights'] = self.runvalues['currweights'] / np.sum(self.runvalues['currweights'])
        self.runvalues['ave_lnpi'] = np.sum(self.runvalues['currweights'] * self.runvalues['lnpi'], axis=1)

        # Calculate helper values & convert weighted-average protection factors to 3 dimensions.
        # On first iteration, set std. dev. of ln(Pf)
        if self.runvalues['curriter'] == 0:
            _sigmalnpi = np.sum(self.runvalues['currweights'] * (self.runvalues['lnpi']**2), axis=1)
            _sigmalnpi = _sigmalnpi - (self.runvalues['ave_lnpi']**2)
            # Round off possible minor fp arithmetic problems if the std. dev is 0
            _sigmalnpi = np.where(np.isclose(_sigmalnpi, 0, atol=10**-12), 0, _sigmalnpi)
            self.runvalues['sigma_lnpi'] = np.sqrt(_sigmalnpi)
            self.runvalues['ave_sigma_lnpi'] = np.mean(self.runvalues['sigma_lnpi'])

        # Convert weighted-average protection factors to 3D arrays of shape [n_segments, n_residues, n_times]
        # This is the same shape as self.runvalues['segfilters'] used to filter arrays by residues belonging to each segment
        self.runvalues['ave_lnpi'] = np.repeat(self.runvalues['ave_lnpi'][:,np.newaxis], len(self.runparams['times']), axis=1)
        self.runvalues['ave_lnpi'] = self.runvalues['ave_lnpi'][np.newaxis,:,:].repeat(self.runvalues['n_segs'], axis=0)

    def update_dfracs_and_mse(self):
        """Convert weighted-average protection factors to deuterated fractions
           and calculate mean square error (MSE) to experiment.

           The MSE is weighted by the number of residues in each segment.

           Updates 'curr_residue_dfracs', 'curr_segment_dfracs' and 'curr_MSE' entries in the MaxEnt.runvalues dictionary"""

        # Do arithmetic always multiplying values by filter, so 'False' entries are not counted
        # Set temporary array for denominator so we can use np.divide to avoid divide-by-zero warning
        denom = self.runvalues['ave_lnpi'] * self.runvalues['segfilters']
        # D(t) = 1 - exp(-kt/P_i)
        self.runvalues['curr_residue_dfracs'] = 1.0 - \
                                                np.exp(np.divide(self.runvalues['minuskt_filtered'], np.exp(denom),
                                                                 out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
                                                                 where=denom!=0))

        _curr_segment_dfracs = np.nanmean(self.runvalues['curr_residue_dfracs'], axis=1)
        # Convert to 3D array of shape [n_segments, n_residues, n_times]
        self.runvalues['curr_segment_dfracs'] = _curr_segment_dfracs[:,np.newaxis,:].repeat(self.runvalues['segfilters'].shape[1], axis=1)
        # Although curr_segment_dfracs is a 3D array, repeated for every residue in a segment, dividing by n_datapoints
        # should give a mean square error that's weighted by the number of residues in each segment
        self.runvalues['curr_MSE'] = np.sum((self.runvalues['curr_segment_dfracs'] * self.runvalues['segfilters']
                                             - self.runvalues['exp_dfrac_filtered'])**2) / self.runvalues['n_datapoints']

    def update_work(self):
        """Calculate current apparent work value using current values of:
           MaxEnt.runvalues['lnpi']
           MaxEnt.runvalues['lambdas']
           MaxEnt.runvalues['currweights']
           MaxEnt.methodparams['kT']

           Updates MaxEnt.runvalues['work'] with the current apparent work value. Units will be determined by the units of kT."""

        self.runvalues['work'] = calc_work(self.runvalues['lnpi'], self.runvalues['lambdas'], self.runvalues['currweights'], self.methodparams['kT'])

    def optimize_parameters_MC(self):
        """Minimize beta parameters using an MC protocol. Beta parameter moves are accepted if they reduce mean
           square error to the target data.

           Each step of minimization updates the values of 'bv_bc' and 'bv_bh' in the
           MaxEnt.methodparams dictionary, and 'ave_lnpi', 'curr_residue_dfracs', 'curr_segment_dfracs', and
           'curr_MSE' in the MaxEnt.runvalues dictionary. Steps can be controlled with the 'param_maxiters'
           entry in the MaxEnt.methodparams dictionary"""

        # Get the weighted-average contacts & H-bonds for each residue
        _ave_contacts = np.sum(self.runvalues['currweights'] * self.runvalues['contacts'], axis=1)
        _ave_hbonds = np.sum(self.runvalues['currweights'] * self.runvalues['hbonds'], axis=1)

        # Then do MC-based minimization
        for curr_mc_iter in range(self.methodparams['param_maxiters']):
            curr_bv_bc = self.methodparams['bv_bc']
            curr_bv_bh = self.methodparams['bv_bh']
            trial_bv_bc, trial_bv_bh = generate_trial_betas(curr_bv_bc, curr_bv_bh,
                                                                  self.methodparams['bv_bcrange'],
                                                                  self.methodparams['bv_bhrange'],
                                                                  self.methodparams['param_stepfactor'])
            trial_ave_lnpi = calc_trial_ave_lnpi(_ave_contacts, _ave_hbonds, trial_bv_bc, trial_bv_bh,
                                                 len(self.runparams['times']), self.runvalues['n_segs'])
            trial_residue_dfracs, trial_segment_dfracs, trial_MSE = calc_trial_dfracs(trial_ave_lnpi,
                                                                                      self.runvalues['segfilters'],
                                                                                      self.runvalues['minuskt_filtered'],
                                                                                      self.runvalues['exp_dfrac_filtered'],
                                                                                      self.runvalues['n_datapoints'])

            if trial_MSE < self.runvalues['curr_MSE']:
                self.methodparams['bv_bc'] = trial_bv_bc
                self.methodparams['bv_bh'] = trial_bv_bh
                self.runvalues['curr_MSE'] = trial_MSE
                self.runvalues['ave_lnpi'] = trial_ave_lnpi
                self.runvalues['curr_segment_dfracs'] = trial_segment_dfracs
                self.runvalues['curr_residue_dfracs'] = trial_residue_dfracs

    def optimize_parameters_gradient(self):
        """Minimize beta parameters using a gradient descent protocol.

           Each step of minimization updates the values of 'bv_bc' and 'bv_bh' in the
           MaxEnt.methodparams dictionary, and 'ave_lnpi', 'curr_residue_dfracs', 'curr_segment_dfracs', and
           'curr_MSE' in the MaxEnt.runvalues dictionary. Steps and convergence can be controlled with the
           'param_maxiters' and 'tolerance' entries in the MaxEnt.methodparams dictionary"""

        # Get the weighted-average contacts & H-bonds for each residue
        _ave_contacts = np.sum(self.runvalues['currweights'] * self.runvalues['contacts'], axis=1)
        _ave_hbonds = np.sum(self.runvalues['currweights'] * self.runvalues['hbonds'], axis=1)

        # Then do gradient descent minimization
        for curr_grad_iter in range(self.methodparams['param_maxiters']):
            # define first derivative of chisquare wrt to Bh
            # define derivative of the sim_dfrac wrt to parameters
            # we need average number of contacts and hbonds calcd above

            denom = self.runvalues['ave_lnpi'] * self.runvalues['segfilters']
            rate_filtered = np.divide(-self.runvalues['minuskt_filtered'], np.exp(denom),
                                      out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
                                      where=denom != 0)
            der_dfrac_bh = np.nanmean(-(np.exp(-rate_filtered))
                                      * rate_filtered * _ave_hbonds[np.newaxis, :, np.newaxis],
                                      axis=1)  # need a sum over residues if data on fragments
            der_dfrac_bc = np.nanmean(-(np.exp(-rate_filtered))
                                      * rate_filtered * _ave_contacts[np.newaxis, :, np.newaxis],
                                      axis=1)  # need a sum over residues if data on fragments
            secder_dfrac_bh = np.nanmean(_ave_hbonds[np.newaxis, :, np.newaxis]**2
                                         * (np.exp(-rate_filtered))
                                         * (rate_filtered - rate_filtered**2), axis=1)
            secder_dfrac_bc = np.nanmean(_ave_contacts[np.newaxis, :, np.newaxis]**2
                                         * (np.exp(-rate_filtered))
                                         * (rate_filtered - rate_filtered**2), axis=1)
            delta_byseg_dfrac = np.nanmean((np.where(self.runvalues['segfilters'], self.runvalues['curr_segment_dfracs'], np.nan)
                                               - np.where(self.runvalues['segfilters'], self.runvalues['exp_dfrac_filtered'], np.nan)), axis=1)
            # Get derivs of betas
            der_bh = 2.0 * np.sum(delta_byseg_dfrac * der_dfrac_bh)
            der_bc = 2.0 * np.sum(delta_byseg_dfrac * der_dfrac_bc)
            secder_bh = 2.0 * np.sum(der_dfrac_bh**2 + secder_dfrac_bh * delta_byseg_dfrac)
            secder_bc = 2.0 * np.sum(der_dfrac_bc**2 + secder_dfrac_bc * delta_byseg_dfrac)

            # now update the parameters
            delta_bh = np.abs(der_bh / np.abs(secder_bh))
            delta_bc = np.abs(der_bc / np.abs(secder_bc))
            if delta_bh / np.abs(self.methodparams['bv_bh']) > self.methodparams['param_stepfactor']: # Scale steps if they're too large
                self.methodparams['bv_bh'] = np.abs(self.methodparams['bv_bh']
                                                       - self.methodparams['param_stepfactor'] * der_bh / np.abs(secder_bh))
            else:
                self.methodparams['bv_bh'] = np.abs(self.methodparams['bv_bh'] - der_bh / np.abs(secder_bh))
            if delta_bc / np.abs(self.methodparams['bv_bc']) > self.methodparams['param_stepfactor']: # Scale steps if they're too large
                self.methodparams['bv_bc'] = np.abs(self.methodparams['bv_bc']
                                                       - self.methodparams['param_stepfactor'] * der_bc / np.abs(secder_bc))
            else:
                self.methodparams['bv_bc'] = np.abs(self.methodparams['bv_bc'] - der_bc / np.abs(secder_bc))

            # Update values for the next iteration
            self.runvalues['ave_lnpi'] = calc_trial_ave_lnpi(_ave_contacts, _ave_hbonds,
                                                             self.methodparams['bv_bc'], self.methodparams['bv_bh'],
                                                             len(self.runparams['times']), self.runvalues['n_segs'])
            self.runvalues['curr_residue_dfracs'], self.runvalues['curr_segment_dfracs'], self.runvalues['curr_MSE'] = calc_trial_dfracs(self.runvalues['ave_lnpi'],
                                                                                                                                         self.runvalues['segfilters'],
                                                                                                                                         self.runvalues['minuskt_filtered'],
                                                                                                                                         self.runvalues['exp_dfrac_filtered'],
                                                                                                                                         self.runvalues['n_datapoints'])

            # Convergence criterion uses overall 'tolerance' values, same for lambdas
            if delta_bh / np.abs(self.methodparams['bv_bh']) < self.methodparams['tolerance']:
                if delta_bc / np.abs(self.methodparams['bv_bc']) < self.methodparams['tolerance']:
                    break

    def sample_parameters_MC(self):
        """Sample range of beta parameters using an MC protocol.

           If sampling is switched on using by setting MaxEnt.methodparams['do_mcsampl'] = True,
           then parameter sampling is controlled by the following dictionary entries:
           MaxEnt.methodparams['mc_equilsteps'] : Equilibration steps and subsequent smoothing (as 1/sqrt(number of steps)) for the initial stage of MC sampling. Default value (-1) switches off equilibration
           MaxEnt.methodparams['param_maxiters'] :  Maximum number of iterations for model parameter minimisation
           MaxEnt.methodparams['mc_refvar'] : Reference variance of the MC sampling distribution (~estimated error on deuterated fractions)

           The current values for the sampling are stored in the MaxEnt.mcsamplvalues dictionary

           The final result of MC parameter sampling updates bv_bc and bv_bh in the MaxEnt.methodparams dictionary"""

        # Define some helpful internal functions for making MC moves and recalculating protection factors & deuterated fractions
        def update_sampled_totals(total_bc, total_bh, total_mse, total_resfracs, total_lambdas_bc, total_lambdas_bh):
            # Add current values of bc, bh, mse, residue fractions, lambdas to totals using a closure.
            def adding_function(bc, bh, mse, resfracs, lambdas):
                # Creates function to add current values to totals. Closure must access nonlocal variables, can't just reassign with +=
                new_bc = total_bc + bc
                new_bh = total_bh + bh
                new_mse = total_mse + mse
                new_resfracs = total_resfracs + resfracs
                new_lambdas_bc = total_lambdas_bc + (bc * lambdas)
                new_lambdas_bh = total_lambdas_bh + (bh * lambdas)
                return new_bc, new_bh, new_mse, new_resfracs, new_lambdas_bc, new_lambdas_bh
            return adding_function

        def update_sampled_totals_nolambda(total_bc, total_bh, total_mse, total_resfracs):
            # Add current values of bc, bh, mse, residue fractions to totals using a closure.
            def adding_function(bc, bh, mse, resfracs):
                # Creates function to add current values to totals. Closure must access nonlocal variables, can't just reassign with +=
                new_bc = total_bc + bc
                new_bh = total_bh + bh
                new_mse = total_mse + mse
                new_resfracs = total_resfracs + resfracs
                return new_bc, new_bh, new_mse, new_resfracs
            return adding_function

        def update_sampled_averages(total_bc, total_bh, total_mse, total_resfracs, total_lambdas_bc, total_lambdas_bh):
            # Turns totals of bc, bh mse, fractions, lambdas into averages
            ave_bc = total_bc / self.methodparams['param_maxiters']
            ave_bh = total_bh / self.methodparams['param_maxiters']
            ave_mse = total_mse / self.methodparams['param_maxiters']
            ave_resfracs = total_resfracs / self.methodparams['param_maxiters']
            ave_lambdas_bc = total_lambdas_bc / self.methodparams['param_maxiters']
            ave_lambdas_bh = total_lambdas_bh / self.methodparams['param_maxiters']
            return ave_bc, ave_bh, ave_mse, ave_resfracs, ave_lambdas_bc, ave_lambdas_bh

        def update_sampled_averages_nolambda(total_bc, total_bh, total_mse, total_resfracs):
            # Turns totals of bc, bh mse, fractions, lambdas into averages
            ave_bc = total_bc / self.methodparams['param_maxiters']
            ave_bh = total_bh / self.methodparams['param_maxiters']
            ave_mse = total_mse / self.methodparams['param_maxiters']
            ave_resfracs = total_resfracs / self.methodparams['param_maxiters']
            return ave_bc, ave_bh, ave_mse, ave_resfracs

        ### End of useful functions
        ### Start of sampling code
        # First, determine if we want to do some equilibration steps, and set the smoothing applied to the step size during equilibration
        if self.methodparams['mc_equilsteps'] > 0:
            self.mcsamplvalues['smoothing_rate'] = self.methodparams['mc_equilsteps'] \
                                 / np.sqrt(self.methodparams['mc_equilsteps']
                                 * (self.methodparams['mc_equilsteps'] + self.runvalues['curriter']))
        else:
            self.mcsamplvalues['smoothing_rate'] = 1.0

        # Get the weighted-average contacts & H-bonds for each residue
        _ave_contacts = np.sum(self.runvalues['currweights'] * self.runvalues['contacts'], axis=1)
        _ave_hbonds = np.sum(self.runvalues['currweights'] * self.runvalues['hbonds'], axis=1)

        # Save original values before sampling if not saved elsewhere
        self.mcsamplvalues['MC_MSE_ave'] = self.runvalues['curr_MSE']
        self.mcsamplvalues['MC_resfracs_ave'] = self.runvalues['curr_residue_dfracs']
        curr_bv_bh = self.methodparams['bv_bh']
        curr_bv_bc = self.methodparams['bv_bc']
        #embed()
        ### Start of main sampling loop
        for curr_mc_iter in range(self.methodparams['param_maxiters']):
            # 1) Make move in betas and recalculate protection factors & deuterated fractions
            trial_bv_bc, trial_bv_bh = generate_trial_betas(curr_bv_bc, curr_bv_bh,
                                                                  self.methodparams['bv_bcrange'],
                                                                  self.methodparams['bv_bhrange'],
                                                                  self.methodparams['param_stepfactor'])
            trial_ave_lnpi = calc_trial_ave_lnpi(_ave_contacts, _ave_hbonds, trial_bv_bc, trial_bv_bh,
                                                 len(self.runparams['times']), self.runvalues['n_segs'])
            trial_residue_dfracs, trial_segment_dfracs, trial_MSE = calc_trial_dfracs(trial_ave_lnpi,
                                                                                      self.runvalues['segfilters'],
                                                                                      self.runvalues['minuskt_filtered'],
                                                                                      self.runvalues['exp_dfrac_filtered'],
                                                                                      self.runvalues['n_datapoints'])

            # Immediately accept move if it improves MSE to experiment
            if trial_MSE < self.runvalues['curr_MSE']:
                curr_bv_bh = trial_bv_bh
                curr_bv_bc = trial_bv_bc
                self.runvalues['curr_MSE'] = trial_MSE
                self.runvalues['ave_lnpi'] = trial_ave_lnpi
                self.runvalues['curr_segment_dfracs'] = trial_segment_dfracs
                self.runvalues['curr_residue_dfracs'] = trial_residue_dfracs
            else:
                # Acceptance test based on Gaussian distribution of MSDs with variance MaxEnt.methodparams['mc_refvar']
                trial_acc_val = self.runvalues['n_segs'] * len(self.runparams['times']) \
                                * trial_MSE / (2.0 * self.methodparams['mc_refvar'])
                orig_acc_val = self.runvalues['n_segs'] * len(self.runparams['times']) \
                               * self.runvalues['curr_MSE'] / (2.0 * self.methodparams['mc_refvar'])
                move_prob = np.exp(-(trial_acc_val - orig_acc_val))
                if move_prob > np.random.random_sample():
                    curr_bv_bh = trial_bv_bh
                    curr_bv_bc = trial_bv_bc
                    self.runvalues['curr_MSE'] = trial_MSE
                    self.runvalues['ave_lnpi'] = trial_ave_lnpi
                    self.runvalues['curr_segment_dfracs'] = trial_segment_dfracs
                    self.runvalues['curr_residue_dfracs'] = trial_residue_dfracs

            if self.methodparams['do_reweight']:
                curr_lambdas = self.calc_lambdas()
                if curr_mc_iter == 0:
                    bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum, lambdas_bc_sum, lambdas_bh_sum = 0, 0, 0, 0, 0, 0
                add_to_totals = update_sampled_totals(bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum, lambdas_bc_sum, lambdas_bh_sum)
                bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum, lambdas_bc_sum, lambdas_bh_sum = add_to_totals(curr_bv_bc, curr_bv_bh, self.runvalues['curr_MSE'], self.runvalues['curr_residue_dfracs'], curr_lambdas)
            else:
                if curr_mc_iter == 0:
                    bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum = 0, 0, 0, 0
                add_to_totals = update_sampled_totals_nolambda(bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum)
                bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum = add_to_totals(curr_bv_bc, curr_bv_bh, self.runvalues['curr_MSE'], self.runvalues['curr_residue_dfracs'])
            ### End of main sampling loop

        # Calculate averages of the sampled values
        if self.methodparams['do_reweight']:
            bv_bc_ave, bv_bh_ave, MSE_ave, residue_dfracs_ave, self.mcsamplvalues['ave_MClambdas_c'], self.mcsamplvalues['ave_MClambdas_h'] = update_sampled_averages(bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum, lambdas_bc_sum, lambdas_bh_sum)
        else:
            bv_bc_ave, bv_bh_ave, MSE_ave, residue_dfracs_ave = update_sampled_averages_nolambda(bv_bc_sum, bv_bh_sum, MSE_sum, residue_dfracs_sum)


        # Finally, update the main dictionaries with averages of Bh, Bc, MSE, lambda etc, applying a scaling factor for the initial equilibration period if desired
        self.methodparams['bv_bh'] = self.methodparams['bv_bh'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                                        self.mcsamplvalues['smoothing_rate'] * bv_bh_ave
        self.methodparams['bv_bc'] = self.methodparams['bv_bc'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                                        self.mcsamplvalues['smoothing_rate'] * bv_bc_ave
        self.mcsamplvalues['MC_MSE_ave'] = self.mcsamplvalues['MC_MSE_ave'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                                           self.mcsamplvalues['smoothing_rate'] * MSE_ave
        self.mcsamplvalues['MC_resfracs_ave'] = self.mcsamplvalues['MC_resfracs_ave'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                                           self.mcsamplvalues['smoothing_rate'] * residue_dfracs_ave  # Not currently used but could be - average of residue fractions

        # Update lambdas using final Bh & Bc values if desired
        if self.methodparams['do_reweight']:
            _lambdanewh = self.mcsamplvalues['final_MClambdas_h'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                          self.mcsamplvalues['smoothing_rate'] * (self.mcsamplvalues['ave_MClambdas_h'] / self.methodparams['param_maxiters'])
            _lambdanewc = self.mcsamplvalues['final_MClambdas_c'] * (1.0 - self.mcsamplvalues['smoothing_rate']) + \
                          self.mcsamplvalues['smoothing_rate'] * (self.mcsamplvalues['ave_MClambdas_c'] / self.methodparams['param_maxiters'])
            _lambdanew = 0.5 * ((_lambdanewh / self.methodparams['bv_bh']) + (_lambdanewc / self.methodparams['bv_bc']))
            self.mcsamplvalues['final_MClambdas_h'] = _lambdanewh
            self.mcsamplvalues['final_MClambdas_c'] = _lambdanewc
            self.mcsamplvalues['final_MClambdas'] = _lambdanew
            self.update_lambdas(self.mcsamplvalues['final_MClambdas'], self.mcsamplvalues['final_MClambdas_c'], self.mcsamplvalues['final_MClambdas_h'])

    def calc_lambdas(self):
        """Calculate target lambda values using current values of the following attributes:
           self.runvalues['ave_lnpi'] : average ln(protection factor) for each residue/segment/timepoint
           self.runvalues['curr_segment_dfracs'] : current deuterated fractions for each residue/segment/timepoint
           self.runvalues['segfilters'] : Boolean filters to determine which residues belong to which segment
           self.runvalues['minuskt_filtered'] : -kt (the numerator in calculating deuterated fractions), filtered using the Boolean filter above
           self.runvalues['exp_dfrac_filtered'] : target deuterate fractions, filtered using the Boolean filter above

           Returns: current_lambdas (np.array)"""
        denom = self.runvalues['ave_lnpi'] * self.runvalues['segfilters']
        curr_lambdas = np.nansum(
            np.sum((self.runvalues['curr_segment_dfracs'] * self.runvalues['segfilters'] - self.runvalues['exp_dfrac_filtered']) * \
                   np.exp(np.divide(self.runvalues['minuskt_filtered'], np.exp(denom),
                                    out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
                                    where=denom != 0)) * \
                   np.divide(-self.runvalues['minuskt_filtered'], np.exp(denom),
                             out=np.full(self.runvalues['minuskt_filtered'].shape, np.nan),
                             where=denom != 0), axis=2) / \
            (np.sum(self.runvalues['segfilters'], axis=1)[:, 0])[:, np.newaxis], axis=0)
        return curr_lambdas

    def update_lambdas(self, target_lambdas, target_lambdas_c=None, target_lambdas_h=None):
        """Update current lambda values by making a step size towards target lambda values

           Input:
           target_lambdas (np.array) : target lambda values calculated using self.calc_lambdas()

           Output:
           Updates self.runvalues['lambdas'] and self.runvalues['curr_lambda_stepsize'].
           Optionally updates self.runvalues['lambdas_c'] and self.runvalues['lambdas_h'] if self.methodparams['do_mcsampl'] == True"""
        # If MC sampling is on, use lambas_h and lambdas_c as accepted moves may be different so final lambdas may be different
        if self.methodparams['do_mcsampl']:
            self.mcsamplvalues['gamma_MClambdas_h'] = self.runparams['gamma'] * target_lambdas_h
            self.mcsamplvalues['gamma_MClambdas_c'] = self.runparams['gamma'] * target_lambdas_c
            self.mcsamplvalues['gamma_MClambdas'] = 0.5 * ((target_lambdas_h / self.methodparams['bv_bh']) +
                                                           (target_lambdas_c / self.methodparams['bv_bc']))

            # Calc example stepsize & make move in lambdas
            ave_deviation = np.sum(np.abs(target_lambdas)) / np.sum(self.mcsamplvalues['gamma_MClambdas'] != 0)
            self.runvalues['curr_lambda_stepsize'] = self.methodparams['stepfactor'] / (self.runparams['gamma'] * ave_deviation * self.runvalues['ave_sigma_lnpi']) # Example stepsize based on ave_sigma_lnpi
            self.mcsamplvalues['lambdas_c'] = self.mcsamplvalues['lambdas_c'] * (1.0 - self.runvalues['curr_lambda_stepsize']) + \
                                              (self.runvalues['curr_lambda_stepsize'] * self.mcsamplvalues['gamma_MClambdas_c'])
            self.mcsamplvalues['lambdas_h'] = self.mcsamplvalues['lambdas_h'] * (1.0 - self.runvalues['curr_lambda_stepsize']) + \
                                              (self.runvalues['curr_lambda_stepsize'] * self.mcsamplvalues['gamma_MClambdas_h'])
            self.runvalues['lambdas'] = self.runvalues['lambdas'] * (1.0 - self.runvalues['curr_lambda_stepsize']) + \
                                        (self.runvalues['curr_lambda_stepsize'] * self.mcsamplvalues['gamma_MClambdas'])
        # All other cases, lambdas should be same for H and C
        else:
            gamma_target_lambdas = self.runparams['gamma'] * target_lambdas
            # Calc example stepsize & make move in lambdas
            ave_deviation = np.sum(np.abs(target_lambdas)) / np.sum(gamma_target_lambdas != 0)
            self.runvalues['curr_lambda_stepsize'] = self.methodparams['stepfactor'] / (self.runparams['gamma'] * ave_deviation * self.runvalues['ave_sigma_lnpi'])  # Example stepsize based on ave_sigma_lnpi
            self.runvalues['lambdas'] = self.runvalues['lambdas'] * (1.0 - self.runvalues['curr_lambda_stepsize']) + \
                                        (self.runvalues['curr_lambda_stepsize'] * gamma_target_lambdas)

    def make_iteration(self):
        """Perform a single HDXer iteration. One iteration consists of up to 4 steps,
           depending on the options chosen for the run in the MaxEnt.runparams and MaxEnt.methodparams dictionaries:

           1. Optimize beta parameters with current values of lambda & weights
           2. Recalculate lambda values and make a step towards target
           3. Update ln(protection factors) and frame weights with current values of lambda
           4. Update predicted deuterated fractions and MSE to target data using current frame weights"""

        # Optimize/sample parameters
        if self.methodparams['do_mcsampl']:
            self.sample_parameters_MC()
        elif self.methodparams['do_mcmin']:
            self.optimize_parameters_MC()
        elif self.methodparams['do_params']:
            self.optimize_parameters_gradient()

        # Update lambdas
        if not self.methodparams['do_mcsampl']: # Lambdas are updated internally in the sampling function if needed
            if self.methodparams['do_reweight']:
                curr_target_lambdas = self.calc_lambdas()
                self.update_lambdas(curr_target_lambdas)

        # Update predicted values
        self.update_lnpi_and_weights()
        self.update_dfracs_and_mse()
        self.update_work()

        # Assess convergence
        if not self.methodparams['do_reweight']:
            self.runvalues['is_converged'] = True
        else:
            _prev_lambda_mod = self.runvalues['lambda_mod']
            _prev_delta_lambda_mod = self.runvalues['delta_lambda_mod']
            self.runvalues['lambda_mod'] = np.sum(np.abs(self.runvalues['lambdas']))
            self.runvalues['delta_lambda_mod'] =  (self.runvalues['lambda_mod'] - _prev_lambda_mod) / self.runvalues['lambda_mod']

            if self.runvalues['curriter'] >= 100:
                if self.runvalues['delta_lambda_mod'] * _prev_delta_lambda_mod < 0:  # if oscillations in sign of delta_lambda
                    self.methodparams['stepfactor'] =  self.methodparams['stepfactor'] / self.methodparams['stepfactor_scaling']

            if np.abs(self.runvalues['delta_lambda_mod']) < self.methodparams['tolerance']:
                self.runvalues['is_converged'] = True
            if self.runvalues['lambda_mod'] < self.methodparams['tolerance']:
                self.runvalues['is_converged'] = True

        # Increase iteration count
        self.runvalues['curriter'] += 1

    def write_iteration(self):
        """Write a single line of iteration output to an all-steps log file"""
        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("%6d %8.6f %8.6f %10.8e %10.8e %10.8e %10.8e %10.8e \n" % (self.runvalues['curriter'],
                                                                               self.runvalues['curr_MSE'],
                                                                               np.sqrt(self.runvalues['curr_MSE']),
                                                                               self.runvalues['lambda_mod'],
                                                                               self.runvalues['delta_lambda_mod'],
                                                                               self.runvalues['curr_lambda_stepsize'],
                                                                               self.methodparams['bv_bc'],
                                                                               self.methodparams['bv_bh']))

    def write_restart(self):
        """Write a single line of iteration output to a restart log file and save a restart pickle file"""
        with open("%sper_restart_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("%6d %8.6f %8.6f %10.8e %10.8e %10.8e %10.8e %10.8e %8.5f \n" % (self.runvalues['curriter'],
                                                                                     self.runvalues['curr_MSE'],
                                                                                     np.sqrt(self.runvalues['curr_MSE']),
                                                                                     self.runvalues['lambda_mod'],
                                                                                     self.runvalues['delta_lambda_mod'],
                                                                                     self.runvalues['curr_lambda_stepsize'],
                                                                                     self.methodparams['bv_bc'],
                                                                                     self.methodparams['bv_bh'],
                                                                                     self.runvalues['work']))
        with open("%srestart.pkl" % self.runparams['out_prefix'], 'wb') as fpkl:
            pickle.dump(self.__dict__, fpkl, protocol=-1) # -1 for size purposes

    def run(self, gamma=10**-2, resultsobj=None, analysisobj=None, restart=None, **run_params):
        """Set up and perform a reweighting run.
        
            A reweighting run can be started in one of three ways, depending on the 
            arguments provided:

            1) Start a new run from scratch, reading input data from specified files
               Arguments required: 
                   data_folders : A list of folder paths containing 'Contacts_' and 'Hbond_' files
                                  for the initial structural ensemble
                   kint_file    : Path to a file containing intrinsic rates for each residue in the protein
                   exp_file     : Path to a file containing target (experimental) deuterated fractions
                   times        : List of deuteration timepoints in minutes, corresponding to the target 
                                  (experimental) data
               Optional arguments:
                   do_subsample (False) : Flag to turn on subsampling of the input trajectory frames
                   sub_start (0) : Starting index (inclusive) for subsampling the trajectory frames
                   sub_end (-1) : Ending index (exclusive, use '-1' to denote final frame) for subsampling the trajectory frames
                   sub_interval (1) : Interval for subsampling the trajectory frames
                   iniweights (None) : Initial weights for each trajectory frame, if each frame is not equally-weighted

            2) Start a new run from scratch, reading input data from calc_hdx objects
               Arguments required:
                   resultsobj   : A complete HDXer.methods.BV object, containing contacts, Hbonds,
                                  rates etc. attributes for the initial structural ensemble
                   analysisobj  : A complete HDXer.analysis.Analysis object, containing target
                                  (experimental) data etc. for the system of interest
               Optional arguments:
                   do_subsample (False) : Flag to turn on subsampling of the input trajectory frames
                   sub_start (0) : Starting index (inclusive) for subsampling the trajectory frames
                   sub_end (-1) : Ending index (exclusive, use '-1' to denote final frame) for subsampling the trajectory frames
                   sub_interval (1) : Interval for subsampling the trajectory frames
                   iniweights (None) : Initial weights for each trajectory frame, if each frame is not equally-weighted

            3) Restart a run that is already partially complete
               Arguments required:
                   restart      : Path to a HDXer.reweighting restart file, with '.pkl' suffix
               Optional arguments:
                   do_subsample (False) : Flag to turn on subsampling of the input trajectory frames
                   sub_start (0) : Starting index (inclusive) for subsampling the trajectory frames
                   sub_end (-1) : Ending index (exclusive, use '-1' to denote final frame) for subsampling the trajectory frames
                   sub_interval (1) : Interval for subsampling the trajectory frames"""

        # 0) Set basic parameters
        self.set_run_params(gamma, resultsobj, analysisobj, restart, run_params)

        # 1) Choose which setup to do. Restart > calc_hdx_objs > Normal
        if self.runparams['from_restart']:
            self.setup_restart(restart)
        elif self.runparams['from_calchdx']:
            self.setup_calc_hdx(resultsobj, analysisobj)
        else:
            try:
                self.setup_no_runobj(self.runparams['data_folders'],
                                     self.runparams['kint_file'],
                                     self.runparams['exp_file'],
                                     self.runparams['times'])
            except KeyError:
                raise HDX_Error("Missing parameters to set up a reweighting run.\n"
                                "Please ensure a restart or calc_hdx object is provided,"
                                "or provide the following arguments to the run() call: "
                                "data_folders, kint_file, exp_file, times")

        # 2) Do run
        # Set initial values of weights, lnpi, ave_lnpi, dfracs, and mse on first iteration
        if self.runvalues['curriter'] == 0:
            self.update_lnpi_and_weights()
            self.update_dfracs_and_mse()
        # Set restart interval if not already set
        try:
            _ = self.runparams['restart_interval']
        except KeyError:
            if self.methodparams['maxiters'] > 1000:
                self.runparams['restart_interval'] = int(self.methodparams['maxiters'] / 1000) # Max of 1000 restarts by default
            else:
                self.runparams['restart_interval'] = self.methodparams['maxiters']
        # Do iterations until EITHER maxiters reached or convergence
        while self.runvalues['curriter'] <= self.methodparams['maxiters'] and self.runvalues['is_converged'] == False:
            self.make_iteration()
            self.write_iteration()
            if (self.runvalues['curriter'] % self.runparams['restart_interval']) == 0:
                self.write_restart()

        # 3) Do final save/cleanup
        # If we calculate MSE to expt from the final segments printed out we'll get a different MSE
        # to that in the work file. Why? Because work file MSE is weighted by the length of the segments.
        np.savetxt("%sfinal_segment_fractions.dat" % self.runparams['out_prefix'],
                   np.sum(self.runvalues['curr_segment_dfracs'] * self.runvalues['segfilters'], axis=1)
                   / np.sum(self.runvalues['segfilters'], axis=1),
                   header="Times: " + " ".join(str(t) for t in self.runparams['times']), fmt="%8.5f")
        np.savetxt("%sfinal_weights.dat" % self.runparams['out_prefix'], self.runvalues['currweights'])

        with open("%swork.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("%5.4e %8.6f %8.6f %8.5f \n" % (self.runparams['gamma'], self.runvalues['curr_MSE'],
                                                    np.sqrt(self.runvalues['curr_MSE']), self.runvalues['work']))

        with open("%sper_iteration_output.dat" % self.runparams['out_prefix'], 'a') as f:
            f.write("%6d %8.6f %8.6f %10.8e %10.8e %10.8e %10.8e %10.8e # FINAL values at convergence or max iterations\n"
                    % (self.runvalues['curriter'],
                       self.runvalues['curr_MSE'],
                       np.sqrt(self.runvalues['curr_MSE']),
                       self.runvalues['lambda_mod'],
                       self.runvalues['delta_lambda_mod'],
                       self.runvalues['curr_lambda_stepsize'],
                       self.methodparams['bv_bc'],
                       self.methodparams['bv_bh']))
