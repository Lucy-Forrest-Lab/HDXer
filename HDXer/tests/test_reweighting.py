"""
Unit and regression test for the HDXer package.
"""

# Import package, test suite, and other packages as needed
from HDXer import reweighting
import numpy as np
import pytest
import sys, os

def test_reweight_initialize():
    """Tests the initialization and method parameters of a
       MaxEnt reweighting object"""

    test_dict = { 'temp' : 350,
                  'do_reweight' : False,
                  'test_param' : 'foo' }

    expected_dict = { 'do_reweight' : False,
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
                      'temp' : 350,
                      'random_initial' : False,
                      'kT' : 2.9101093,
                      'test_param' : 'foo'
                         }

    test_obj = reweighting.MaxEnt(**test_dict)
    assert test_obj.methodparams == expected_dict

def test_reweight_data_io_1():
    """Test that runparams are assigned correctly and
       data files are read in correctly.
       Test 1 checks contacts & H-bond inputs"""

    expected_nframes = 10
    expected_hbonds = np.array([ 
                      [ 1, 1, 1, 0, 0, 2, 0, 0, 0, 0 ],
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
                      [ 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 ],
                      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                      [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                      [ 1, 0, 2, 0, 1, 0, 2, 0, 1, 0 ] ])

    expected_contacts = np.array([ 
                        [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                        [ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 ],
                        [ 20, 20, 20, 20, 20, 10, 10, 10, 10, 10 ],
                        [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                        [ 4, 4, 22, 4, 11, 4, 23, 3, 10, 5 ] ])

    expected_kint = np.array([10.0, 100.0, 1000.0, 0.1, 268.21, 8.5]) 
    
    test_times = np.array([ 0.5, 5.0, 60.0 ])
    
    expected_minuskt = -expected_kint[:,np.newaxis] * test_times

    expected_expt = np.array([
                    [ 0.72823031, 0.83269571, 0.97653617 ],
                    [ 0.75439649, 0.79088392, 0.97067021 ],
                    [ 0.99994366, 1.00000000, 1.00000000 ] ])

    


    test_folders = [ 'HDXer/tests/data/reweighting_1' ]
    test_kint_file = os.path.join('HDXer/tests/data/reweighting_1', 'intrinsic_rates.dat')
    test_exp_file = os.path.join('HDXer/tests/data/reweighting_1', 'experimental_data.dat')
    test_contacts_prefix = 'Contacts_chain_0_res_'
    test_hbonds_prefix = 'Hbonds_chain_0_res_'
    test_param_dict = { 'hbonds_prefix' : test_hbonds_prefix,
                        'contacts_prefix' : test_contacts_prefix,
                        'data_folders' : test_folders,
                        'kint_file' : test_kint_file,
                        'exp_file' : test_exp_file,
                        'times' : test_times }


    test_obj = reweighting.MaxEnt()
    test_obj.set_run_params(10**-2, None, None, test_param_dict)
    test_obj.setup_no_runobj(test_obj.runparams['data_folders'],
                             test_obj.runparams['kint_file'],
                             test_obj.runparams['exp_file'],
                             test_obj.runparams['times'])
            

    assert np.array_equal(test_obj.runvalues['contacts'], expected_contacts)
    assert np.array_equal(test_obj.runvalues['hbonds'], expected_hbonds)
    assert test_obj.runvalues['nframes'] == expected_nframes
    # Broadcast expected_minuskt to 3 dims for segments * residues * times 
    assert np.array_equal(test_obj.runvalues['minuskt'], np.repeat(expected_minuskt[np.newaxis,:,:], len(expected_expt), axis=0) )
    # Internally the exp_dfrac_filtered arrays have 0 for residues not in the segment.
    # So, test for closeness, not identity
    test_expt_normalised = np.sum(test_obj.runvalues['exp_dfrac_filtered'], axis=1) / np.sum(test_obj.runvalues['segfilters'], axis=1)
    assert np.allclose(test_expt_normalised, expected_expt)

def test_update_lnpi_1():
    """Test that iniweights, radou parameters are assigned correctly
       and that weights are calculated correctly with lambda = 0"""

    expected_iniweights = np.ones(10)
    expected_currweights = np.ones(10) / 10
    expected_lnpi = np.array([[5.5, 5.5, 5.5, 1.75, 1.75, 11., 1.75, 1.75, 1.75, 1.75],
                              [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5],
                              [11., 11., 11., 11., 11., 5.5, 5.5, 5.5, 5.5, 5.5],
                              [3.5, 3.5, 3.5, 1.75, 1.75, 7., 1.75, 1.75, 1.75, 1.75],
                              [0. , 2., 0., 0., 0., 0., 0., 0., 2., 0.],
                              [3.4, 1.4, 11.7, 1.4, 5.85, 1.4, 12.05, 1.05, 5.5, 1.75]])
    expected_avelnpi = np.array([3.8 , 5.5 , 8.25, 2.8 , 0.4 , 4.55])
    # Calculated with Bc = 0.25, Bh = 5.25
    expected_lnpi_newbetas = np.array([[7.75, 7.75, 7.75, 1.25, 1.25, 15.5, 1.25, 1.25, 1.25, 1.25],
                                       [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75],
                                       [15.5, 15.5, 15.5, 15.5, 15.5, 7.75, 7.75, 7.75, 7.75, 7.75],
                                       [2.5, 2.5, 2.5, 1.25, 1.25, 5., 1.25, 1.25, 1.25, 1.25],
                                       [0., 5.25, 0., 0., 0., 0., 0., 0., 5.25, 0.],
                                       [6.25, 1., 16., 1., 8., 1., 16.25, 0.75, 7.75, 1.25]])
    expected_avelnpi_newbetas = np.array([ 4.625,  7.75 , 11.625,  2.   ,  1.05 ,  5.925])




    test_folders = [ 'HDXer/tests/data/reweighting_1' ]
    test_kint_file = os.path.join('HDXer/tests/data/reweighting_1', 'intrinsic_rates.dat')
    test_exp_file = os.path.join('HDXer/tests/data/reweighting_1', 'experimental_data.dat')
    test_times = np.array([ 0.5, 5.0, 60.0 ])
    test_contacts_prefix = 'Contacts_chain_0_res_'
    test_hbonds_prefix = 'Hbonds_chain_0_res_'
    test_param_dict = { 'hbonds_prefix' : test_hbonds_prefix,
                        'contacts_prefix' : test_contacts_prefix,
                        'data_folders' : test_folders,
                        'kint_file' : test_kint_file,
                        'exp_file' : test_exp_file,
                        'times' : test_times }


    test_obj = reweighting.MaxEnt()
    test_obj.set_run_params(10**-2, None, None, test_param_dict)
    test_obj.setup_no_runobj(test_obj.runparams['data_folders'],
                             test_obj.runparams['kint_file'],
                             test_obj.runparams['exp_file'],
                             test_obj.runparams['times'])

    test_obj.update_lnpi_and_weights()
    assert np.array_equal(test_obj.runvalues['iniweights'], expected_iniweights)
    assert np.allclose(test_obj.runvalues['lnpi'], expected_lnpi)
    assert np.allclose(test_obj.runvalues['ave_lnpi'], expected_avelnpi)
    test_obj.methodparams['radou_bc'] = 0.25
    test_obj.methodparams['radou_bh'] = 5.25
    test_obj.update_lnpi_and_weights()
    assert np.allclose(test_obj.runvalues['lnpi'], expected_lnpi_newbetas)
    assert np.allclose(test_obj.runvalues['ave_lnpi'], expected_avelnpi_newbetas)
    assert np.array_equal(test_obj.runvalues['currweights'], expected_currweights)

def test_update_lnpi_2():
    """Test that weights are calculated correctly with provided lambdas"""

    test_folders = [ 'HDXer/tests/data/reweighting_1' ]
    test_kint_file = os.path.join('HDXer/tests/data/reweighting_1', 'intrinsic_rates.dat')
    test_exp_file = os.path.join('HDXer/tests/data/reweighting_1', 'experimental_data.dat')
    test_times = np.array([ 0.5, 5.0, 60.0 ])
    test_contacts_prefix = 'Contacts_chain_0_res_'
    test_hbonds_prefix = 'Hbonds_chain_0_res_'
    test_param_dict = { 'hbonds_prefix' : test_hbonds_prefix,
                        'contacts_prefix' : test_contacts_prefix,
                        'data_folders' : test_folders,
                        'kint_file' : test_kint_file,
                        'exp_file' : test_exp_file,
                        'times' : test_times }
    test_methodparam_dict = { 'radou_bc' : 0.25,
                              'radou_bh' : 5.25 }
    test_obj = reweighting.MaxEnt(**test_methodparam_dict)
    test_obj.set_run_params(10**-2, None, None, test_param_dict)
    test_obj.setup_no_runobj(test_obj.runparams['data_folders'],
                             test_obj.runparams['kint_file'],
                             test_obj.runparams['exp_file'],
                             test_obj.runparams['times'])

   # Now apply lambdas and re-test weights are calculated correctly

    test_lambdas = np.ones(6)
    test_obj.runvalues['lambdas'] = test_lambdas
    test_obj.update_lnpi_and_weights()

    # Calculated with Bc = 0.25, Bh = 5.25
    expected_lnpi_newbetas = np.array([[7.75, 7.75, 7.75, 1.25, 1.25, 15.5, 1.25, 1.25, 1.25, 1.25],
                                       [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 7.75],
                                       [15.5, 15.5, 15.5, 15.5, 15.5, 7.75, 7.75, 7.75, 7.75, 7.75],
                                       [2.5, 2.5, 2.5, 1.25, 1.25, 5., 1.25, 1.25, 1.25, 1.25],
                                       [0., 5.25, 0., 0., 0., 0., 0., 0., 5.25, 0.],
                                       [6.25, 1., 16., 1., 8., 1., 16.25, 0.75, 7.75, 1.25]])

    # Calculated manually as code below:
    # biasfactor = np.sum(test_lambdas[:, np.newaxis] * expected_lnpi_newbetas, axis=0) 
    # expected_currweights = expected_iniweights * np.exp(biasfactor)
    # expected_currweights = expected_currweights / np.sum(expected_currweights)
    # expected_avelnpi_newbetas = np.sum(expected_currweights * expected_lnpi_newbetas, axis=1)
    expected_currweights = np.array([5.82876279e-05, 5.82876279e-05, 9.99879306e-01, 
                                     1.31749240e-10, 1.44480585e-07, 3.72620339e-06, 
                                     2.38208213e-07, 4.41969461e-14, 9.23633476e-09, 7.28684451e-14])
    expected_avelnpi_newbetas = np.array([7.75002633e+00, 7.75000000e+00, 1.54999692e+01, 
                                          2.50000883e+00, 3.06058537e-04, 1.59985003e+01])

    assert np.allclose(test_obj.runvalues['lnpi'], expected_lnpi_newbetas)
    assert np.allclose(test_obj.runvalues['ave_lnpi'], expected_avelnpi_newbetas)
    assert np.allclose(test_obj.runvalues['currweights'], expected_currweights)
    assert np.isclose(np.sum(test_obj.runvalues['currweights']), 1.0)

    # Case with -ve lambda, should still work
    test_lambdas = np.array([0, 0, -0.1, 0.5, 0.00005, 2])
    test_obj.runvalues['lambdas'] = test_lambdas
    test_obj.update_lnpi_and_weights()

    # Calculated manually as code below:
    # biasfactor = np.sum(test_lambdas[:, np.newaxis] * expected_lnpi_newbetas, axis=0) 
    # expected_currweights = expected_iniweights * np.exp(biasfactor)
    # expected_currweights = expected_currweights / np.sum(expected_currweights)
    # expected_avelnpi_newbetas = np.sum(expected_currweights * expected_lnpi_newbetas, axis=1)
    expected_currweights = np.array([1.16557025e-09, 3.21040924e-14, 3.42989521e-01, 
                                     1.71795721e-14, 2.06602270e-08, 2.43160328e-13, 
                                     6.57010430e-01, 2.26174337e-14, 2.72069635e-08, 6.14805589e-14])
    expected_avelnpi_newbetas = np.array([3.47943189e+00, 7.75000000e+00, 1.04081690e+01, 
                                          1.67873690e+00, 1.42836727e-07, 1.61642522e+01])

    assert np.allclose(test_obj.runvalues['lnpi'], expected_lnpi_newbetas)
    assert np.allclose(test_obj.runvalues['ave_lnpi'], expected_avelnpi_newbetas)
    assert np.allclose(test_obj.runvalues['currweights'], expected_currweights)
    assert np.isclose(np.sum(test_obj.runvalues['currweights']), 1.0)
