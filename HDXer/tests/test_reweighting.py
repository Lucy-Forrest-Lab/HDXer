"""
Unit and regression test for the HDXer package.
"""

# Import package, test suite, and other packages as needed
from HDXer import Reweighting
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

    test_obj = Reweighting.MaxEnt(**test_dict)
    assert test_obj.methodparams == expected_dict

def test_reweight_data_io_1():
    """Test that runparams are assigned correctly and
       data files are read in correctly"""

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

    


    test_folders = [ 'tests/data/reweighting_1' ]
    test_kint_file = os.path.join('tests/data/reweighting_1', 'intrinsic_rates.dat')
    test_exp_file = os.path.join('tests/data/reweighting_1', 'experimental_data.dat')
    test_contacts_prefix = 'Contacts_chain_0_res_'
    test_hbonds_prefix = 'Hbonds_chain_0_res_'
    test_param_dict = { 'hbonds_prefix' : test_hbonds_prefix,
                        'contacts_prefix' : test_contacts_prefix,
                        'data_folders' : test_folders,
                        'kint_file' : test_kint_file,
                        'exp_file' : test_exp_file,
                        'times' : test_times }


    test_obj = Reweighting.MaxEnt()
    test_obj.set_run_params(10**-2, None, None, test_param_dict)
    test_obj.setup_no_runobj(test_obj.runparams['data_folders'],
                             test_obj.runparams['kint_file'],
                             test_obj.runparams['exp_file'],
                             test_obj.runparams['times'])
            

    assert np.array_equal(test_obj.runvalues['contacts'], expected_contacts)
    assert np.array_equal(test_obj.runvalues['hbonds'], expected_hbonds)
    assert test_obj.runvalues['nframes'] == expected_nframes

