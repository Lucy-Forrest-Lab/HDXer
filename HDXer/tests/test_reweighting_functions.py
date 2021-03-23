"""
Unit and regression test for the HDXer package.
"""

# Import package, test suite, and other packages as needed
from HDXer import reweighting_functions
import numpy as np
import os
from glob import glob


def test_read_contacts_hbonds():
    """Test the processing of contact/H-bond files from calc_hdx.
       Also covers strip_filename and files_to_array"""

    test_folder = 'HDXer/tests/data/reweighting_1'
    test_contacts_prefix = 'Contacts_chain_0_res_'
    test_hbonds_prefix = 'Hbonds_chain_0_res_'

    expected_resids = [ 2, 3, 4, 5, 6, 7 ]
    c_files = sorted(glob(os.path.join(test_folder, test_contacts_prefix+'*')))
    h_files = sorted(glob(os.path.join(test_folder, test_hbonds_prefix+'*')))
    out_c = [ reweighting_functions.strip_filename(fn, test_contacts_prefix) for fn in c_files ]
    out_h = [ reweighting_functions.strip_filename(fn, test_hbonds_prefix) for fn in h_files ]

    assert out_c == out_h == expected_resids

    # * 2 for double folders
    expected_hbonds = np.array([ 
                      [ 1, 1, 1, 0, 0, 2, 0, 0, 0, 0 ] * 2,
                      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ] * 2,
                      [ 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 ] * 2,
                      [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] * 2,
                      [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ] * 2,
                      [ 1, 0, 2, 0, 1, 0, 2, 0, 1, 0 ] * 2 ])

    expected_contacts = np.array([ 
                        [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ] * 2,
                        [ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 ] * 2,
                        [ 20, 20, 20, 20, 20, 10, 10, 10, 10, 10 ] * 2,
                        [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ] * 2,
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] * 2,
                        [ 4, 4, 22, 4, 11, 4, 23, 3, 10, 5 ] * 2 ])

    assert np.array_equal(reweighting_functions.files_to_array(c_files), expected_contacts[:,:10])
    assert np.array_equal(reweighting_functions.files_to_array(h_files), expected_hbonds[:,:10])

    expected_resids = [ [ 2, 3, 4, 5, 6, 7 ] , [ 2, 3, 4, 5, 6, 7, 8, 9 ] ]

    test_folders = [ 'HDXer/tests/data/reweighting_1', 
                     'HDXer/tests/data/reweighting_2' ]
    out_c, out_h, out_res = reweighting_functions.read_contacts_hbonds(test_folders,
                                                                       test_contacts_prefix,
                                                                       test_hbonds_prefix)
    assert np.array_equal(out_c, expected_contacts)
    assert np.array_equal(out_h, expected_hbonds)
    assert out_res == expected_resids


def test_read_kints_segments():
    """Test the processing of intrinsic rates files and target
       (experimental) HDX data files."""

    # First check all residues can be read in correctly
    test_folder = 'HDXer/tests/data/reweighting_1'
    test_rates_file = os.path.join(test_folder, 'intrinsic_rates.dat')
    test_expt_file = os.path.join(test_folder, 'experimental_data.dat')
    test_times = np.array([0.5, 5.0, 60.0])
    test_n_res = 6
    test_resids = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int16)  # 2D array as residues usually come from a list of input files

    expected_minuskt = np.array([10.0, 100.0, 1000.0, 0.1, 268.21, 8.5])
    expected_minuskt = np.repeat(expected_minuskt[:, np.newaxis], len(test_times), axis=1)*test_times
    expected_minuskt = expected_minuskt[np.newaxis,:,:].repeat(3, axis=0)  # Repeat 3 because there are 3 segments in this target data file
    expected_minuskt *= -1

    expected_expt = np.array([[ 0.72823031, 0.83269571, 0.97653617 ],
                              [ 0.75439649, 0.79088392, 0.97067021 ],
                              [ 0.99994366, 1.00000000, 1.00000000 ]])
    expected_expt = expected_expt[:,np.newaxis,:].repeat(test_n_res, axis=1)

    # Filter is hard coded to skip the first residue in the segment 
    expected_segfilters = np.array([[ False, True, True, True, True, True ],
                                    [ False, True, True, True, True, False ],
                                    [ False, False, True, True, False, False ]])
    expected_segfilters = np.repeat(expected_segfilters[:, :, np.newaxis], len(test_times), axis=2)

    out_minuskt, out_expt, out_segfilters = reweighting_functions.read_kints_segments(test_rates_file,
                                                                                      test_expt_file,
                                                                                      test_n_res,
                                                                                      test_times,
                                                                                      test_resids)
    assert np.array_equal(out_minuskt, expected_minuskt)
    assert np.array_equal(out_expt, expected_expt)
    assert np.array_equal(out_segfilters, expected_segfilters)


def test_generate_trial_betas():
    """Test the generation of trial beta_H and beta_H
       parameters for Monte Carlo sampling."""

    # Check using default values from MaxEnt class
    test_bc, test_bh = 0.35, 2.0
    test_bcrange, test_bhrange = 1.5, 16.0
    test_step_multiplier = 0.1
    test_random_state = 561113

    # Based on np.random.RandomState(561113).random_sample()
    expected_bc, expected_bh = (0.31793770441840996, 2.041046758186777)

    out_bc, out_bh = reweighting_functions.generate_trial_betas(test_bc, test_bh,
                                                                test_bcrange, test_bhrange,
                                                                test_step_multiplier,
                                                                test_random_state)
    assert out_bc == expected_bc and out_bh == expected_bh
    
    # Check data cleanup with -ve bc
    test_bc = -0.01

    # Based on np.random.RandomState(561113).random_sample()
    expected_bc, expected_bh = (0.0018941199468254567, 2.041046758186777)

    out_bc, out_bh = reweighting_functions.generate_trial_betas(test_bc, test_bh,
                                                                test_bcrange, test_bhrange,
                                                                test_step_multiplier,
                                                                test_random_state)
    assert out_bc == expected_bc and out_bh == expected_bh

    # Finally check some extreme value of params & step size
    test_bc, test_bh = 1000.5, 10**9
    test_step_multiplier = 1000

    # Based on np.random.RandomState(561113).random_sample()
    expected_bc, expected_bh = (679.8770441840998, 1000000410.4675819)

    out_bc, out_bh = reweighting_functions.generate_trial_betas(test_bc, test_bh,
                                                                test_bcrange, test_bhrange,
                                                                test_step_multiplier,
                                                                test_random_state)
    assert out_bc == expected_bc and out_bh == expected_bh


def test_calc_trial_ave_lnpi():
    """Test the calculation of trial ave_lnpi
       and reshaping to [n_segments, n_residues, n_times].
       Should be a very basic function!"""

    test_hbonds = np.array([ 
                  [ 1, 1, 1, 0, 0, 2, 0, 0, 0, 0 ],
                  [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
                  [ 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 ],
                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                  [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                  [ 1, 0, 2, 0, 1, 0, 2, 0, 1, 0 ] ])

    test_contacts = np.array([ 
                    [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                    [ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 ],
                    [ 20, 20, 20, 20, 20, 10, 10, 10, 10, 10 ],
                    [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                    [ 4, 4, 22, 4, 11, 4, 23, 3, 10, 5 ] ])

    test_ave_contacts = np.mean(test_contacts, axis=1)
    test_ave_hbonds = np.mean(test_hbonds, axis=1)
    test_bc, test_bh = 0.35, 2.0
    test_n_times, test_n_segs = 3, 2

    # Array of (bc * ave_contacts) + (bh * ave_hbonds) in shape [n_segs, n_residues, n_times]
    expected_ave_lnpi = np.array([[[3.8 , 3.8 , 3.8 ],
                                   [5.5 , 5.5 , 5.5 ],
                                   [8.25, 8.25, 8.25],
                                   [2.8 , 2.8 , 2.8 ],
                                   [0.4 , 0.4 , 0.4 ],
                                   [4.55, 4.55, 4.55]],

                                  [[3.8 , 3.8 , 3.8 ],
                                   [5.5 , 5.5 , 5.5 ],
                                   [8.25, 8.25, 8.25],
                                   [2.8 , 2.8 , 2.8 ],
                                   [0.4 , 0.4 , 0.4 ],
                                   [4.55, 4.55, 4.55]]])

    out_ave_lnpi = reweighting_functions.calc_trial_ave_lnpi(test_ave_contacts, 
                                                             test_ave_hbonds, 
                                                             test_bc, test_bh, 
                                                             test_n_times, test_n_segs)
    assert np.array_equal(out_ave_lnpi, expected_ave_lnpi)


def test_calc_trial_dfracs():
    """Test the calculation of trial dfracs and MSE
       and reshaping to [n_segments, n_residues, n_times]."""

    test_ave_lnpi = np.array([[[3.8 , 3.8 , 3.8 ],
                               [5.5 , 5.5 , 5.5 ],
                               [8.25, 8.25, 8.25],
                               [2.8 , 2.8 , 2.8 ],
                               [0.4 , 0.4 , 0.4 ],
                               [4.55, 4.55, 4.55]],

                              [[3.8 , 3.8 , 3.8 ],
                               [5.5 , 5.5 , 5.5 ],
                               [8.25, 8.25, 8.25],
                               [2.8 , 2.8 , 2.8 ],
                               [0.4 , 0.4 , 0.4 ],
                               [4.55, 4.55, 4.55]]])
    
    test_segfilters = np.array([[[0, 0, 0],
                                 [0, 0, 0],
                                 [1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]],

                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]]], dtype=bool)
    test_times = np.array([0.5, 5.0, 60.0])
    test_minuskt = np.array([10.0, 100.0, 1000.0, 0.1, 268.21, 8.5])
    test_minuskt = np.repeat(test_minuskt[:, np.newaxis], len(test_times), axis=1)*test_times
    test_minuskt = test_minuskt[np.newaxis,:,:].repeat(test_segfilters.shape[0], axis=0)
    test_minuskt *= -1

    test_expt = np.array([[ 0.72823031, 0.83269571, 0.97653617 ],
                          [ 0.75439649, 0.79088392, 0.97067021 ]])
    test_expt = test_expt[:,np.newaxis,:].repeat(6, axis=1)

    test_n_datapoints = np.sum(test_segfilters)

    expected_residue_dfracs = np.array([[[np.nan, np.nan, np.nan],
                                         [np.nan, np.nan, np.nan],
                                         [0.12245696, 0.72917781, 0.99999984],
                                         [0.00303589, 0.02994745, 0.30570642],
                                         [1.0, 1.0, 1.0],
                                         [0.04391707, 0.36180167, 0.99543471]],
 
                                        [[np.nan, np.nan, np.nan],
                                         [np.nan, np.nan, np.nan],
                                         [np.nan, np.nan, np.nan],
                                         [0.00303589, 0.02994745, 0.30570642],
                                         [1.0, 1.0, 1.0],
                                         [0.04391707, 0.36180167, 0.99543471]]])
    expected_segment_dfracs = np.array([[[0.29235248, 0.53023173, 0.82528524],
                                         [0.29235248, 0.53023173, 0.82528524],
                                         [0.29235248, 0.53023173, 0.82528524],
                                         [0.29235248, 0.53023173, 0.82528524],
                                         [0.29235248, 0.53023173, 0.82528524],
                                         [0.29235248, 0.53023173, 0.82528524]],
 
                                        [[0.34898432, 0.46391637, 0.76704704],
                                         [0.34898432, 0.46391637, 0.76704704],
                                         [0.34898432, 0.46391637, 0.76704704],
                                         [0.34898432, 0.46391637, 0.76704704],
                                         [0.34898432, 0.46391637, 0.76704704],
                                         [0.34898432, 0.46391637, 0.76704704]]])
    expected_mse = 0.1026471781333345

    out_residue_dfracs, out_segment_dfracs, out_mse = reweighting_functions.calc_trial_dfracs(test_ave_lnpi, 
                                                                                              test_segfilters, 
                                                                                              test_minuskt*test_segfilters, 
                                                                                              test_expt * test_segfilters, 
                                                                                              test_n_datapoints)

    print(out_residue_dfracs)
    assert np.allclose(out_residue_dfracs, expected_residue_dfracs, equal_nan=True)
    assert np.allclose(out_segment_dfracs, expected_segment_dfracs)
    assert out_mse == expected_mse
    

def test_calc_work():
    """Tests that apparent work is calculated correctly
       from a given set of initial lnpi, lambdas, weights, and value of kT"""

    # First, check work returns 0 for 0 lambdas & equal weights
    test_hbonds = np.array([
                  [ 1, 1, 1, 0, 0, 2, 0, 0, 0, 0 ],
                  [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
                  [ 2, 2, 2, 2, 2, 1, 1, 1, 1, 1 ],
                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                  [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                  [ 1, 0, 2, 0, 1, 0, 2, 0, 1, 0 ] ])

    test_contacts = np.array([
                    [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                    [ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 ],
                    [ 20, 20, 20, 20, 20, 10, 10, 10, 10, 10 ],
                    [ 10, 10, 10, 5, 5, 20, 5, 5, 5, 5 ],
                    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                    [ 4, 4, 22, 4, 11, 4, 23, 3, 10, 5 ] ])
    test_bc, test_bh = 0.35, 2.0
    test_lnpi = (test_bc * test_contacts) + (test_bh * test_hbonds)

    # weights = 0.1 each
    test_weights = np.ones(10) / 10
    # 6 residues
    test_lambdas = np.zeros(6) 
    test_kT = 2.4942  # for kJ/mol @ 300 K

    expected_work = 0.0
    
    out_work = reweighting_functions.calc_work(test_lnpi, test_lambdas, test_weights, test_kT)
    assert out_work == expected_work

    # Now check work with made up lambdas & weights
    test_lambdas = np.array([0.0, 0.1, 0.1, 1.0, 0.0001, 0.25392])
    test_weights = np.array([0.0001, 0.0099, 0.090, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.4])
    assert np.sum(test_weights) == 1.0

    expected_work = 2.294138034922346

    out_work = reweighting_functions.calc_work(test_lnpi, test_lambdas, test_weights, test_kT)

    assert out_work == expected_work
