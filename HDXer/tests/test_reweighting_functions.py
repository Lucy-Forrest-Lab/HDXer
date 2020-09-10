"""
Unit and regression test for the HDXer package.
"""

# Import package, test suite, and other packages as needed
from HDXer import reweighting_functions
import numpy as np
import pytest
import sys, os
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
    out_c, out_h, out_res = reweighting_functions.read_contacts_hbonds(test_folders, test_contacts_prefix, test_hbonds_prefix)
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
    test_resids = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int16) # 2D array as residues usually come from a list of input files

    expected_minuskt = np.array([10.0, 100.0, 1000.0, 0.1, 268.21, 8.5])
    expected_minuskt = np.repeat(expected_minuskt[:, np.newaxis], len(test_times), axis=1)*test_times
    expected_minuskt = expected_minuskt[np.newaxis,:,:].repeat(3, axis=0) # Repeat 3 because there are 3 segments in this target data file
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


