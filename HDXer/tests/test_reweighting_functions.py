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


