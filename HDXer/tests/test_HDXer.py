"""
Unit and regression test for the HDXer package.
"""

# Import package, test suite, and other packages as needed
import HDXer
import pytest
import sys

def test_HDXer_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "HDXer" in sys.modules

def test_HDXer_helptext():
    expected_helptext = ("Welcome to HDXer!\n"
               "A package to predict Hydrogen-Deuterium exchange data from biomolecular simulations,\n"
               "compare to experiment, and perform ensemble refinement to fit a structural ensemble\n"
               "to the experimental data.\n\n"
               "If you are lost, please see the webpage "
               "github.com/TMB-CSB/HDXer for details and example uses")
    out_helptext = HDXer.hdxer.helptext()
    assert out_helptext == expected_helptext
