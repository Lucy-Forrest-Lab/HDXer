"""
HDXer
HDXer is a package to predict Hydrogen-Deuterium exchange data from biomolecular simulations, compare to experiment, and perform ensemble refinement to fit a structural ensemble to the experimental data
"""

# Add imports here
from .hdxer import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
