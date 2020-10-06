"""
hdxer.py
HDXer is a package to predict Hydrogen-Deuterium exchange data from biomolecular simulations, compare to experiment, and perform ensemble refinement to fit a structural ensemble to the experimental data

Handles some wrapper functions
"""

def helptext():
    """Prints helptext explaining the hdxer package"""
    helptext = ("Welcome to HDXer!\n"
               "A package to predict Hydrogen-Deuterium exchange data from biomolecular simulations,\n"
               "compare to experiment, and perform ensemble refinement to fit a structural ensemble\n"
               "to the experimental data.\n\n"
               "If you are lost, please see the webpage "
               "github.com/TMB-CSB/HDXer for details and example uses")
    return helptext



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(helptext())
