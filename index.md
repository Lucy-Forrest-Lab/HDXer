## Welcome to the HDXer homepage

HDXer is a Python package to predict Hydrogen-Deuterium exchange data from biomolecular simulations, compare to experiment, and perform ensemble refinement to fit a simulated structural ensemble to the experimental data.

### License & citation

HDXer is released under the [BSD 3-Clause License](https://choosealicense.com/licenses/bsd-3-clause/). If you find HDXer useful in your work, we would be grateful if you could acknowledge the following article:

R.T. Bradshaw, F. Marinelli, J.D. Faraldo-Gómez & L.R. Forrest, [“Interpretation of HDX Data by Maximum-Entropy Reweighting of Simulated Structural Ensembles”](http://dx.doi.org/10.1016/j.bpj.2020.02.005), _Biophys. J._, 2020, **118 (7)**, 1649-1664

## FAQs

Below are some answers to frequently asked questions. If you have a problem or question that isn't answered here, please contact the authors, or raise an issue on the [HDXer Github](https://github.com/TMB-CSB/HDXer)

### Theory & software

**1. Where can I find out more about HDXer?**
  The theory and implementation of the Maximum Entropy ensemble reweighting approach used in HDXer is described in our Biophysical Journal article above. The article also contains investigations of the accuracy and structural fidelity of the approach, and an example application with experimental HDX-MS data. Further background information can be found in multiple excellent studies and reviews, concerning either [prediction](http://dx.doi.org/10.1042/BST20190880) of [deuterium](http://dx.doi.org/10.1021/acs.jpcb.8b07494) [exchange](http://dx.doi.org/10.1021/acs.biochem.5b00215), or ensemble refinement approaches to [incorporate](http://dx.doi.org/10.1063/1.4937786) [experimental](http://dx.doi.org/10.1016/j.sbi.2016.12.004) [data in](http://dx.doi.org/10.3390/computation6010015) [simulations](http://dx.doi.org/10.1016/bs.pmbts.2019.12.006).

**2. How can I obtain and install the software?**
  All versions of the HDXer package are available at the [HDXer Github repository](https://github.com/TMB-CSB/HDXer). Installation instructions are provided in the README file, visible on the main repository page.

**3. Are tests available to check my installation or code changes?**
  Yes, simple unit tests are provided that cover the major functions of the code. Application tests with example outputs are also provided to cover many smaller functions and identify changes in input/output formats. Tests are designed to help check that the code is numerically correct, and that any adaptations/updates have maintained correct functionality. Tests don’t thoroughly check that your Python environment is set up correctly, so please check this independently first (see the README on the Github for the environment requirements and dependencies).

  From the base HDXer directory, tests can be run using:
  ```
  pytest -v
  ```
  All tests should pass.

**4. Are there example uses or tutorials?**
  Yes, an example protocol is available in the `protocol/notebooks` sub-directory. The protocol uses HDXer to predict residue-based protection factors and HDX deuterated fractions from a simulated trajectory, and then performs ensemble reweighting to fit the structural ensemble to a set of experimental HDX data. The protocol is available as a series of interactive Jupyter notebooks, with input and output data provided.

### Usage practicalities

**5. How does HDXer predict HDX protection factors?**
  Two methods are available for calculation of protection factors from a trajectory of structural co-ordinates. The first, accessible as the `Radou` class in `methods.py`, calculates protection factors as a function of the hydrogen bonds and heavy atom contacts for each residue amide([1](http://dx.doi.org/10.1016/j.bpj.2014.06.039), [2](http://dx.doi.org/10.1016/j.str.2005.09.012), [3](http://dx.doi.org/10.1021/ja036523z). The second, available as the `PH` class, calculates protection factors as a function of counting ‘open’ and ‘closed’ trajectory frames for each residue, defined by the number of [amide-water contacts](http://dx.doi.org/10.1073/pnas.1506079112).

  Only the `Radou` methodology is currently valid for ensemble reweighting applications. During reweighting, protection factors are calculated using the ensemble-averaged contacts and hydrogen bonds for each residue amide.

**6. For reweighting, what structures should the initial structural ensemble contain? Can I use multiple simulations? Can I use homology models, or experimental structures, instead of simulations?**
  HDXer does not constrain the type or number of input files for HDX predictions and reweighting. So long as each frame can be described with the same topology (i.e. it has the same number of atoms, bonded in the same order), predictions and reweighting can be performed for the whole ensemble.

  However, please remember that Maximum Entropy reweighting approaches minimally bias the ensemble _as a whole_, such that the _ensemble average_ HDX data is fitted to the target. HDXer does not apply a Maximum Parsimony or similar approach to select a small number of best-fitting structures from a larger ensemble. Therefore, including large numbers of irrelevant or dissimilar structures in the initial ensemble will likely reduce the accuracy and structural fidelity of reweighting results, or even affect the ability of HDXer to converge upon a solution.

  A better approach to compare multiple structural ensembles may be to perform reweighting experiments for each ensemble individually. The bias required to fit each ensemble to the target data with the same accuracy (i.e. the same (root) mean square deviation) can then be compared.

**7. When running ensemble reweighting, how do I know the most appropriate gamma value for my system?**

  There’s no easy, general answer to this, however it is helpful to run a range of gamma values (all must be > 0) and construct a decision plot of Apparent Work vs. Mean Square Deviation to the target data. See the example protocol for details. A near-vertical line on this plot signifies it has become impossible to improve the fit to the experimental data by further increasing gamma. This implies the reweighting process has reached the uncertainty limit resulting from the experimental data, forward model, and initial ensemble. If the decision plot identifies a clear ‘elbow’ point, then it’s sensible to choose the smallest gamma value that gives the equivalent fit to the experimental data. If there is no clear ‘elbow’ in the data, we have generally chosen gamma values that result in Apparent Work between 1-5 kJ/mol.  The work value is a metric of how much bias has been applied to the initial ensemble, and for initial ensembles resulting from equilibrium MD simulations, we’ve often made the assumption that we do not wish to apply > 5 kJ/mol bias to our initial ensemble.

  Choosing the level of bias to apply is a common challenge with all reweighting & experimental biasing processes, and there’s a great discussion of the problem in some of the reviews linked above.

**8. How can I analyze my results to see the effects of reweighting?**
  HDXer provides output files containing the final segment-averaged predicted HDX data for comparison to the target data, and the final weights of each frame of the ensemble after reweighting. The final weights of each frame can be used to calculate the effects of reweighting on any structural metric of interest. For example, in the HDXer publication, we use the final weights to structurally cluster the final ensemble, and identify conformational populations resulting from the reweighting process. However, there are many further ways to assess the final ensemble – all that’s required is to compare a structural metric of interest (e.g. a distance, RMSD, secondary structure classification, etc.) for the initial ensemble where every frame is weighted equally, and for the final weighted ensemble where each frame has been assigned a different weight. Each protein system is different therefore, and you’ll have to make a subjective choice about the best way to analyze your data.

  If comparing multiple reweighting experiments against one another, e.g. using different target datasets or different initial ensembles, you should take care to ensure structural comparisons of the results are fair and robust. Final ensembles should be compared either at equivalent Apparent Work values, or equivalent MSD to the target data. Ideally, target datasets should contain identical peptides, so that each set of target HDX data has the same coverage and redundancy. Robustness of the results can then be tested by subsampling of the target dataset (e.g. removing peptides or timepoints), or of the initial structural ensemble (e.g. adding/removing simulation frames). It’s important to remember that ensemble reweighting does not absolutely determine the structure in best agreement with target data, but instead how to minimally bias the initial ensemble towards the target data. Multiple control reweighting experiments are therefore likely to be necessary to validate your structural interpretations. 

## Copyright

R.T. Bradshaw, F. Marinelli, S.P. Lee, 2020

National Institutes of Health, Bethesda, MD, USA

