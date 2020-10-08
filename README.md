HDXer
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/rtb1c13/HDXer.svg?branch=master)](https://travis-ci.com/rtb1c13/HDXer)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/rtb1c13/HDXer/branch/master)
[![codecov](https://codecov.io/gh/rtb1c13/HDXer/branch/master/graph/badge.svg)](https://codecov.io/gh/rtb1c13/HDXer/branch/master)

HDXer is a package to predict Hydrogen-Deuterium exchange data from biomolecular simulations, compare to experiment, and perform ensemble refinement to fit a structural ensemble to the experimental data

## **Introduction**

***HDXer*** is composed of two Python 3 packages (***calc_hdx*** and ***ensemble_modeling***). The first Python package, ***calc_hdx***, predicts HDX deuterated fractions from structural ensembles (e.g. molecular dynamics). The subsequent Python package, ***ensemble_modeling***, applies maximum-entropy bias to the structural ensembles in order to reweight the ensembles to match up with experimental HDX-MS data.

---

## **Requirements**
- [Python 3 (preferably Anaconda Python 3.x version)](https://www.anaconda.com/distribution/#download-section)
- [git](https://git-scm.com/downloads)
- [SciPy](https://www.scipy.org/)
- [NumPy](https://numpy.org/)
- [MDTraj](http://mdtraj.org/1.9.3/)
- [matplotlib](https://matplotlib.org)
- [HDXer](https://github.com/TMB-CSB/HDXer)

---

## **Installation**

You should run all the code shown below on terminal using Bash.

Bash is a command language interpreter that's already available on Linux/Mac operating systems.

See the [Bash tutorial](https://linuxconfig.org/bash-scripting-tutorial-for-beginners) if you are not familiar with bBash.

If you are using Windows computer, you will have to download and use [Git for Windows](https://git-scm.com/download/win) instead of terminal.

Git for Windows will let you use both Bash and git on a Windows computer.

<br>

### Python 3

For Python 3, we recommend you using *Anaconda Python 3.x version*, a free and open-source distribution that comes with useful standard Python libraries.

You will be able to download, access, and manage Python packages more effectively by using conda, a package manager within *Anaconda*.

*Ananconda Python 3.x version* can be downloaded from the [Anaconda website](https://www.anaconda.com/distribution/#download-section).

[A user guide for Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) are great resources if you are new to using *Anaconda* and want to learn more about it.

<br>

### git

*git* is a free and open-source distributed version-control system.

Again, if you are using a Windows computer, download [Git for Windows](https://git-scm.com/download/win).

Some operating systems (e.g. MacOS or Linux) may have *git* installed already.

You can check this by running the following command:

```bash
git --version
```

If there is no *git* available on your computer, install using conda.

```bash
conda install -c conda-forge git
```

<br>

### *HDXer* Python Package

Download the *HDXer* Python package using git clone.

```bash
git clone https://github.com/TMB-CSB/HDXer
```

Once the *HDXer* Python package is downloaded, create a new conda environment that includes all of the dependencies.

```bash
cd HDXer
conda create -f HDXER_ENV.yml
```

Every time you use *HDXer*, you have to activate the *HDXER_ENV* using conda activate.

```bash
conda activate HDXER_ENV
```

Within the *HDXER_ENV* environment, you have all the dependencies available to install the *HDXer* Python package to your Python environment.

```bash
cd ..
pip install -e HDXer
```

Now, you finished installing the *HDXer* Python package. Let's add the path to the *HDXer* directory as in your *.bashrc* and/or *.bash_profile*. The *HDXer* directory you are in will be used throughout the protocol referred as \$HDXER_PATH.

```bash
cd HDXer
echo -e "\nexport HDXER_PATH='${PWD}'" >> ~/.bashrc
source ~/.bashrc
```

```bash
echo -e "\nexport HDXER_PATH='${PWD}'" >> ~/.bash_profile
source ~/.bash_profile
```

You can easily access the *HDXer* directory using the \$HDXER_PATH variable. For example, to move to the *HDXer* directory, you simply have to type the following command on terminal:

```bash
cd $HDXER_PATH
```

---

## **Protocol**

The protocol for ***HDXer*** is available in a series of easy to follow Jupyter notebooks. These notebooks can be accessed within the [GitHub page](https://github.com/rtb1c13/ensemble_modeling/tree/master/protocol) or with Jupyter Lab. You will be able to run code interactively within each notebook using Jupyter Lab. Run the following commands on terminal to access the notebooks using Jupyter Lab:

```bash
cd $HDXer/ensemble_modeling/protocol/notebooks
jupyter lab
```

In the *notebooks* directory, there are five Jupyter notebooks.

- 01_data_prep.ipynb
- 02_calc_hdx.ipynb
- 03_ensemble_modeling.ipynb
- 04_decision_plot.ipynb
- 05_heatmap.ipynb

These notebooks will walk you through how to run ***HDXer*** and analyze the results with an example.

### Copyright

Copyright (c) 2020, Richard T. Bradshaw, Fabrizio Marinelli


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
