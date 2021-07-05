HDXer
==============================
## **Introduction**

***HDXer*** is a Python package that can be used to:
- Predict Hydrogen-Deuterium exchange (HDX) data from an atomistic ensemble of protein structures (e.g. from molecular dynamics simulations)
- Reweight a candidate ensemble by applying a maximum-entropy biasing scheme, so that the predicted HDX data conform to a target experimental set of HDX-MS data, within a defined level of error

---

## **Requirements**
- [Python 3 (preferably Anaconda Python 3.x version)](https://www.anaconda.com/distribution/#download-section)
- [git](https://git-scm.com/downloads)
- [NumPy](https://numpy.org/)
- [MDTraj](http://mdtraj.org/1.9.3/)
- [matplotlib](https://matplotlib.org)
- [pytest](https://docs.pytest.org/en/stable)
- [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)
- [HDXer](https://github.com/TMB-CSB/HDXer)

---

## **Installation**

You should run all the code shown below on terminal using Bash. Bash is a command language interpreter that's already available on Linux/Mac operating systems. See the [Bash tutorial](https://linuxconfig.org/bash-scripting-tutorial-for-beginners) if you are not familiar with Bash.

If you are using a Windows computer, you will have to download and use [Git for Windows](https://git-scm.com/download/win) instead of a terminal. Git for Windows will let you use both Bash and git on a Windows computer.

<br>

### Python 3

For Python 3, we recommend *Anaconda Python 3.x version*, a free and open-source distribution that comes with useful standard Python libraries. You will be able to download, access, and manage Python packages more effectively by using conda, a package manager within *Anaconda*. *Ananconda Python 3.x version* can be downloaded from the [Anaconda website](https://www.anaconda.com/distribution/#download-section).

[A user guide for Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf) are great resources if you are new to using *Anaconda* and want to learn more about it.

<br>

### git

*git* is a free and open-source distributed version-control system. 

Again, if you are using a Windows computer, download [Git for Windows](https://git-scm.com/download/win). Some operating systems (e.g. MacOS or Linux) may have *git* installed already. You can check this by running the following command:

```bash
git --version
```

If there is no *git* available on your computer, you can install it using conda, or your operating system's package manager.

```bash
conda install -c conda-forge git
```

If you encounter errors at this point, or using the ```conda``` commands below, you may need to update your ```$PATH``` environment variable and/or initialize ```conda``` on your system. The *Anaconda* package will usually provide an initialization script for this purpose, for example, on Windows available in:
```
~/Anaconda/etc/profile.d/conda.sh
```
You can call this script or other initialization command in your ```~/.bashrc``` file (for example, by running ```echo ". ~/Anaconda/etc/profile.d/conda.sh" >> ~/.bashrc```)  to make sure it's run every time you open a new Bash terminal. Check out the *Anaconda* documentation for your filesystem if you continue to have problems, as ```conda``` commands are required for the remaining parts of the installation.

<br>

### *HDXer* Python Package

Download the *HDXer* Python package using git clone.

```bash
git clone https://github.com/TMB-CSB/HDXer
```

Once the *HDXer* Python package is downloaded, create a new conda environment that includes all of the dependencies.

```bash
cd HDXer
conda env create -f HDXER_ENV.yml
```

Every time you use *HDXer*, you have to activate the *HDXER_ENV* using conda activate.

```bash
conda activate HDXER_ENV
```

Within the *HDXER_ENV* environment, you have all the dependencies available to install the *HDXer* Python package to your Python environment. *HDXer* is still in development, so we recommend installing the package in 'editable' mode, using either pip:

```bash
cd ..
pip install -e HDXer
```

or conda:

```bash
cd ..
conda develop HDXer
```

Installing in 'editable' mode will allow you to pull updates directly from this Github repository to your local *HDXer* installation, without having to reinstall the package.

Now, you finished installing the *HDXer* Python package. Let's add the path to the *HDXer* directory as in your *.bashrc* and/or *.bash_profile*. The *HDXer* directory will be used throughout the tutorials and referred to as \$HDXER_PATH.

```bash
cd HDXer

echo -e "\nexport HDXER_PATH='${PWD}'" >| ~/.HDXER_PATH_variable

grep -qxF "if [ -f ~/.HDXER_PATH_variable ]; then source ~/.HDXER_PATH_variable; fi" ~/.bashrc || echo -e "if [ -f ~/.HDXER_PATH_variable ]; then source ~/.HDXER_PATH_variable; fi" >> ~/.bashrc
source ~/.bashrc

grep -qxF "if [ -f ~/.HDXER_PATH_variable ]; then source ~/.HDXER_PATH_variable; fi" ~/.bash_profile || echo -e "if [ -f ~/.HDXER_PATH_variable ]; then source ~/.HDXER_PATH_variable; fi" >> ~/.bash_profile
source ~/.bash_profile
```

You can easily access the *HDXer* directory using the \$HDXER_PATH variable. For example, to move to the *HDXer* directory, you simply have to type the following command on terminal:

```bash
cd $HDXER_PATH
```

---

## **Testing**

After you've carried out the steps above, we recommend that you run the functional tests provided with HDXer to check for a successful installation. You can also use this as an opportunity to check your Bash and Conda environments have been successfully created. 

First, check the folder ```$HDXER_PATH/HDXer/tests/data/tmp_test_output``` is empty. This is where the test outputs will be stored, so remove any files from previous test runs before running the tests anew. Then, to run the tests, open a new Bash shell and type:

```bash
cd $HDXER_PATH
conda activate HDXER_ENV
pytest -v
```

The tests will take roughly 60 seconds to complete. All tests should pass. If you encounter any failures, you can check the output of the tests in the folder where they're stored (```$HDXER_PATH/HDXer/tests/data/tmp_test_output```) to help determine the cause of the error. If the failure persists and you believe there's a problem with HDXer, please contact the developers or raise an issue on this GitHub repository.

---


## **Tutorials**

The tutorials for the ***HDXer*** are available in a series of easy-to-follow Jupyter notebooks. These notebooks can be viewed within the [GitHub page](https://github.com/TMB-CSB/HDXer/tree/master/tutorials) or with Jupyter Lab. You will be able to run code interactively within each notebook using Jupyter Lab. Run the following commands on terminal to access the notebooks using Jupyter Lab:

```bash
cd $HDXER_PATH/tutorials/notebooks
jupyter lab
```

In the *notebooks/* directory, there are five Jupyter notebooks.

- 01_data_prep.ipynb
- 02_calc_hdx.ipynb
- 03_reweighting.ipynb
- 04_decision_plot.ipynb
- 05_heatmap.ipynb

These notebooks will walk you through how to run ***HDXer*** and how to analyze the results with an example application of both HDX predictions and ensemble reweighting.

N.B. As per the Jupyter website, the Jupyter Notebook aims to support the latest versions of these browsers:

- Chrome
- Safari
- Firefox

Up to date versions of Opera and Edge may also work, but if they don’t, please use one of the supported browsers.

---

### Copyright

Copyright (c) 2020, 2021, Richard T. Bradshaw, Fabrizio Marinelli, Paul Suhwan Lee


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
