# Synthetic signal propagation using synNotch receptors: data analysis and simulation

This package contains custom code used to run data analysis and plotting for the manuscript:

    Control of spatio-temporal patterning via cell density in a multicellular synthetic gene circuit
    Marco Santorelli, Pranav S. Bhamidipati, Andriu Kavanagh, Victoria A. MacKrell, Trusha Sondkar, Matt Thomson, Leonardo Morsut
    bioRxiv 2022.10.04.510900; doi: https://doi.org/10.1101/2022.10.04.510900

Run an interactive example: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pranav-bhamidipati/lateral-signaling/v0.2.0-pr3?labpath=lateral_signaling%2Fexample.ipynb)


### Supplementary data

[![DOI](https://data.caltech.edu/badge/DOI/10.22002/q8n10-tsk03.svg)](https://doi.org/10.22002/q8n10-tsk03)

To use this package, please also download the supplementary data (click the badge above). (`data_minimal` and `data` are identical except for a couple of large simulation results folders).

## Package setup

At the end of setup, you should have the following directory structure (starred folders not included in the repo):

       your-repo-name/
       ├─ LICENSE
       ├─ README.md
       ├─ environment.yml
       ├─ lateral_signaling/
    *  │  ├─ sacred/
       │  ├─ example.ipynb
       │  ├─ lateral_signaling.py
       │  ├─ ...
    ** ├─ env/
    ** ├─ data/
       │  ├─ analysis/
       │  ├─ simulations/
       │  ├─ FACS/
       │  ├─ .../
    *  ├─ plots/
    *  │  ├─ tmp/

The `data` folder should be unzipped from the supplementary data ([see above](#Supplementary-data)). It can be unzipped directly into your cloned repository as shown above, or you can store it somewhere else in your filesystem and make `repo-name/data` a symbolic link to the data. The `env` folder will be created in the next step and holds the environment for running the codebase. The folders marked with a single `*` will be created as new empty folders, either by the user or when the main module is imported for the first time. The `sacred` folder will hold any simulations that are run (simulation scripts can also be edited to write to a different folder). Plotting outputs will be saved to `plots` or `plots/tmp`. 

### 1. Clone the repo

Clone the repo onto your local machine.

    git clone https://github.com/pranav-bhamidipati/lateral-signaling.git
    
Or `git clone git@github.com:pranav-bhamidipati/lateral-signaling.git` if you have `ssh` credentials.
    
### 2. Build the environment

The package uses `conda` for environment management,. You can download a minimal version for your machine [here](https://docs.conda.io/en/latest/miniconda.html). Once installed, navigate to the repo directory on the command line and create the environment.

    conda env create --file environment.yml --prefix ./env

It may take a long time to solve the environment before eventually installing (5-30 minutes). This will install the environment locally, within the repo directory, but `--name` can be used instead of `--prefix` to store it in `conda`'s default location. 

Once all the packages are installed, activate the environment.

    conda deactivate
    conda activate ./env

To use the environment in a Jupyter notebook (such as `example.ipynb`) you will also need to create an IPython kernel. With the environment activated, run

    python -m ipykernel install --name lateral-signaling --prefix ./env

Then, you can access the interactive notebook by running `jupyter notebook` or `jupyter lab`, opening the browser link, and selecting the `lateral-signaling` kernel. 

### 3. Download supplementary data

Download the [supplementary data](https://doi.org/10.22002/q8n10-tsk03) for this project and extract into a folder named `data`.

### 4. (Optional) Folders for results output

Create empty folders `plots`, `plots/tmp`, and `lateral_signaling/sacred`, which will be used to save plotting and simulation results. They should be created by default the first time `lateral_signaling.py` is imported.

At this point, the directory tree should look like [above](#Package-setup). The package relies on this folder structure by default to properly load the contents. 

## Using the package

The main contents of the project are contained in the `lateral_signaling` directory. 

### The main Python module (`lateral_signaling.py`)

The file `lateral_signaling/lateral_signaling.py` is the main module custom code for data analysis, plotting, and simulation that is accessed by the other files in the folder. It is often abbreviated `lsig`. There are also accessory modules, preceded by underscores.

### Data analysis

All analysis scripts (`analyze_XXX.py`) read in data from the `data` folder, perform analysis, and save the result to `./data/analysis` by default. This can be set optionally by changing the value of `lsig.analysis_dir`.

### Plotting

Scripts for plotting (`plot_XXX.py`) save to `./plot` or `./plot/tmp`. These can also be changed via `lsig.plot_dir` and `lsig.temp_plot_dir`.

### Simulation

The mathematical model for synNotch signaling was simulated using the `sacred` package for experimental data provenance, which keeps a record of all the metadata (source code, system info, parameter settings, etc) used to produce a simulated dataset. See [example.ipynb](lateral_signaling/example.ipynb).

Briefly, a simulation is run by changing to the `./lateral_signaling` directory and running a script such as `simulate_XXX_run_many.py`. This executes the simulation under multiple parameter configurations and saves the result to the output directory specified in the corresponding `run_one.py` file (defaults to `lateral_signaling/sacred`). The simulation itself is contained in the `XXX_simulation_logic.py` file.
