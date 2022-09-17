# Code for ``Unifying pairwise interactions in complex dynamics''

This repository illustrates how the figures in the paper, ``Unifying pairwise interactions in complex dynamics'' were created.

It contains both precomputed databases with which to recreate the figures, as well as scripts to generate these databases from scratch.

# Download pyspi and create an environment

First, download `pyspi` and create a conda environment to install the package [as per the documentation](https://pyspi-toolkit.readthedocs.io/en/latest/).
In linux, this involves the following steps from a terminal (in your desired directory):
```
git clone git@github.com:olivercliff/pyspi.git
cd pyspi
conda create -n pyspi python=3.6.7
conda activate pyspi
```

> **NOTE:** Python version 3.6.7 should only be used for the legacy `pynats` branch. If using the latest `pyspi`, please create a conda environment with `python=3.9`. 

You will also likely need to download and install `octave`; [follow the instructions here](https://octave.org/download). 

## Switch to the `pynats` branch and install

The `pynats` branch was retroactively added to the `pyspi` repository as legacy code that was used to generate the results from the main paper.
If you would like to exactly replicate the results from the paper, I would recommend checking out the `pynats` branch, which is [also available as a release](https://github.com/olivercliff/pyspi/releases/tag/pynats-v0.1).

In the `pyspi` folder and environment that were created above, checkout and install the `pynats` branch (this will take a while):
```
git checkout pynats
pip install .
```

# Generating the figures

This repository regenerates Figs. 2 and 3 of the main paper, and a number of supporting figures from the appendix.
Because of the large amount of processing required to generate these figures, we have provided pre-computed databases from which you can easily regenerate the figures using [`generate_figures.ipynb`](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/generate_figures.ipynb).