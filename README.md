# Code for ``Unifying pairwise interactions in complex dynamics''

This repository illustrates how the figures in the paper, ``Unifying pairwise interactions in complex dynamics'' were created.

It contains both precomputed databases with which to recreate the figures, as well as scripts to generate these databases from scratch.

# Download and install pyspi

First, download `pyspi` and create a conda environment to install the package [as per the documentation](https://pyspi-toolkit.readthedocs.io/en/latest/).
In linux, this involves the following steps from a terminal (in your desired directory):
```
git clone git@github.com:olivercliff/pyspi.git
cd pyspi
conda create -n pyspi python=3.9
conda activate pyspi
```

You will also likely need to download and install `octave`; [follow the instructions here](https://octave.org/download). 

## Switch to the `pynats` branch

The `pynats` branch was retroactively added to the `pyspi` repository as it is older code that was used to generate the results from the main paper.
If you would like to exactly replicate the results from the paper, I would recommend checking out the `pynats` branch, which is [also available as a release](https://github.com/olivercliff/pyspi/releases/tag/pynats-v0.1).
In the `pyspi` folder and environment that were created above, checkout and install the `pynats` branch (this will take a while):
```
git checkout pynats
pip install .
```

# Generating the figures

The main figures from the paper that are generated from the code in this repository are Figs. 2 and 3.
Because of the large amount of processing required to generate these figures, we have provided pre-computed databases from which you can easily regenerate the figures in [generate_figures.ipynb](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/generate_figures.ipynb)
