# Code for ``Unifying pairwise interactions in complex dynamics''

This repository illustrates how the figures in the paper, ``Unifying pairwise interactions in complex dynamics'', were created.

We provide both precomputed CSV files, with which to recreate the figures, as well as scripts to generate these CSVs from scratch.

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
If you would like to replicate the results from the paper as closely as possible, I would recommend checking out the `pynats` branch, which is [also available as a release](https://github.com/olivercliff/pyspi/releases/tag/pynats-v0.1).
However, given this work was not computed via a container, such as docker, the results may vary slightly from those originally reported.

In the `pyspi` folder and environment that were created above, checkout and install the `pynats` branch (this will take a while):
```
git checkout pynats
pip install .
```

# Generating the figures

This repository regenerates Figs. 2 and 3 of the main paper, and a number of supporting figures from the appendix.
Because of the large amount of processing required to generate these figures from scratch, we have provided pre-computed CSV files (in the `data` directory) from which you can easily regenerate the figures using the [`generate_figures.ipynb`](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/generate_figures.ipynb) notebook.

# Re-generating results for figures

In order to re-compute the CSV files from the raw MTS data, we have provided three scripts:
- [`process_mts_database.py`](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/process_mts_database.py), which computes all SPIs for each of the 1053 MTS datasets in the database;
- [`process_uea_data.py`](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/process_uea_data.py), which computes all SPIs for each of the 40 MTS datasets in the [UEA `BasicMotions` dataset](http://www.timeseriesclassification.com/description.php?Dataset=BasicMotions) for the classification study;
- [`classify.py`](https://github.com/olivercliff/nat-comp-sci-paper/blob/main/classify.py), which uses the precomputed SPIs for the UEA dataset (i.e., the output of `process_uea_data.py`) for classifying the time series.

Once each of these scripts have been executed, re-run the `generate_figure.ipynb` notebook, pointing to the location of the new CSV files to regenerate the figures.

> **NOTE:** The script `process_mts_database.py` was computed on a cluster, where each dataset was evaluated separately. If running this script locally, it will likely take several months to complete (and so I would advise using the [pyspi distribute](https://github.com/olivercliff/pyspi-distribute) script to recompute all SPIs on all 1053 datasets, if necessary).
