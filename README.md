# hetmech: identifying mechanisms using hetnets

## Environment

This repository uses [conda](http://conda.pydata.org/docs/ "Conda package management system and environment management system documentation") to manage its environment and install packages. If you don't have conda installed on your system, you can [download it here](http://conda.pydata.org/miniconda.html "Miniconda Homepage"). You can install the Python 2 or 3 version of Miniconda (or Anaconda), which determines the Python version of your root environment. Since we create a dedicated environment for this project, named `hetmech` whose explicit dependencies are specified in [`environment.yml`](environment.yml), the version of your root environment will not be relevant.

With conda, you can create the `hetmech` environment using:

```sh
# Create or overwrite the cognoma-machine-learning conda environment
conda env create --file environment.yml
```

Activate the environment by running `source activate hetmech` on Linux or OS X and `activate hetmech` on Windows. Once this environment is active in a terminal, run `jupyter notebook` to start a notebook server.

## hetnet codes

* `path_charting.py` -- Identifies gene-compound pairs connected by a number of paths (of length <= 3) inside a certain specified range; outputs a TSV of node-pairs and a TXT of all paths between such node pairs. For now, the range of numbers of paths is hard-coded (between 10 and 100 paths of length 2; between 20 and 100 paths of length 3), but easily changed. These parameter choices were selected to produce a small number of node pairs (only one node pair for these parameter settings!).
