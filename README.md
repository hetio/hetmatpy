# hetmech: identifying mechanisms using hetnets

## Environment

This repository uses [conda](http://conda.pydata.org/docs/ "Conda package management system and environment management system documentation") to manage its environment and install packages. If you don't have conda installed on your system, you can [download it here](http://conda.pydata.org/miniconda.html "Miniconda Homepage"). You can install the Python 2 or 3 version of Miniconda (or Anaconda), which determines the Python version of your root environment. Since we create a dedicated environment for this project, named `hetmech` whose explicit dependencies are specified in [`environment.yml`](environment.yml), the version of your root environment will not be relevant.

With conda, you can create the `hetmech` environment using:

```sh
# Create or overwrite the cognoma-machine-learning conda environment
conda env create --file environment.yml
```

Activate the environment by running `source activate hetmech` on Linux or OS X and `activate hetmech` on Windows. Once this environment is active in a terminal, run `jupyter notebook` to start a notebook server.
