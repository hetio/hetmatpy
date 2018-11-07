# hetmat: a Python 3 package for matrix operations of hetnets

[![Build Status](https://travis-ci.org/hetio/hetmat.svg?branch=master)](https://travis-ci.org/hetio/hetmat)

Hetmech aims to identify the relevant network connections between a set of query nodes.
The method is designed to operate on hetnets (networks with multiple node or relationship types).

This project is still under development. Use with caution.

## Environment

This repository uses [conda](http://conda.pydata.org/docs/) to manage its environment as specified in [`environment.yml`](environment.yml).
Install the environment with:

```sh
conda env create --file=environment.yml
```

Then use `conda activate hetmech` and `conda deactivate` to activate or deactivate the environment.

For local development, run the following with the hetmech environment activated:

`pip install --editable .`

## Acknowledgments

This work is supported through a research collaboration with [Pfizer Worldwide Research and Development](https://www.pfizer.com/partners/research-and-development).
This work is funded in part by the Gordon and Betty Moore Foundation’s Data-Driven Discovery Initiative through Grants [GBMF4552](https://www.moore.org/grant-detail?grantId=GBMF4552) to Casey Greene and [GBMF4560](https://www.moore.org/grant-detail?grantId=GBMF4560) to Blair Sullivan.
