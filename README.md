# hetmatpy: a Python 3 package for matrix operations of hetnets

[![Documentation](https://img.shields.io/badge/-Documentation-purple?logo=read-the-docs&style=for-the-badge)](https://hetio.github.io/hetmatpy/)
[![PyPI](https://img.shields.io/pypi/v/hetmatpy.svg?logo=PyPI&style=for-the-badge)](https://pypi.org/project/hetmatpy/)
[![GitHub Actions CI Tests Status](https://img.shields.io/github/workflow/status/hetio/hetmatpy/Tests?label=actions&logo=github&style=for-the-badge)](https://github.com/hetio/hetmatpy/actions)
<!--
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=Python)](https://github.com/psf/black)
-->

This codebase enables identifying the relevant network connections between a set of query nodes.
The method is designed to operate on hetnets (networks with multiple node or relationship types).

This project is still under development.
Use with caution.

## Environment

Install via pip from GitHub using:

```shell
# install the latest release from PyPI
pip install hetmatpy

# install latest version on GitHub
pip install git+https://github.com/hetio/hetmatpy

# for local development, run the following inside the development environment:
pip install --editable .
```

## Development

This repo uses pre-commit checks:


```shell
# run once per local repo before committing
pre-commit install
```

## Acknowledgments

This work is supported through a research collaboration with [Pfizer Worldwide Research and Development](https://www.pfizer.com/partners/research-and-development).
This work is funded in part by the Gordon and Betty Moore Foundation’s Data-Driven Discovery Initiative through Grants [GBMF4552](https://www.moore.org/grant-detail?grantId=GBMF4552) to Casey Greene and [GBMF4560](https://www.moore.org/grant-detail?grantId=GBMF4560) to Blair Sullivan.
