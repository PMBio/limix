# Limix

[![PyPIl](https://img.shields.io/pypi/l/limix.svg?style=flat-square)](https://pypi.python.org/pypi/limix/) [![PyPIv](https://img.shields.io/pypi/v/limix.svg?style=flat-square)](https://pypi.python.org/pypi/limix/) [![Anaconda](https://anaconda.org/conda-forge/limix/badges/version.svg)](https://anaconda.org/conda-forge/limix) [![Travis](https://img.shields.io/travis/PMBio/limix.svg?style=flat-square)](https://travis-ci.org/PMBio/limix)


Limix is a flexible and efficient linear mixed model library with interfaces
to Python. Genomic analyses require flexible models that can be adapted to the needs of
the user. Limix is smart about how particular models are fitted to save
computational cost.

LATEST: [iSet, interaction set tests for gene-context interactions, is now in Limix.](https://github.com/PMBio/limix-tutorials/tree/master/iSet)

## Installation

### Using Conda package manager

Conda is a package manager designed for Python and R users/developers of
scientific tools, and comes with the [Anaconda distribution](https://www.continuum.io/downloads).
Currently we support this installation for Linux 64 bits and OSX operating
systems.

```
conda install -c conda-forge limix
```

### Using Pip

If you don't have Conda (or don't want to use the above method), Limix can be
installed via Pip package manager.
```
pip install limix
```
This approach is not as straightforward as the first one because it requires
compilation of C/C++ and (potentially) Fortran code, and some understanding
of dependency resolution is likely to be required. We provide bellow recipes
for some popular Limix distributions, assuming you have the `wget` command line
tool.

- Ubuntu

    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/apt_limix_install)
    ```

- Fedora
    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/dnf_limix_install)
    ```

- OpenSUSE
    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/zypper_limix_install)
    ```

### From source

This is more tricky in terms of dependency resolution but useful for developers.

```
git clone https://github.com/PMBio/limix.git
cd limix
python setup.py install # or python setup.py develop
```

## Usage

A good starting point is our package Vignettes. These tutorials are available from this repository: https://github.com/PMBio/limix-tutorials.

The main package vignette can also be viewed using the ipython notebook viewer:
http://nbviewer.ipython.org/github/pmbio/limix-tutorials/blob/master/index.ipynb.

Alternatively, the source file is available in the separate Limix tutorial repository:
https://github.com/PMBio/limix-tutorials

## Problems

If you want to use Limix and encounter any issues, please contact us via `limix@mixed-models.org`.

## Authors

- `Franceso Paolo Casale` (`casale@ebi.ac.uk`)
- `Danilo Horta` (`horta@ebi.ac.uk`)
- `Christoph Lippert` (`christoph.a.lippert@gmail.com`)
- `Oliver Stegle` (`stegle@ebi.ac.uk`)

## License

See [Apache License (Version 2.0, January 2004)](https://github.com/PMBio/limix/blob/master/LICENSE).
