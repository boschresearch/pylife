# Installation / Getting started

## Just a glimpse

If you just want to check out pyLife's demos, you can use the our notebooks at
[mybinder](https://mybinder.org/v2/gh/boschresearch/pylife/master?filepath=demos%2Findex.ipynb). We
will add new notebooks as soon as we have new functionality.


## Installation to use pyLife

### Prerequisites

You need a python installation e.g. a virtual environment with `pip` a recent
(brand new ones might not work) python versions installed. There are several
ways to achieve that.


#### Using miniconda or anaconda

Install [miniconda](https://conda.io/miniconda.html) or
[anaconda](http://anaconda.com) on your computer and create a virtual
environment with python installed. See the [conda
documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
on how to do that. The newly created environment must be activated.

The following command lines should do it
```
conda create -n pylife-env python=3.12 --yes
conda activate pylife-env
```

#### Using virtualenv

Setup a python virtual environment containing pip according to [these
instructions](https://docs.python.org/3/tutorial/venv.html) and activate it.


#### Using the python installation of your Linux distribution

That's not recommended. If you really want to do that, you probably know how to
do it.


### Install the pyLife package

The simplest way to install pyLife is just using the pip package
```
pip install pylife[all]
```
That installs pyLife with all the dependencies to use pyLife in python
programs. You might want to install some further packages like `jupyter` in
order to work with jupyter notebooks.

There is no conda package as of now, unfortunately.


## Installation to develop pyLife

For general contribution guidelines please read [CONTRIBUTING.md](CONTRIBUTING.md)

#### Install uv

* Install the [uv](https://docs.astral.sh/uv/) python management tool.


### Clone the git repository

Depending on your tools. From the command line
```
git clone https://github.com/boschresearch/pylife.git
```
will do it.

### Install the dependencies

Install pylife and all the dependencies into the environment using

```
uv sync --dev --all-extras
```

### Test the installation

You can run the test suite by the command
```
uv run pytest
```

If it creates an output ending like below, the installation was successful.
```
================ 1171 passed, 7 skipped, 6 deselected, 3 warnings in 23.06s ===============
```

There might be some `DeprecationWarning`s. Ignore them for now.
