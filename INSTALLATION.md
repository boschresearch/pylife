# Installation

## Installation to use pyLife

### Prerequisites

You need a python installation with a recent (brand new ones might not work)
python version installed. The recommended way to manage that is
[uv](https://docs.astral.sh/uv/), a fast python package and project manager.

Although you can install and use pyLife with Python version >= 3.9 and pandas
version >= 2.2, it is strongly recommended to use at least Python 3.11 and
pandas 3.0.0.


#### Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
following the instructions for your operating system. uv can also install
and manage the python versions for you, so a separate python installation is
not required.


### Install the pyLife package

The simplest way to install pyLife is to add it to a uv-managed project
```
uv add "pylife[all]"
```
That installs pyLife with all the dependencies to use pyLife in python
programs. You might want to install some further packages like `jupyter` in
order to work with jupyter notebooks.
```
uv add "pylife[all,extras]"
```
might be a good start.

If you prefer a plain virtual environment instead of a uv-managed project,
you can create and populate one with
```
uv venv
uv pip install "pylife[all]"
```
and activate it as usual.


## Installation to develop pyLife

For general contribution guidelines please read [CONTRIBUTING.md](CONTRIBUTING.md)


### Prerequisites

As pyLife now is using Cython extensions for performance reasons. Therefore you
will need a C-compiler available on your system. On Linux systems, there
usually is a `gcc` compiler available. On Windows you will need a current
version of Microsoft Visual C++.


### Install uv

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
