# Installation of pyLife 

For now we only have the python code repos, no pip or conda packages yet, sorry.

## Download

Clone the git repository and enter it with a unix shell, on windows e.g. `git-bash`.

## Perquisites

pyLife being a Python library needs a running Python installation. As we make
heavy use of the numpy/scipy/pandas stack and also some other useful libraries
for numeric and data analysis, we recommend a python package manager like conda
or pip to set up the python environment.

### Using anaconda

Create an anaconda environment with all the requirements by running
```
./create_environment.sh
```
which will take a couple of minutes, sorry.

Then activate it:
```
conda activate _venv
```

### Using pip

tbw.


## Installing

### For users of pylife

If you want to use pylife for your project you can install it via pip into your
active python environment by.
```
pip install .
```

### If you want to develop pylife itself

If you intent to develop for pylife you should to
```
python ./setup.py develop
```

## Test the installation

You can run the test suite by the command
```
pytest
```

If it creates an output like below, the installation was successful.
```
============================= test session starts ==============================
platform linux -- Python 3.7.4, pytest-5.2.1, py-1.8.0, pluggy-0.13.0
rootdir: /home/jmu3si/Devel/pylife, inifile: pytest.ini, testpaths: tests
collected 35 items

tests/strength/test_meanstress.py ....                                   [ 11%]
tests/stress/test_equistress.py .                                        [ 14%]
tests/stress/test_rainflow.py .................                          [ 62%]
tests/utils/test_failure_probability.py ...                              [ 71%]
tests/utils/test_meshapping.py .....                                     [ 85%]
tests/utils/test_meshops.py .....                                        [100%]

============================== 35 passed in 7.29s ==============================
```

There might be some `DeprecationWarning`s. Ignore them for now.


# Building the documentation

As long as the docs are not available online, you have to build them yourself
using the command
```
./build_docs.sh
```
The docs then are in `doc/build/index.html`.
