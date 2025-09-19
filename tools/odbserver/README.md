# pylife-odbserver

A server for odbAccess to be accessed by pylife-odbclient


## Purpose

Unfortunately Abaqus usually comes with an outdated python engine. So you can't
access an Abaqus odb file from within modern python code using the latest
packages. This python package is the client part of a client server setup to
make odb files accessible from within python code using a current python
version in a transparent way.


## Solution

This package provides a slim server that as python software that is running
with old python versions of Abaqus (even 2.7), that can be run inside the
Abaqus python engine. It accepts command via `sys.stdin` and according to the
command is querying data from the `odbAccess` interface and returning them in a
pickle object.

The sibling package `pylife-odbclient` comes with a python class `OdbClient`
that spawns the server in the background when an instance of `OdbClient` is
instantiated. Then the client object can be used to transparently access data
from the odb file via the server. Once the client object goes out of scope
i.e. is deleted, the server process is stopped automatically.


## Installation

* Create and activate a plain python-2.7 environment without additional
  packages.  For example by
```
conda create -n odbserver python=2.7 pip
```

Instead of `2.7` you must choose the python version of your abaqus version. You
can find it out using

```
abaqus python --version
```

* Run
```
pip install pylife-odbserver
```

* See the <a href="../odbclient/">instructions in `pylife-odbclient`</a> on how
  to install the client.
