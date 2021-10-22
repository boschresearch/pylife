# pylife-odbserver

A server for odbAccess to be accessed by pylife-odbclient


## Purpose

Unfortunately Abaqus still comes with a python-2.x engine. So you can't access
an Abaqus odb file from within modern python code. This python package is the
client part of a client server setup to make odb files accessible from within
python-3.x code in a transparent way.


## Solution

This package provides a slim server that as python-2.7 software, that can be
run inside the Abaqus python engine. It accepts command via `sys.stdin` and
according to the command is querying data from the `odbAccess` interface and
returning them in a pickle object.

The sibling package `pylife-odbclient` comes with a python class `OdbClient`
that spawns the server in the background when an instance of `OdbClient` is
instantiated. Then the client object can be used to transparently access data
from the odb file via the server. Once the client object goes out of scope
i.e. is deleted, the server process is stopped automatically.


## Installation

As of now there is no released version of `pylife-odbserver`, same for
`pylife-odbclient`. Therefore there are some manual steps required.

* Clone the pyLife repository.

* Change to the directory `tools/odbserver`

* Create and activate a plain python-2.7 environment without additional
  packages.

* Run `pip install -e .`

* See the [instructions in `pylife-odbclient`](../odbclient/README.md) on how
  to install the client.
