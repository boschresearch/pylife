# pylife-odbclient

A Python 3 client for odbAccess using pylife-odbserver


## Purpose

Unfortunately Abaqus still comes with a python-2.x engine. So you can't access
an Abaqus odb file from within modern python code. This python package is the
client part of a client server setup to make odb files accessible from within
python-3.x code in a transparent way.


## Solution

The sibling package `pylife-odbserver` provides a slim server that as
python-2.7 software, that can be run inside the Abaqus python engine. It
accepts command via `sys.stdin` and according to the command is querying data
from the `odbAccess` interface and returning them in a pickle object.

This package comes with a python class `OdbClient` that spawns the server in
the background when an instance of `OdbClient` is instantiated. Then the client
object can be used to transparently access data from the odb file via the
server. Once the client object goes out of scope i.e. is deleted, the server
process is stopped automatically.


## Installation

* Install the odbclient using `pip` with the command
```
pip install pylife-odbclient
```

* See the [instructions in `pylife-odbserver`](../odbserver/README.md) on how
  to install the server.


Once there are released versions the installation will be easier.

* Install the server using `pip install pylife-odbserver` in a python-2.0
  environment that is usable from the current Abaqus python engine.

* Install the client package using `pip install pylife-odbclient`.


## Usage

Usually you only will see the `OdbClient` class interface when you access an
odb file. The only point you care about the server is when you instantiate an
`OdbClient` object. You need to know the following things

* The path to the Abaqus executable

* The path to the python environment `pylife-server` is installed into.

Then you can instantiate a `OdbClient` object using

```python
import odbclient as CL

client = CL.OdbClient("<path-to-abaqus>/abaqus", "<path-to-env>", "yourodb.odb")
```

See the API doc of `OdbClient` for details. (At the moment only in the sources,
sorry.)


## Limitations

Only a subset of Abaqus variable locations are supported. These are: nodal,
element nodal, whole element and centroid. Integration point variables are
extrapolated to element nodal.

You can only extract data from an odb file, not write to it.
