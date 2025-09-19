# pylife-odbclient

A Modern Python client for odbAccess using pylife-odbserver


## Purpose

Unfortunately Abaqus usually comes with an outdated python engine. So you can't
access an Abaqus odb file from within modern python code using the latest
packages. This python package is the client part of a client server setup to
make odb files accessible from within python code using a current python
version in a transparent way.



## Solution

The sibling package `pylife-odbserver` provides a slim server that as python
software that is running with old python versions of Abaqus (even 2.7), that
can be run inside the Abaqus python engine. It accepts command via `sys.stdin`
and according to the command is querying data from the `odbAccess` interface
and returning them in a pickle object.

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

* See the <a href="../odbserver/">instructions in `pylife-odbserver`</a> on how
  to install the server.


## Usage

Usually you only will see the `OdbClient` class interface when you access an
odb file. The only point you care about the server is when you instantiate an
`OdbClient` object. You need to know the following things

* The path to the Abaqus executable

* The path to the python environment `pylife-server` is installed into.

Then you can instantiate a `OdbClient` object using

```python
import odbclient as CL

client = CL.OdbClient("yourodb.odb")
```

See the [API docs of `OdbClient`][1]
for details.


## Limitations

### Limited functionality

Only a subset of Abaqus variable locations are supported. These are: nodal,
element nodal, whole element and centroid. Integration point variables are
extrapolated to element nodal.

You can only extract data from an odb file, not write to it.

### String literals

So far only names made of `ascii` strings are supported.  That means that
instance names, node that names and the like containing non-ascii characters
like German umlauts will not work.

___
[1]: https://pylife.readthedocs.io/en/latest/tools/odbclient/odbclient.html
