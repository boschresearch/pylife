"""Expose version metadata for the Abaqus-side ODB server package.

Users normally do not import this package directly.  ``pylife-odbclient``
starts it inside Abaqus so that ODB data can be queried from a regular Python
process.
"""

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("pylife-odbserver").version
except:
    __version__ = 'unknown'
