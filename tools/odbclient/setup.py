"""
    Setup file for odbclient.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

import sys
from setuptools import setup

if __name__ == "__main__":
    if sys.version_info[0] < 3:
        sys.exit("Python 3 environment is required.")
    try:
        setup(
            use_scm_version = {
                "root": "../..",
                "relative_to": __file__,
                "version_scheme": "no-guess-dev"
            },
            python_requires = ">=3"
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
