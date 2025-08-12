"""
    Setup file for pylife.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import os
from setuptools import setup
from distutils.core import Extension
import numpy

ext = Extension(
    name='pylife.rainflow_ext',
    sources=['src/pylife/stress/rainflow/extension.pyx'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"]
)

if __name__ == "__main__":
    scm_version_setup = {"version_scheme": "no-guess-dev"}
    if os.environ.get("CI") == "true":
        scm_version_setup |= {"local_scheme": "no-local-version"}
    try:
        setup(
            use_scm_version=scm_version_setup,
            ext_modules=[ext]
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
