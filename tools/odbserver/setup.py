#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for odbserver.

    This file was generated with PyScaffold 2.5, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import os
import sys
from setuptools import setup


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    scm_version_setup = {
        "root": "../..",
        "relative_to": __file__,
    }
    if os.environ.get("CI") == "true":
        scm_version_setup.update({"local_scheme": "no-local-version"})
    setup(
        use_scm_version=scm_version_setup,
        setup_requires=['six', 'setuptools_scm'] + sphinx
    )


if __name__ == "__main__":
    setup_package()
