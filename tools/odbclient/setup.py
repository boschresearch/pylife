# Copyright (c) 2021-2026 - for information on the respective copyright owner
# see the NOTICE file and/or the repository
# https://github.com/boschresearch/pylife
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Setup file for odbclient.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import os
import sys
from setuptools import setup

if __name__ == "__main__":
    if sys.version_info[0] < 3:
        sys.exit("Python 3 environment is required.")
    scm_version_setup = {
        "root": "../..",
        "relative_to": __file__,
        "version_scheme": "no-guess-dev",
    }
    if os.environ.get("CI") == "true":
        scm_version_setup.update({"local_scheme": "no-local-version"})
    try:
        setup(use_scm_version=scm_version_setup, python_requires=">=3")
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
