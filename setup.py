"""
Setup file for pylife.
Use setup.cfg to configure your project.


This file was generated with PyScaffold 4.0.1.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""

import platform
import re
from distutils.core import Extension
from subprocess import CalledProcessError

import numpy
from setuptools import setup

try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

# Determine compiler-specific optimization flags
if platform.system() == "Windows" and "GCC" not in platform.python_compiler():
    # For Windows with MSVC compiler - maximum optimization
    extra_compile_args = ["/Ox"]
else:
    # For other platforms or GCC
    extra_compile_args = ["-O3"]

# First extension: rainflow
rainflow_ext = Extension(
    name="pylife.rainflow_ext",
    sources=["src/pylife/stress/rainflow/extension.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

# Second extension: FKM linear functions
fkm_extension = Extension(
    name="pylife._fkm_linear_functions",
    sources=["src/pylife/strength/fkm_linear/extension.pyx.in"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

VECTORIZE = "\n# vectorize"
MATCH_STRING_FUNCTION = r"def +(\w*)\s*\(([\w,\s+]*)\):"
function_matcher = re.compile(MATCH_STRING_FUNCTION)

TEMPLATE = """
def {{ function_name_pure }}(*args):
    if all(isinstance(p, float) for p in args):
        return _{{ function_name_pure }}(*args)
    return {{ function_name }}_array(*[np.array(np.asarray(p), copy=True) for p in args])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def {{ function_name }}_array({{ parameter_list_array }}):
    cdef size_t i
    cdef size_t length = len({{ first_parameter }})

    result = np.empty(length, dtype=np.float64)

    for i in range(length):
        result[i] = {{ function_name }}({{ parameter_list_indexed }})

    return result

"""


def vectorize(in_code):
    """Process Cython code and add vectorized versions of marked functions."""

    def split_type(param):
        sp = param.split(" ")
        return (sp[0].strip(), sp[1].strip())

    out_code = "cimport cython\nimport numpy as np\nfrom cython.cimports.libc.math import exp, log, log10, sqrt, isnan\n"

    while True:
        pos = in_code.find(VECTORIZE)
        out_code += in_code[:pos]

        if pos == -1:
            break

        in_code = in_code[pos + len(VECTORIZE) + 1 :]
        hit = function_matcher.match(in_code)
        assert hit, "vectorize hint without matching function declaration"

        function_name = hit.group(1)
        function_name_pure = function_name[1:]

        parameter_string = hit.group(2)
        parameters = [split_type(p.strip()) for p in parameter_string.split(",")]

        first_parameter = parameters[0][1]
        parameter_list_array = ", ".join(f"{p[0]}[::1] {p[1]}" for p in parameters)
        parameter_list_indexed = ", ".join(f"{p[1]}[i]" for p in parameters)

        vectorized = (
            TEMPLATE.replace("{{ function_name }}", function_name)
            .replace("{{ function_name_pure }}", function_name_pure)
            .replace("{{ parameter_list_array }}", parameter_list_array)
            .replace("{{ first_parameter }}", first_parameter)
            .replace("{{ parameter_list_indexed }}", parameter_list_indexed)
        )

        out_code += f"{vectorized}\n"

    return out_code


def apply_template_to_source(source):
    """Convert .pyx.in template files to .pyx files."""
    if not source.endswith(".in"):
        return source

    target = source[:-3]
    print(f"Generating {target} from {source}")

    with open(source) as in_file:
        out_code = vectorize(in_file.read())

    with open(target, "w") as out_file:
        out_file.write(out_code)

    return target


if __name__ == "__main__":
    # Process template sources
    fkm_extension.sources = [
        apply_template_to_source(src) for src in fkm_extension.sources
    ]

    # Collect all extensions
    ext_modules = [rainflow_ext, fkm_extension]

    # Cythonize if available
    if HAS_CYTHON:
        ext_modules = cythonize(
            ext_modules, compiler_directives={"language_level": "3"}
        )

    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"}, ext_modules=ext_modules
        )
    except CalledProcessError:
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm, wheel, and Cython with:\n"
            "   pip install -U setuptools setuptools_scm wheel Cython\n\n"
        )
        raise
