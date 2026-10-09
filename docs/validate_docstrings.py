# Copyright (c) 2019-2026 - for information on the respective copyright owner
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

"""Check pyLife docstrings against the numpy/scipy docstring conventions.

This script walks the public API of :mod:`pylife` and runs
:func:`numpydoc.validate.validate` on every public module, class, method and
function.  It is deliberately kept out of ``conf.py``: ``numpydoc`` and
``sphinx.ext.napoleon`` both rewrite docstrings and must not be active in the
same Sphinx build, and the documentation CI builds with ``-W`` where a style
warning would abort the build.

Run it directly::

    python docs/validate_docstrings.py

Compare against a recorded baseline so that the situation can only improve::

    python docs/validate_docstrings.py --baseline docs/docstring_baseline.txt

Require that every public object is documented::

    python docs/validate_docstrings.py --min-coverage 100

Exit codes
----------
0
    No violations, or no more violations than the baseline allows.
1
    Violations were found that are not covered by the baseline, or the
    documentation coverage dropped below ``--min-coverage``.
"""

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
from pathlib import Path

try:
    from numpydoc.validate import validate
except ImportError:  # pragma: no cover - depends on the docs dependency group
    sys.exit("numpydoc is not installed. Install the 'docs' dependency group.")


#: Checks that are switched off for the whole code base.
#:
#: ``ES01``/``EX01``/``SA01`` demand an extended summary, an examples section
#: and a see-also section on *every* object, which is stricter than what numpy
#: and scipy enforce themselves.  ``GL08`` (missing docstring) is enforced.
DEFAULT_IGNORED_CHECKS = frozenset({"ES01", "EX01", "SA01"})

#: Objects whose docstrings are not authored by pyLife (generated or vendored).
IGNORED_PREFIXES = ("pylife.vmap.VMAPExport.__", "pylife.vmap.VMAPImport.__")

#: Modules that must not be imported by the documentation tooling.
#:
#: ``pylife.materialdata.woehler.bayesian`` deliberately raises on import
#: because the Bayesian Wöhler analyzer has been shut down.
IGNORED_MODULES = ("pylife.materialdata.woehler.bayesian",)

#: Matches the ``{'b', 'a'}`` set reprs that numpydoc embeds into some messages.
_SET_REPR = re.compile(r"\{[^{}]*\}")


def _normalize(description):
    """Make a numpydoc message reproducible across runs.

    Several numpydoc checks (``PR01``, ``PR02``, ...) interpolate a Python
    ``set`` into their message.  The iteration order of a ``set`` of strings
    varies between interpreter runs, so the raw message cannot be compared
    against a recorded baseline.  This function rewrites every embedded set
    repr with its elements in sorted order.

    Parameters
    ----------
    description : str
        The message as produced by :func:`numpydoc.validate.validate`.

    Returns
    -------
    normalized : str
        The message with all embedded set reprs sorted.

    Examples
    --------
    >>> _normalize("Parameters {'b', 'a'} not documented")
    "Parameters {'a', 'b'} not documented"
    """

    def sort_set(match):
        items = sorted(item.strip() for item in match.group(0)[1:-1].split(","))
        return "{" + ", ".join(items) + "}"

    return _SET_REPR.sub(sort_set, description)


def _iter_modules(package):
    """Yield the fully qualified names of ``package`` and all its subpackages.

    Parameters
    ----------
    package : module
        The already imported top level package to walk.

    Yields
    ------
    name : str
        Importable module name, e.g. ``'pylife.stress.rainflow'``.
    """
    yield package.__name__
    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if name in IGNORED_MODULES:
            continue
        yield name


def _iter_objects(module_name):
    """Yield the numpydoc object names to validate for one module.

    Parameters
    ----------
    module_name : str
        Importable module name.

    Yields
    ------
    name : str
        Object name in the dotted notation understood by
        :func:`numpydoc.validate.validate`.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import side effects
        print(f"skipping {module_name}: {exc}", file=sys.stderr)
        return

    yield module_name

    for obj_name, obj in vars(module).items():
        if obj_name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != module_name:
            continue
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue

        yield f"{module_name}.{obj_name}"

        if not inspect.isclass(obj):
            continue

        for member_name, member in vars(obj).items():
            if member_name.startswith("_"):
                continue
            if not (inspect.isfunction(member) or isinstance(member, property)):
                continue
            yield f"{module_name}.{obj_name}.{member_name}"


def collect_violations(ignored_checks):
    """Validate the whole public pyLife API.

    Parameters
    ----------
    ignored_checks : set of str
        numpydoc check codes that are not reported, e.g. ``{'EX01'}``.

    Returns
    -------
    violations : list of str
        One ``'<object>: <code> <description>'`` line per violation, sorted.
    documented : int
        Number of validated objects that have a docstring.
    total : int
        Number of validated objects.
    """
    import pylife

    violations = []
    documented = 0
    total = 0

    for module_name in _iter_modules(pylife):
        for obj_name in _iter_objects(module_name):
            if obj_name.startswith(IGNORED_PREFIXES):
                continue
            try:
                result = validate(obj_name)
            except Exception:  # pragma: no cover - unresolvable object
                continue

            total += 1
            if result["docstring"]:
                documented += 1

            for code, description in result["errors"]:
                if code in ignored_checks:
                    continue
                violations.append(f"{obj_name}: {code} {_normalize(description)}")

    return sorted(violations), documented, total


def main(argv=None):
    """Run the validation and report the result.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    exit_code : int
        ``0`` on success, ``1`` if unbaselined violations were found.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline",
        type=Path,
        help="file with known violations; only new violations fail the run",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="write the current violations to the --baseline file and succeed",
    )
    parser.add_argument(
        "--ignore",
        default=",".join(sorted(DEFAULT_IGNORED_CHECKS)),
        help="comma separated numpydoc check codes to skip",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="fail if less than this percentage of the objects are documented",
    )
    args = parser.parse_args(argv)

    ignored = {code.strip() for code in args.ignore.split(",") if code.strip()}
    violations, documented, total = collect_violations(ignored)

    coverage = 100.0 * documented / total if total else 100.0
    print(f"validated {total} objects, {documented} documented ({coverage:.1f}%)")

    if args.write_baseline:
        if args.baseline is None:
            parser.error("--write-baseline requires --baseline")
        args.baseline.write_text("\n".join(violations) + "\n", encoding="utf-8")
        print(f"wrote {len(violations)} violations to {args.baseline}")
        return 0

    known = set()
    if args.baseline is not None and args.baseline.exists():
        known = {
            line
            for line in args.baseline.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    new_violations = [v for v in violations if v not in known]

    print(f"{len(violations)} violations, {len(new_violations)} of them new")
    for violation in new_violations:
        print(f"  {violation}")

    if coverage + 1e-9 < args.min_coverage:
        print(
            f"documentation coverage {coverage:.1f}% is below the required "
            f"{args.min_coverage:.1f}%"
        )
        return 1

    return 1 if new_violations else 0


if __name__ == "__main__":
    sys.exit(main())
