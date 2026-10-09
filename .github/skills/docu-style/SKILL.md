---
name: docu-style
description: Write or review pyLife docstrings and narrative documentation using the numpy/scipy (numpydoc) style, rendered via Sphinx napoleon. Use whenever asked to document, docstring, or improve documentation for pyLife Python code (functions, classes, modules, PylifeSignal accessors) or to review existing docs for style/UX compliance.
user-invocable: true
---

# NumPy/SciPy-style Documentation for pyLife

Use this skill to write new docstrings, rewrite existing ones, or review
documentation for compliance with the numpydoc conventions ([numpy docs
guide](https://numpydoc.readthedocs.io/en/latest/format.html),
[numpy.org](https://numpy.org/doc/stable/index.html),
[docs.scipy.org](https://docs.scipy.org/doc/scipy/index.html)) as already
used throughout pyLife (`docs/conf.py` enables `sphinx.ext.napoleon` with
`napoleon_custom_sections = ["Limitations"]`).

## Core principle: write for the pyLife user, not the author

Every docstring must let a scientist/engineer using pyLife understand **what
a function/class does, what to pass in, what they get back, and what units
or assumptions apply** without reading the implementation. Prefer plain
engineering language over internal jargon. If a symbol has a physical
meaning (stress, cycle number, slope, scattering range, ...), say so.

## Docstring skeleton (numpydoc section order)

Use only the sections that are needed; never invent empty ones.

```python
def function_name(param1, param2=default):
    """One-line summary in imperative mood, ending with a period.

    Optional extended summary giving context: why this exists, what
    domain concept it represents, and how it fits into the pyLife
    workflow. Wrap prose at ~79 columns like the rest of the codebase.

    Parameters
    ----------
    param1 : type
        Description of param1, including units if physical
        (e.g. "stress amplitude in MPa").
    param2 : type, optional
        Description. Mention the default and when to override it.
        Default is ``default``.

    Returns
    -------
    result_name : type
        Description of the return value(s). Name multiple return
        values on separate lines when a function returns a tuple.

    Raises
    ------
    ValueError
        Condition under which this is raised.

    Notes
    -----
    Background, formulas (rendered via ``.. math::`` if needed),
    algorithmic detail, or references to standards (e.g. DIN 50100).

    Examples
    --------
    >>> function_name(1.0)
    2.0

    See Also
    --------
    other_function : One-line relation to this function.

    References
    ----------
    .. [1] Author, "Title", Publisher, Year.
    """
```

Section order (only include what applies): short summary → extended
summary → `Parameters` → `Returns` → `Yields` → `Raises` → `Warns` →
`Warnings` → `See Also` → `Notes` → `Examples` → `References`. pyLife also
allows a custom `Limitations` section (registered in
`napoleon_custom_sections`) for stating physical/numerical validity ranges
— use it whenever a method only holds under certain assumptions.

## Formatting rules (from numpy/scipy style guides)

1. **Summary line**: a single sentence, imperative mood ("Convert...",
   "Compute...", not "Converts..."), on the line right after the opening
   `"""`, no blank line before it.
2. **Underlines** for section headers must match the header text length
   exactly (`Parameters` → 10 dashes, `Returns` → 7 dashes, etc.).
3. **Parameter/return typing**: `name : type` with a single space around
   the colon. Use concrete types (`float`, `pandas.DataFrame`,
   `numpy.ndarray`, `int`), add `, optional` for keyword args with
   defaults, and use `{'a', 'b'}` for a fixed set of string choices.
4. Indent descriptions by 4 spaces under the `name : type` line; wrap
   long descriptions rather than using one long line.
5. **Cross references**: use Sphinx roles for other objects,
   e.g. :class:`~pylife.materiallaws.WoehlerCurve`, :func:`numpy.log10`,
   :meth:`.some_method` — this is what makes `sphinx.ext.autodoc` /
   `intersphinx` (already configured for numpy/scipy/pandas/sklearn in
   `docs/conf.py`) generate working links.
6. **Examples** must be valid, runnable doctest (`sphinx.ext.doctest` is
   enabled) — verify with `python -m doctest` mentally or by running the
   snippet before finalizing.
7. Use double backticks for inline code/values in prose (e.g. ``` ``TN`` ```),
   matching existing pyLife docstrings (see
   `src/pylife/materiallaws/woehlercurve.py`,
   `src/pylife/utils/functions.py`).
8. Keep line length consistent with the surrounding module (~79-100 cols)
   and match the file's existing indentation style exactly.

## pyLife-specific conventions to preserve

- **Class docstrings for `PylifeSignal` accessors** (see
  `WoehlerCurve`): document the *signal contract* — mandatory keys, optional
  keys with their default/derivation, and the physical meaning of each key —
  as a bulleted list right in the class docstring, not just parameter
  types.
- **Physical units and standards**: whenever a quantity has a technical
  standard behind it (DIN, FKM, Wöhler/SN-curve terminology), name it in
  `Notes` so users can look it up.
- Module-level docstrings for new modules should briefly state the
  module's purpose and, if relevant, which standard/algorithm it
  implements.
- Do not remove the Apache License header when editing files.

## Workflow

1. Read the target function/class/module fully (signature, body, existing
   docstring) to understand real behavior — do not guess parameter
   semantics from names alone.
2. For new/rewritten docstrings, follow the skeleton above; omit
   inapplicable sections.
3. Check formatting against the rules above (underline lengths, spacing,
   section order).
4. If documenting a `PylifeSignal` accessor, verify the mandatory/optional
   key list against `_validate()`.
5. If Sphinx docs exist for the touched area (`docs/*.rst`), keep
   cross-references and toctree entries consistent; do not duplicate
   content already generated via autodoc.
6. Validate: run `python -c "import ast; ast.parse(open(r'<file>').read())"`
   or the project's tests/lint to ensure the docstring didn't break
   parsing, and mentally re-read as a first-time pyLife user would.

## When not to use it

- Pure narrative user guides unrelated to source docstrings (e.g. README
  prose) — numpydoc section formatting doesn't apply there, though the
  same clarity/UX principles still should.
- Non-Python code in the repository.
