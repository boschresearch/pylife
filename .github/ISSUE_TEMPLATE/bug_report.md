---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Code snippet to reproduce the bug (no screenshot but pasted code)
```
import pylife

# code that triggers the bug
```

**Expected result**
A clear and concise description of what you expected to happen.

**Observed result**
Paste the observed result with the whole backtrace (again, no screenshots but pasted test)
Like for example:
```
In [5]: divide(1,0)
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-5-4fffc6193274> in <module>
----> 1 divide(1,0)

<ipython-input-3-6377349725d3> in divide(a, b)
      1 def divide(a,b):
----> 2     return a/b
      3 

ZeroDivisionError: division by zero
```

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu Linux 20.04, Windows 10]
 - How installed [pip, manually]
 - Version [1.01, branch, commit]

**Additional context**
Add any other context about the problem here.
