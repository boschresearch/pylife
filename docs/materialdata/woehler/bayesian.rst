The ``Bayesian`` module
#######################

Until version 2.1.0 pyLife had a module for Bayesian Wöhler analysis.  The
Bayesian Wöhler analysis is not well established in the community, so it was a
bit ahead of its time.  Now that we have collected experiences with it, it
turns out, that its result are too often inaccurate.  Therefore we decided to
disable it.

That means, that the code remains in the code base, we are only preventing you
from importing it.  That means, if you want to experiment with the code, you
can grab the module and delete the exception raising.  We do not recommend to
use it for production, though.

If you want to propose an improvement of the code that leads to better results,
we are open for PRs (see :doc:`/CONTRIBUTING`) and might eventually enable it
again.
