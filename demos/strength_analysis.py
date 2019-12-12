
import numpy as np
import pandas as pd

import pylife.stress.equistress
import pylife.strength.meanstress
import pylife.strength.infinite
import pylife.materialdata.woehler

from pylife.utils import meshplot

import matplotlib.pyplot as plt

filename = 'plate_with_hole.h5'

stress = pd.read_hdf(filename, 'node_data')
stress['S13'] = np.zeros_like(stress['S11'])
stress['S23'] = np.zeros_like(stress['S11'])

goodman = {
    'M': 0.5,
    'M2': 0.5/3.
}

R_goal = -1.

woehler_data = {
    'strength_inf': 450.,
    'strength_scatter': 1.3
}

result = (stress.groupby('element_id').mean()
          .equistress.mises().rename(columns={'mises': 'sigma_a'})
          .cyclic_stress.constant_R(0.0)
          .meanstress_mesh.FKM_goodman(pd.Series(goodman), R_goal)
          .infinite_security.factors(pd.Series(woehler_data), allowed_failure_probability=1e-6))

print("Lowest security factor:", result.security_factor.min())

fig, ax = plt.subplots()
meshplot.plotmesh(ax, stress, 1./result.security_factor.to_numpy())
plt.show()
