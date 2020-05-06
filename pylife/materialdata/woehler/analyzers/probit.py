from .elementary import Elementary

import numpy as np
import scipy.stats as stats
import pylife.utils.functions as functions
from pylife.utils.probability_data import ProbabilityFit


class Probit(Elementary):
    def _specific_analysis(self, wc):
        wc['1/TS'], wc['SD_50'], wc['ND_50'] = self.__probit_analysis()
        return wc

    def __probit_rossow_estimation(self):
        inf_groups = self._fd.infinite_zone.groupby('load')
        frac_num = inf_groups.fracture.sum().astype(int).to_numpy()
        tot_num = inf_groups.fracture.count().to_numpy()

        fprobs = np.empty_like(frac_num, dtype=np.float)
        c1 = frac_num == 0
        w = np.where(c1)
        fprobs[w] = 1. - 0.5**(1./tot_num[w])

        c2 = frac_num == tot_num
        w = np.where(c2)
        fprobs[w] = 0.5**(1./tot_num[w])

        c3 = np.logical_and(np.logical_not(c1), np.logical_not(c2))
        w = np.where(c3)
        fprobs[w] = (3*frac_num[w] - 1) / (3*tot_num[w] + 1)

        return fprobs, inf_groups.load.mean()

    def __probit_analysis(self):
        fprobs, load = self.__probit_rossow_estimation()

        self._probability_fit = ProbabilityFit(fprobs, load)

        TS_inv = functions.std2scatteringRange(1./self._probability_fit.slope)
        SD_50 = 10**(-self._probability_fit.intercept/self._probability_fit.slope)
        ND_50 = self._transition_cycles(SD_50)

        return TS_inv, SD_50, ND_50

    def fitter(self):
        return self._probability_fit
