import pylife.stress.rainflow as rainflow
from pylife.strength import meanstress, Fatigue
import numpy as np
import pandas as pd

def process_rainflow(load_signal,residuum = True):
    """
    Wrapper function for calling the pylife rainflow module.

    Parameters
    ----------
    load_signal : array
        Series, to be counted (no filtering will be applied, will be reduced to turning points).
    residuum : bool, optional
        if residuum cycles should be counted.
        If True, they will be counted as full cycles. The default is True.

    Returns
    -------
    mean : array
        Mean load of the cycles (unbinned).
    amplitude : arry
        Amplitude load of the cycles (unbinned).
    """
    recorder = rainflow.LoopValueRecorder()
    detector = rainflow.FourPointDetector(recorder=recorder)
    detector.process(load_signal)
    if residuum:
            residuals = detector.residuals
            detector.process(residuals)

    f, t = np.array(recorder.values_from), np.array(recorder.values_to)
    mean = (f + t) / 2
    amplitude = np.abs((f - t) / 2)

    return mean, amplitude

def process_meanstress_conversion(mean, amplitude, M=0, M2=0, R_goal = 0.0):
    """
    Wrapper function for computing meanstress conversion

    Parameters
    ----------
    amplitude : array
        Series of stress amplitudes
    mean : array
        Series, mean stresses
    M : float
        the mean stress sensitivity between R=-inf and R=0. The default is 0.
    M2 : float
        the mean stress sensitivity beyond R=0. The default is 0.
    R_goal : float, optional
        the R-value to transform to. The default is 0.

    Returns
    -------
    amplitude : array
        Series of stresses after meanstress conversion
    """
    amplitude_corr = meanstress.fkm_goodman(np.array(amplitude), np.array(mean), M, M2, R_goal=R_goal)

    return amplitude_corr

def process_damage_calculation(amplitude_corr, k, k1, Sd, Nd = 1e6):
    """
    Wrapper function for calculating the damage sum directly from a series, without binning the load.

    Parameters
    ----------
    amplitude_corr : array
        Series of stresses after meanstress conversion.
    Sd : float
        Endurance strength, expected to be in load range format.
    Nd : int, optional
        Endurance cycle count. Default is 1e6
    k : float
        Woehler exponent above Sd.
    k1 : TYPE
        Woehler exponent below Sd.
    Returns
    -------
    D : float
        Damage value.
    """
    df = pd.DataFrame()
    df['amplitude'] = 2 * amplitude_corr
    df['cycles'] = 1
    D = Fatigue(pd.Series({'k_1': k, 'k_2': k1, 'ND': Nd, 'SD': Sd})).damage(df).sum()

    return D
