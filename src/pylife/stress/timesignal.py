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

"""Handle sampled time signals for pyLife stress workflows.

Provide helpers for generating, resampling, filtering, spectral analysis, and
cleaning load-time histories before rainflow counting or other fatigue
post-processing.

Warnings
--------
This module is not considered finalized even though it is part of
``pylife-2.0``. Breaking changes might occur in upcoming minor releases.
"""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal

try:
    import tsfresh as ts

    _HAVE_TSFRESH = True
except ModuleNotFoundError:
    _HAVE_TSFRESH = False


class TimeSignalGenerator:
    r"""Generate mixed sinusoidal load-time signals.

    Create synthetic load signals, e.g. stress in MPa, for examples and tests
    in the time series handling workflow. The generated signal is a sum of
    sinusoidal components with normally distributed amplitudes, frequencies,
    offsets, and uniformly distributed phases.

    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz used when ``query`` advances the generated
        signal.
    sine_set : dict
        Definition of the sinusoidal components. Mandatory keys are
        ``"number"``, ``"amplitude_median"``, ``"amplitude_std_dev"``,
        ``"frequency_median"``, ``"frequency_std_dev"``,
        ``"offset_median"``, and ``"offset_std_dev"``.
    gauss_set : dict
        Reserved for Gaussian random components. The current implementation
        accepts the argument for compatibility but does not use it.
    log_gauss_set : dict
        Reserved for lognormal random components. The current implementation
        accepts the argument for compatibility but does not use it.

    Notes
    -----
    Each component is evaluated as

    .. math::

        s_i(t) = A_i \sin(\omega_i t + \phi_i) + c_i,

    where :math:`A_i`, :math:`\omega_i`, and :math:`c_i` are drawn from the
    normal distributions described by ``sine_set`` and :math:`\phi_i` is
    drawn uniformly from ``[0, 2\pi)``. The returned signal is

    .. math::

        S(t) = \sum_i s_i(t).
    """

    def __init__(self, sample_rate, sine_set, gauss_set, log_gauss_set):
        sine_amplitudes = stats.norm.rvs(
            loc=sine_set["amplitude_median"],
            scale=sine_set["amplitude_std_dev"],
            size=sine_set["number"],
        )
        sine_frequencies = stats.norm.rvs(
            loc=sine_set["frequency_median"],
            scale=sine_set["frequency_std_dev"],
            size=sine_set["number"],
        )
        sine_offsets = stats.norm.rvs(
            loc=sine_set["offset_median"],
            scale=sine_set["offset_std_dev"],
            size=sine_set["number"],
        )
        sine_phases = 2.0 * np.pi * np.random.rand(sine_set["number"])

        self.sine_set = list(
            zip(sine_amplitudes, sine_frequencies, sine_phases, sine_offsets)
        )

        self.sample_rate = sample_rate
        self.time_position = 0.0

    def query(self, sample_num):
        """Return the next contiguous chunk of the generated time signal.

        Use repeated calls to build a longer synthetic load-time history. The
        first returned sample follows the internal time cursor, and subsequent
        calls continue without a time gap.

        Parameters
        ----------
        sample_num : int
            Number of samples requested.

        Returns
        -------
        numpy.ndarray
            Requested samples of the generated load signal.
        """
        samples = np.zeros(sample_num)
        end_time_position = self.time_position + (sample_num - 1) / self.sample_rate

        for ampl, omega, phi, offset in self.sine_set:
            periods = np.floor(self.time_position / omega)
            start = self.time_position - periods * omega
            end = end_time_position - periods * omega
            time = np.linspace(start, end, sample_num)
            samples += ampl * np.sin(omega * time + phi) + offset

        self.time_position = end_time_position + 1.0 / self.sample_rate

        return samples

    def reset(self):
        """Reset the generator to the start of the time signal.

        After resetting, the next ``query`` call returns the same time
        positions as a newly created generator with the same random component
        parameters.
        """
        self.time_position = 0.0


def fs_calc(df):
    """Calculate the sampling rate of a time-indexed signal.

    Use this helper before filtering or spectral estimation when the sampling
    rate is encoded by an equidistant numeric time index.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series with a numeric time index in s.

    Returns
    -------
    float
        Sampling rate in Hz, rounded to the nearest integer value.

    Examples
    --------
    >>> from pylife.stress.timesignal import fs_calc
    >>> df = pd.DataFrame({"stress": [0.0, 1.0, 0.0]},
    ...                   index=[0.0, 0.5, 1.0])
    >>> float(fs_calc(df))
    2.0
    """
    try:
        fs = np.rint(1 / np.mean(np.diff(df.index)))
    except TypeError:
        print("Index has to be a number not a string. We assume fs = 1")
        fs = 1
    return fs


def resample_acc(df, fs=1):
    """Resample a pandas time series to an equidistant time index.

    Interpolate each column of a load-time history onto a new time index. This
    prepares measured signals for filters, PSD estimation, or rainflow
    counting algorithms that expect a constant sampling rate.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series with a numeric time index in s and load columns, e.g.
        stress in MPa.
    fs : float, optional
        Sampling rate in Hz of the resampled time series. Default is ``1``.

    Returns
    -------
    pandas.DataFrame
        Resampled time series with an equidistant time index in s.
    """
    index_new = np.arange(df.index.min(), df.index.max() + 1 / fs, 1 / fs)

    df_rs = pd.DataFrame(
        df.apply(lambda x: np.interp(index_new, df.index, x)).values,
        index=index_new,
        columns=df.columns,
    )
    return df_rs


def butter_bandpass(df, lowcut, highcut, order=5):
    """Apply a zero-phase Butterworth band-pass filter to a time signal.

    Filter each column of a load-time history before fatigue evaluation. This
    is a thin wrapper around :func:`scipy.signal.butter` and
    :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series with a numeric time index in s and load columns, e.g.
        stress in MPa.
    lowcut : float
        Lower cut-off frequency in Hz.
    highcut : float
        Upper cut-off frequency in Hz.
    order : int, optional
        Butterworth filter order. Default is ``5``.

    Returns
    -------
    pandas.DataFrame
        Filtered time series with the same time index and columns as ``df``.

    Notes
    -----
    The filter is an ``order``-th order digital Butterworth band-pass filter
    applied forward and backward with :func:`scipy.signal.filtfilt`, resulting
    in zero phase shift.
    """
    fs = fs_calc(df)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return df.apply(lambda x: signal.filtfilt(b, a, x, padlen=int(fs / 2)))


def psd_df(df_ts, nfft=512, nperseg=256):
    """Estimate power spectral density columns with Welch's method.

    Convert a time-domain load signal into a frequency-indexed PSD for
    spectral comparison or frequency-domain fatigue workflows. This is a thin
    wrapper around :func:`scipy.signal.welch`.

    Parameters
    ----------
    df_ts : pandas.DataFrame
        Time series with a numeric time index in s and load columns, e.g.
        stress in MPa.
    nfft : int, optional
        Length of the FFT used by Welch's method. Default is ``512``.
    nperseg : int, optional
        Segment length used by Welch's method. Values larger than ``nfft`` are
        clipped to ``nfft``. Default is ``256``.

    Returns
    -------
    pandas.DataFrame
        Power spectral density with frequency index in Hz and one PSD column
        per input column.

    Notes
    -----
    The PSD scaling is the :func:`scipy.signal.welch` default
    ``scaling="density"``. For a load signal in MPa, the resulting PSD has
    units MPa²/Hz and integrates to the mean square value of the signal.
    """

    nperseg = min(nperseg, nfft)
    fs = fs_calc(df_ts)
    df_psd = {}
    for col in df_ts:
        freq, df_psd[col] = signal.welch(
            df_ts[col].values, fs=fs, nperseg=nperseg, nfft=nfft
        )
    df_psd = pd.DataFrame(df_psd, index=pd.Index(freq, name="frequency"))
    return df_psd


def _prepare_rolling(df):
    """Add ``id`` and relative ``time`` columns for tsfresh rolling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series with a numeric time index in s.

    Returns
    -------
    pandas.DataFrame
        Output data with added ``id`` and relative ``time`` columns.
    """
    prep_roll = df.copy()
    prep_roll["id"] = 0
    prep_roll["time"] = df.index.values
    prep_roll["time"] = prep_roll["time"].subtract(prep_roll["time"].values[0])
    prep_roll.index = prep_roll["time"]

    return prep_roll


def _roll_dataset(prep_roll_df, window_size=1000, overlap=200):
    """Roll a prepared time series into overlapping windows.

    Parameters
    ----------
    prep_roll_df : pandas.DataFrame
        Output from ``_prepare_rolling``.
    window_size : int, optional
        Window size of the rolled segments in samples. Default is ``1000``.
    overlap : int, optional
        Overlap between adjacent windows in samples. Default is ``200``.

    Returns
    -------
    pandas.DataFrame
        Rolled data frame for feature extraction with tsfresh.
    """

    # Create Rolled Dataset with Parameter rolling_direction & window_size
    # throws away the last halfshift
    rolling_direction = window_size - overlap
    cycles = int((len(prep_roll_df) - window_size) / rolling_direction) + 1

    parts = []
    # shiften
    for i in range(cycles):
        position = (rolling_direction) * i
        shift = prep_roll_df.iloc[position : position + window_size, :].copy()
        # change IDs to format (id,time)
        df = pd.DataFrame(
            {
                "id": np.int64(np.zeros(len(shift), dtype=int)),
                "max_time": shift.iloc[-1, -1],
            }
        )

        shift["id"] = pd.MultiIndex.from_frame(df).to_numpy()

        parts.append(shift)

    return pd.concat(parts, ignore_index=True)


def _extract_feature_df(df_rolled, feature="maximum"):
    """Extract one tsfresh feature from each rolled window.

    Parameters
    ----------
    df_rolled : pandas.DataFrame
        Rolled data frame from ``_roll_dataset``.
    feature : str, optional
        Feature calculator name from tsfresh. Only calculators without extra
        parameters are supported. Default is ``"maximum"``.

    Returns
    -------
    pandas.DataFrame
        Extracted feature values, one row per rolled window.
    """
    # extract features

    # fc_parameters = {"abs_energy", "maximum"}
    fc_parameters = {
        feature: None,
    }
    extracted_features = ts.extract_features(
        df_rolled,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters,
        n_jobs=0,
    )
    extracted_features.index = range(len(extracted_features))
    return extracted_features


def _select_relevant_windows(
    prep_roll,
    extracted_features,
    comparison_column_ex,
    fraction_max=0.25,
    window_size=1000,
    overlap=200,
    n_gridpoints=3,
    method="keep",
):
    """Select windows by comparing one extracted feature with a threshold.

    Parameters
    ----------
    prep_roll : pandas DataFrame
        Prepared input data, normally output from ``_prepare_rolling``.
    extracted_features : pandas.DataFrame
        Feature values returned by ``_extract_feature_df``.
    comparison_column_ex : str
        Name of the extracted feature column. It is built as
        ``comparison_column + "__" + feature``.
    fraction_max : float
        Fraction of the maximum extracted feature used as threshold.
    window_size : int
        Window size of the rolled segments in samples. Default is ``1000``.
    overlap : int, optional
        Overlap between adjacent windows in samples. Default is ``200``.
    n_gridpoints : int, optional
        Number of grid points left to support polynomial interpolation.
        Default is ``3``.
    method : {'keep', 'remove'}, optional
        Return convention. ``"keep"`` drops low-feature windows, while
        ``"remove"`` returns the low-feature windows. Default is ``"keep"``.

    Returns
    -------
    pandas.DataFrame
        Relevant windows according to ``method``.
    """
    # get added up abs energy of interval x, if too low set None
    rolling_direction = window_size - overlap

    relevant_feature = extracted_features[comparison_column_ex]
    relevant_windows = prep_roll.copy()
    just_added_NaNs = False
    liste = []
    for i in range(len(extracted_features)):
        if relevant_feature[i] <= relevant_feature.max() * fraction_max:
            if just_added_NaNs is True:
                liste.append(
                    list(
                        range(
                            0 + i * rolling_direction,
                            window_size + i * rolling_direction,
                        )
                    )
                )

            else:
                liste.append(
                    list(
                        range(
                            0 + i * rolling_direction,
                            window_size + i * rolling_direction - n_gridpoints,
                        )
                    )
                )
                relevant_windows.iloc[
                    i * rolling_direction
                    + window_size
                    - n_gridpoints : i * rolling_direction
                    + window_size,
                    0 : relevant_windows.shape[1] - 2,
                ] = None
                just_added_NaNs = True
        else:
            just_added_NaNs = False

    index_liste = []
    """
    tail = (len(prep_roll)-window_size) % rolling_direction+1
    for i in range(tail):
        liste.append(len(prep_roll)-i-1)
    """
    liste = list(pd.core.common.flatten(liste))
    liste = list(set(liste))
    for i in range(len(liste)):
        index_liste.append(relevant_windows.index[liste[i]])
    if method == "keep":
        relevant_windows = relevant_windows.drop(index_liste, axis=0)
    elif method == "remove":
        relevant_windows = relevant_windows.loc[index_liste]
    return relevant_windows


def _polyfit_gridpoints(grid_points, prep_roll, order=3, verbose=False, n_gridpoints=3):
    """Fill grid points by polynomial interpolation.

    Parameters
    ----------
    grid_points : pandas.DataFrame
        Data frame with gaps marked as ``NaN`` values.
    prep_roll : pandas.DataFrame
        Prepared time series used to create the time axis.
    order : int, optional
        Polynomial interpolation order. Default is ``3``.
    verbose : bool, optional
        Accepted for compatibility. The current implementation does not use
        this argument. Default is ``False``.
    n_gridpoints : int, optional
        Number of grid points. Default is ``3``.

    Returns
    -------
    pandas.DataFrame
        Data frame with polynomial values at the grid points.
    """

    # add a null row at the start and reset time index
    delta_t = prep_roll.index[1] - prep_roll.index[0]
    line = pd.DataFrame(grid_points.iloc[:1], index=[-delta_t])
    grid_points = pd.concat([grid_points, line], ignore_index=False)
    grid_points.index = grid_points.index + delta_t
    poly_gridpoints = grid_points.sort_index()
    poly_gridpoints.iloc[0, :] = 0
    ts_time = prep_roll.iloc[: len(poly_gridpoints)]

    poly_gridpoints["time"] = ts_time.index.values
    poly_gridpoints.index = poly_gridpoints["time"]

    # %% smooth the gaps with polynomial values
    poly_gridpoints.interpolate(method="polynomial", order=order, inplace=True)

    return poly_gridpoints


def clean_timeseries(
    df,
    comparison_column,
    window_size=1000,
    overlap=800,
    feature="abs_energy",
    method="keep",
    n_gridpoints=3,
    percentage_max=0.05,
    order=3,
):
    r"""Clean a time series by removing low-feature windows.

    Use this helper to reduce a load-time history before rainflow counting.
    The function rolls the signal into windows, extracts one tsfresh feature,
    removes or keeps windows based on a threshold, and fills short gaps by
    polynomial interpolation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series with a numeric time index in s and load columns,
        e.g. stress in MPa.
    comparison_column : str
        Column used for feature comparison with ``percentage_max``.
    window_size : int, optional
        Window size of the rolled segments in samples. Default is ``1000``.
    overlap : int, optional
        Overlap between adjacent windows in samples. Default is ``800``.
    feature : str, optional
        Feature calculator name from tsfresh. Only calculators without extra
        parameters are supported. Default is ``"abs_energy"``.
    method : {'keep', 'remove'}, optional
        Return convention. ``"keep"`` keeps windows above the feature
        threshold, while ``"remove"`` keeps windows at or below it. Default is
        ``"keep"``.
    n_gridpoints : int, optional
        Number of grid points used to bridge removed windows. Default is
        ``3``.
    percentage_max : float, optional
        Minimum fraction of the maximum feature value for a window to remain
        in ``"keep"`` mode. Default is ``0.05``.
    order : int, optional
        Polynomial interpolation order. Default is ``3``.

    Returns
    -------
    pandas.DataFrame
        Cleaned time series with the helper ``id`` column removed.

    Raises
    ------
    ImportError
        Raised if tsfresh is not installed.

    Notes
    -----
    A window is classified as low-feature if

    .. math::

        x_i \leq p \max_j x_j,

    where :math:`x_i` is the selected feature value and :math:`p` is
    ``percentage_max``.
    """

    if not _HAVE_TSFRESH:
        raise ImportError(
            "tsfresh and dependencies are not installed. "
            "Use `pip install pylife[tsfresh]` to install it."
        )

    df_prep = _prepare_rolling(df)
    ts_time = df_prep.copy()
    # adding a row
    delta_t = ts_time.index[1] - ts_time.index[0]
    line = pd.DataFrame(ts_time.iloc[:1], index=[-delta_t])
    ts_time = pd.concat([ts_time, line], ignore_index=False)
    ts_time.index = ts_time.index + delta_t
    ts_time = ts_time.sort_index()

    ts_time["time"] = ts_time.index.values

    comparison_column_ex = comparison_column + "__" + feature
    df_rolled = _roll_dataset(df_prep, window_size=window_size, overlap=overlap)
    extracted_features = _extract_feature_df(df_rolled, feature)
    grid_points = _select_relevant_windows(
        df_prep,
        extracted_features,
        comparison_column_ex,
        percentage_max,
        window_size,
        overlap,
        method=method,
    )

    poly_gridpoints = _polyfit_gridpoints(
        grid_points, ts_time, order=order, verbose=False, n_gridpoints=n_gridpoints
    )

    # Remove NaN's at the end - should be maximum 2n
    cleaned = poly_gridpoints.dropna(axis=0, how="any")
    cleaned.pop("id")

    return cleaned
