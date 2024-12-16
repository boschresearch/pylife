
import numpy as np
import pylife.stress.timesignal as TS


if __name__ == "__main__":
    np.random.seed(23424711)

    load_signal = TS.TimeSignalGenerator(
        10,
        {
            'number': 50_000,
            'amplitude_median': 50.0,
            'amplitude_std_dev': 0.5,
            'frequency_median': 4,
            'frequency_std_dev': 3,
            'offset_median': 0,
            'offset_std_dev': 0.4,
        },
        None,
        None,
    ).query(1_000_000)

    np.savetxt('load_signal.csv', load_signal)
