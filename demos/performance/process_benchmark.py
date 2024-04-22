import pylife.stress.timesignal as TS
import time
from wrapper import process_rainflow, process_meanstress_conversion, process_damage_calculation
from matplotlib import pyplot as plt
import numpy as np

#Generate load signal
load_signal = TS.TimeSignalGenerator(
        10,
        {
            'number': 50_000,
            'amplitude_median': 50.0, 'amplitude_std_dev': 0.5,
            'frequency_median': 4, 'frequency_std_dev': 3,
            'offset_median': 0, 'offset_std_dev': 0.4
        }, None, None
    ).query(50000)

# Get computation time for rainflow four point
tic = time.perf_counter()
mean_arr, amplitude_arr = process_rainflow(load_signal)
toc = time.perf_counter()
print(f"Processing rainflow took {toc - tic:0.4f} seconds")

# plt.plot(amplitude_arr)
# plt.show()

# Get computation time for meanstress conversion
tic = time.perf_counter()
amplitude_corr = process_meanstress_conversion(mean_arr, amplitude_arr)
toc = time.perf_counter()
print(f"Computation of meanstress conversion took {toc - tic:0.4f} seconds")

# plt.plot(amplitude_corr)
# plt.show()

# Get computation time for damage calculation
tic = time.perf_counter()
D = process_damage_calculation(amplitude_corr, k = 5, k1 = 5, Sd= 230, Nd = 1e6)
toc = time.perf_counter()
print(f"Computation of damage took {toc - tic:0.4f} seconds")