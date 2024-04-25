import sys
import pylife.stress.timesignal as TS
import time
from wrapper import process_rainflow, process_meanstress_conversion, process_damage_calculation
from matplotlib import pyplot as plt
import numpy as np



# Generate load signal

# load_signal = TS.TimeSignalGenerator(
#         10,
#         {
#             'number': 50_000,
#             'amplitude_median': 50.0, 'amplitude_std_dev': 0.5,
#             'frequency_median': 4, 'frequency_std_dev': 3,
#             'offset_median': 0, 'offset_std_dev': 0.4
#         }, None, None
#     ).query(1_000_000)

# np.savetxt('load.txt', load_signal)

# sys.exit()

load_signal = np.loadtxt('load.txt')

from pyinstrument import Profiler

profiler = Profiler(interval=0.001)
profiler.start()

# Get computation time for rainflow four point
tic = time.perf_counter()
mean_arr, amplitude_arr = process_rainflow(load_signal)
toc = time.perf_counter()
print(f"Processing rainflow took {toc - tic:0.4f} seconds")


# np.savetxt('mean_arr.txt', mean_arr)
# np.savetxt('amplitude_arr.txt', amplitude_arr)

# sys.exit()

# # plt.plot(amplitude_arr)
# # plt.show()

# mean_arr = np.loadtxt('mean_arr.txt')
# amplitude_arr = np.loadtxt('amplitude_arr.txt')


# Get computation time for meanstress conversion
# tic = time.perf_counter()
# amplitude_corr = process_meanstress_conversion(mean_arr, amplitude_arr)
# toc = time.perf_counter()
# print(f"Computation of meanstress conversion took {toc - tic:0.4f} seconds")

# plt.plot(amplitude_corr)
# plt.show()

# Get computation time for damage calculation
# tic = time.perf_counter()
# D = process_damage_calculation(amplitude_corr, k = 5, k1 = 5, Sd= 230, Nd = 1e6)
# toc = time.perf_counter()
# print(f"Computation of damage took {toc - tic:0.4f} seconds")

profiler.stop()

profiler.print()
