import pylife.stress.rainflow as RF


def test_threepoint(benchmarker):
    benchmark_time = 0.05  # seconds
    elapsed = benchmarker(RF.ThreePointDetector(recorder=RF.FullRecorder()))

    assert (
        elapsed < benchmark_time
    ), f"Benchmark time of {benchmark_time} s not exceeded. Needed {elapsed:0.4f} s."


def test_fourpoint(benchmarker):
    benchmark_time = 0.05  # seconds
    elapsed = benchmarker(RF.FourPointDetector(recorder=RF.FullRecorder()))

    assert (
        elapsed < benchmark_time
    ), f"Benchmark time of {benchmark_time} s not exceeded. Needed {elapsed:0.4f} s."
