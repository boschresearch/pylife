


from .threepoint import ThreePointDetector
from .fkm import FKMRainflowDetector
from .recorders import GenericRainflowRecorder


class AbstractRainflowCounter:
    def __init__(self):
        self._recorder = GenericRainflowRecorder()

    @property
    def loops_from(self):
        return self._recorder.loops_from

    @property
    def loops_to(self):
        return self._recorder.loops_to

    def residuals(self):
        return self._detector.residuals

    def get_rainflow_matrix(self, bins):
        return self._recorder.matrix(bins)

    def get_rainflow_matrix_frame(self, bins):
        return self._recorder.matrix_frame(bins)


class RainflowCounterThreePoint(AbstractRainflowCounter):
    def __init__(self):
        super(RainflowCounterThreePoint, self).__init__()
        self._detector = ThreePointDetector(recorder=self._recorder)

    def process(self, samples):
        self._detector.process(samples)
        return self


class RainflowCounterFKM(AbstractRainflowCounter):
    def __init__(self):
        super(RainflowCounterFKM, self).__init__()
        self._detector = FKMRainflowDetector(recorder=self._recorder)

    def process(self, samples):
        self._detector.process(samples)
        return self
