

import pylife.stress.rainflow as RF
import pylife.stress.rainflow.recorders as RFR

def test_three_point_detector_new_no_residuals():
    grr = RFR.GenericRainflowRecorder()
    dtor = RF.ThreePointDetector(recorder=grr)
    assert dtor.recorder is grr
    assert len(dtor.residuals) == 0
