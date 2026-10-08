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

"""Test that rainflow detectors default to a LoopValueRecorder."""

__author__ = "Johannes Mueller"
__maintainer__ = __author__

import pytest
import pandas as pd

import pylife.stress.rainflow as RF
import pylife.stress.rainflow.general as RFG
import pylife.stress.rainflow.recorders as RFR


SIGNAL = [0.0, 3.0, -1.0, 2.0, -2.0, 0.0]


@pytest.mark.parametrize("detector_class", [
    RF.ThreePointDetector,
    RF.FourPointDetector,
    RF.FKMDetector,
])
def test_detector_without_recorder_creates_loop_value_recorder(detector_class):
    detector = detector_class()
    assert isinstance(detector.recorder, RF.LoopValueRecorder)


@pytest.mark.parametrize("detector_class", [
    RF.ThreePointDetector,
    RF.FourPointDetector,
    RF.FKMDetector,
])
def test_detector_without_recorder_matches_explicit_recorder(detector_class):
    default_detector = detector_class()
    explicit_detector = detector_class(recorder=RF.LoopValueRecorder())

    default_detector.process(SIGNAL, flush=True)
    explicit_detector.process(SIGNAL, flush=True)

    pd.testing.assert_frame_equal(
        default_detector.recorder.collective, explicit_detector.recorder.collective
    )


@pytest.mark.parametrize("detector_class", [
    RF.ThreePointDetector,
    RF.FourPointDetector,
    RF.FKMDetector,
])
def test_detectors_without_recorder_do_not_share_state(detector_class):
    first_detector = detector_class()
    second_detector = detector_class()

    assert first_detector.recorder is not second_detector.recorder

    first_detector.process(SIGNAL, flush=True)

    assert len(second_detector.recorder.collective) == 0


@pytest.mark.parametrize("detector_class", [
    RF.ThreePointDetector,
    RF.FourPointDetector,
    RF.FKMDetector,
])
def test_detector_accepts_recorder_positionally(detector_class):
    recorder = RFR.FullRecorder() if detector_class is not RF.FKMDetector else RF.LoopValueRecorder()
    detector = detector_class(recorder)
    assert detector.recorder is recorder


@pytest.mark.parametrize("detector_class", [
    RF.ThreePointDetector,
    RF.FourPointDetector,
    RF.FKMDetector,
])
def test_detector_accepts_recorder_by_keyword(detector_class):
    recorder = RFR.FullRecorder() if detector_class is not RF.FKMDetector else RF.LoopValueRecorder()
    detector = detector_class(recorder=recorder)
    assert detector.recorder is recorder


def test_abstract_recorder_is_reexported_from_general():
    assert RFG.AbstractRecorder is RFR.AbstractRecorder


def test_abstract_recorder_importable_from_package():
    assert RF.AbstractRecorder is RFR.AbstractRecorder
