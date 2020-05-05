import unittest
import numpy as np
import pandas as pd

from summit.multiview_platform.result_analysis import feature_importances
from summit.multiview_platform.monoview.monoview_utils import MonoviewResult


class FakeClassifier:
    def __init__(self, i=0):
        self.feature_importances_ = [i, i + 1]


class FakeClassifierResult(MonoviewResult):

    def __init__(self, i=0):
        self.i = i
        self.hps_duration = i * 10
        self.fit_duration = (i + 2) * 10
        self.pred_duration = (i + 5) * 10
        self.clf = FakeClassifier(i)
        self.view_name = 'testview' + str(i)
        self.classifier_name = "test" + str(i)

    def get_classifier_name(self):
        return self.classifier_name


class Test_get_duration(unittest.TestCase):

    def test_simple(self):
        results = [FakeClassifierResult(), FakeClassifierResult(i=1)]
        feat_importance = feature_importances.get_feature_importances(results)
        pd.testing.assert_frame_equal(feat_importance["testview1"],
                                      pd.DataFrame(index=None, columns=['test1'],
                                                   data=np.array(
                                                       [1, 2]).reshape((2, 1)),
                                                   ), check_dtype=False)
