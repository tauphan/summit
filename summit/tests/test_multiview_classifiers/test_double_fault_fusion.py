
import numpy as np
import unittest

from summit.multiview_platform.multiview_classifiers import double_fault_fusion


class Test_disagree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoview_decision_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        cls.monoview_decision_2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        cls.ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        cls.clf = double_fault_fusion.DoubleFaultFusion()

    def test_simple(cls):
        double_fault = cls.clf.diversity_measure(cls.monoview_decision_1,
                                                 cls.monoview_decision_2,
                                                 cls.ground_truth)
        np.testing.assert_array_equal(double_fault,
                                      np.array([False, True, False, False, False, False, True, False]))
