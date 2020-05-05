import unittest

import numpy as np

from summit.multiview_platform.multiview_classifiers import difficulty_fusion


class Test_difficulty_fusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.classifiers_decisions = cls.random_state.randint(
            0, 2, size=(5, 3, 5))
        cls.combination = [1, 3, 4]
        cls.y = np.array([1, 1, 0, 0, 1])
        cls.difficulty_fusion_clf = difficulty_fusion.DifficultyFusion()

    def test_simple(cls):
        difficulty_measure = cls.difficulty_fusion_clf.diversity_measure(
            cls.classifiers_decisions,
            cls.combination,
            cls.y)
        cls.assertAlmostEqual(difficulty_measure, 0.1875)
