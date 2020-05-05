import unittest
import numpy as np
import pandas as pd
import os

from summit.multiview_platform.result_analysis import tracebacks_analysis
from summit.tests.utils import tmp_path, rm_tmp


class FakeClassifierResult:

    def __init__(self, i=0):
        self.i = i
        if i == 0:
            self.hps_duration = 10
            self.fit_duration = 12
            self.pred_duration = 15
        else:
            self.hps_duration = 1
            self.fit_duration = 2
            self.pred_duration = 5

    def get_classifier_name(self):
        if self.i == 0:
            return 'test1'
        else:
            return 'test2'


class Test_funcs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.res_file = open(os.path.join(tmp_path, "tmp.txt"), "w")

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_save_dict_to_text(self):
        keys = tracebacks_analysis.save_dict_to_text(
            {"a": "i", "b": "j"}, self.res_file)
        self.res_file.close()
        self.assertEqual(list(keys), ["a", "b"])
        with open(os.path.join(tmp_path, "tmp.txt"), 'r') as res_file:
            self.assertEqual(
                res_file.read(),
                'Failed algorithms : \n\ta,\n\tb.\n\n\na\n\ni\n\n\nb\n\nj\n\n\n')
