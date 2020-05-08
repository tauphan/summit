import os
import unittest
import yaml
import numpy as np

from summit.tests.utils import rm_tmp, tmp_path
from summit.multiview_platform.utils import configuration


class Test_get_the_args(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.path_to_config_file = tmp_path + "config_temp.yml"
        os.mkdir(tmp_path)
        data = {"log": 10, "name": [12.5, 1e-06], "type": True}
        with open(cls.path_to_config_file, "w") as config_file:
            yaml.dump(data, config_file)

    @classmethod
    def tearDownClass(cls):
        os.remove(tmp_path + "config_temp.yml")
        os.rmdir(tmp_path)

    def test_file_loading(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(type(config_dict), dict)

    def test_dict_format(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertIn("log", config_dict)
        self.assertIn("name", config_dict)

    def test_arguments(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(config_dict["log"], 10)
        self.assertEqual(config_dict["name"], [12.5, 1e-06])
        self.assertEqual(config_dict["type"], True)


class Test_save_config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rm_tmp()
        path_file = os.path.dirname(os.path.abspath(__file__))
        make_tmp_dir = os.path.join(path_file, "../tmp_tests")
        os.mkdir(make_tmp_dir)

    def test_simple(self):
        configuration.save_config(tmp_path, {"test": 10})
        with open(os.path.join(tmp_path, "config_file.yml"), 'r') as stream:
            yaml_config = yaml.safe_load(stream)
        self.assertEqual(yaml_config, {"test": 10})

    @classmethod
    def tearDownClass(cls):
        os.remove(os.path.join(tmp_path, "config_file.yml"))


if __name__ == '__main__':
    unittest.main()
