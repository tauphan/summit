import builtins
from distutils.util import strtobool as tobool
import yaml
import os


def get_the_args(path_to_config_file="../config_files/config.yml"):
    """This is the main function for extracting the args for a '.yml' file"""
    # config_path = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(config_path, "../..")
    # path_to_config_file = os.path.join(config_path, path_to_config_file)

    with open(path_to_config_file, 'r') as stream:
        yaml_config = yaml.safe_load(stream)
    return yaml_config
