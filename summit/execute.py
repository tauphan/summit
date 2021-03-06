"""This is the execution module, used to execute the code"""

import os


def execute(config_path=None):  # pragma: no cover
    import sys

    from summit.multiview_platform import exec_classif
    if config_path is None:
        exec_classif.exec_classif(sys.argv[1:])
    else:
        if config_path == "example 0":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_0.yml")
        elif config_path == "example 1":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_1.yml")
        elif config_path == "example 2.1.1":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_1_1.yml")
        elif config_path == "example 2.1.2":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_1_2.yml")
        elif config_path == "example 2.2":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_2.yml")
        elif config_path == "example 2.3":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_2_3.yml")
        elif config_path == "example 3":
            config_path = os.path.join(
                os.path.dirname(
                    os.path.realpath(__file__)),
                "examples",
                "config_files",
                "config_example_3.yml")
        exec_classif.exec_classif(["--config_path", config_path])


if __name__ == "__main__":
    execute()
