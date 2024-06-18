import json
import logging
import os
from datetime import datetime
from pathlib import Path


def project_dir() -> Path:
    """Returns the path of the root directory."""
    return Path(__file__).resolve().parent


def datasets_dir() -> Path:
    """Returns the path of the `datasets` directory."""
    return Path(__file__).resolve().parent / "datasets"


def create_results_dir(directory: str, lm_name: str, methods: [str], config: dict, tasks: [] = []) -> Path:
    """
    Creates the results directory for the given dataset and methods.
    :param directory: The directory inside which the results directory is created.
    :param lm_name: The name of the LM used.
    :param methods: The list of methods used.
    :param config: Other configuration data for the results.
    :param tasks: The list of tasks used. leave empty if single task or no subdirectories are required for different tasks.
    :return: The path to the created results directory.
    """

    results_dir = os.path.join(directory, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    if tasks:
        for task in tasks:
            for method in methods:
                # create a results directory for the task and method
                os.makedirs(os.path.join(results_folder, task, method.__name__))
    else:
        for method in methods:
            # create a results directory for the method
            os.makedirs(os.path.join(results_folder, method.__name__))
    return Path(results_folder)
