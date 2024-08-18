import datetime
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Callable

from tqdm import tqdm

import project_utils as project
from graph_of_thoughts import controller, language_models, operations
try:
    from .src import utils
    from .src.parser import CommonsenseQAParser
    from .src.prompter import CommonsenseQAPrompter
except ImportError:
    from src import utils
    from src.parser import CommonsenseQAParser
    from src.prompter import CommonsenseQAPrompter


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def io_zeroshot():
    return io()


def cot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def cot_zeroshot() -> operations.GraphOfOperations:
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def cot_sc() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT-SC method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    num_branches = 5
    operations_graph = operations.GraphOfOperations()

    generate_operation = operations.Generate(1, num_branches)
    operations_graph.append_operation(generate_operation)
    operations_graph.append_operation(
        operations.ScoreByFrequency(ignore_none=True)
    )
    operations_graph.append_operation(operations.KeepBestN(1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def plan_solve() -> operations.GraphOfOperations:
    """
         Generates the Graph of Operations for the Plan and Solve method.

         :return: Graph of Operations
         :rtype: GraphOfOperations
         """
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def plan_solve_plus() -> operations.GraphOfOperations:
    """
         Generates the Graph of Operations for the Plan and Solve Plus method.

         :return: Graph of Operations
         :rtype: GraphOfOperations
         """
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


# def tot_style() -> operations.GraphOfOperations:
#     operations_graph = operations.GraphOfOperations()
#
#     operations_graph.append_operation(operations.Generate(1, 5))
#     operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
#     operations_graph.append_operation(operations.KeepBestN(1))
#     operations_graph.append_operation(operations.GroundTruth(utils.test_answer))
#
#     return operations_graph


def tot_base() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.
    ToT uses a wider tree, where on each level there are more branches.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 5))
    operations_graph.append_operation(operations.Score(1, False))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
    keep_best_1 = operations.KeepBestN(1)
    operations_graph.append_operation(keep_best_1)

    for _ in range(3):
        operations_graph.append_operation(operations.Generate(1, 5))
        operations_graph.append_operation(operations.Score(1, False))
        operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
        keep_best_2 = operations.KeepBestN(1)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    operations_graph.append_operation(operations.KeepBestN(1))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
) -> float:
    orig_budget = budget

    datasets_dir: Path = project.datasets_dir() / "CommonSenseQA"
    data_path = datasets_dir / "train_rand_split.jsonl"

    data = []
    if os.path.exists(data_path):  # test if exits
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    json_line = json.loads(line)
                    json_line["id"] = i
                    data.append(json_line)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {i}: {line}")  # print error
    else:
        logging.error("File does not exist.")

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder, exist_ok=True)
    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    for method in methods:
        # create a results directory for the method
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data in tqdm(selected_data, desc="Data"):

        logging.info(f"Running data {data['id']: }{data['question']}: {data['answerKey']}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            raw_path = os.path.join(
                os.path.dirname(__file__),
                "../../graph_of_thoughts/language_models/config.json")
            abs_path = os.path.abspath(raw_path)
            lm = language_models.ChatGPT(
                abs_path,
                model_name=lm_name,
                cache=True,
            )

            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                CommonsenseQAPrompter(),
                CommonsenseQAParser(),
                {
                    "original": utils.generate_question_string(data["question"]),
                    "ground_truth": data["answerKey"],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data['id']}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


def run_process(samples, approaches, budget, model):
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )
    return run(samples, approaches, budget, model)

def main_process_pool():
    budget = 30
    samples = []  # runs all the data
    approaches = [io_zeroshot, io, cot, cot_zeroshot, cot_sc, plan_solve, plan_solve_plus, tot_base]
    lm_name = "chatgpt"

    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = {
            executor.submit(run_process, samples, [approach], budget, lm_name): approach
            for approach in approaches
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                spent = future.result()
                logging.info(f"Task {task} spent {spent} out of {budget} budget.")
            except Exception as e:
                logging.error(f"Task {task} generated an exception: {e}")

def main():
    budget = 30
    samples = []  # runs all the data
    approaches = [io_zeroshot, io, cot, cot_zeroshot, cot_sc, plan_solve, plan_solve_plus, tot_base]
    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")


if __name__ == "__main__":
    main_process_pool()
